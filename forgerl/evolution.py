"""Paired harness experiments. Fixture mode is a deterministic simulation, not an LLM benchmark."""
import argparse, asyncio, hashlib, json, os, random, sqlite3, time, urllib.request
from pathlib import Path
from dataclasses import dataclass, asdict
from forge import digest
from .world import World

@dataclass(frozen=True)
class Harness:
    persist_constraints: bool = False
    restrict_actions: bool = False
    cache_reads: bool = False

def tasks():
    # Disjoint task families, not merely different random seeds.
    for split, families in [('discovery', ['sum','max','min']), ('heldout',['product','count','last'])]:
        for family in families:
            expr={'sum':'sum(xs)','max':'max(xs)','min':'min(xs)','product':'__import__("math").prod(xs)','count':'len(xs)','last':'xs[-1]'}[family]
            for i in range(10):
                xs=[i+2,i+3,i+5]
                expected={'sum':sum(xs),'max':max(xs),'min':min(xs),'product':xs[0]*xs[1]*xs[2],'count':len(xs),'last':xs[-1]}[family]
                yield dict(id=f'{family}-{i}',split=split,goal=f'Return {family} of xs',initial='def solve(xs):\n    return None\n',correct=f'def solve(xs):\n    return {expr}\n',args=xs,expected=expected)

def fixture_action(task, config, step, source):
    if step==0 or (step==1 and not config.cache_reads): return {'tool':'read'}
    patch_step=1 if config.cache_reads else 2
    if step==patch_step:
        source_new=task['correct'] if config.persist_constraints else task['correct'].replace('solve(xs)','solve(xs, extra=None)')
        return {'tool':'patch','before':digest(source),'text':source_new}
    if step==patch_step+1 and not config.restrict_actions: return {'tool':'execute','code':'print("unnecessary diagnostic")'}
    return {'tool':'finish'}

async def chat(model,messages):
    def call():
        headers={'Content-Type':'application/json'}
        if os.getenv('MODEL_API_KEY'): headers['Authorization']='Bearer '+os.environ['MODEL_API_KEY']
        body=json.dumps(dict(model=model,messages=messages,temperature=0,max_tokens=1024)).encode()
        with urllib.request.urlopen(urllib.request.Request(os.environ['MODEL_URL'],body,headers),timeout=90) as r:
            response=json.load(r)
        return json.loads(response['choices'][0]['message']['content']),response.get('usage',{})
    return await asyncio.to_thread(call)

async def episode(task, config, model, image, sample):
    fixture=model=='fixture'
    start=time.perf_counter(); source=task['initial']; events=[]; tokens=0; cache=None
    world=None if fixture else World(image)
    messages=[{'role':'system','content':'Return JSON actions read, patch (before SHA256 and text), execute (code), or finish. Preserve exact solve(xs) signature.'},{'role':'user','content':task['goal']+'; preserve exact solve(xs) signature.'}]
    try:
        if world: await world.start(source)
        for step in range(8):
            if config.persist_constraints: messages.append({'role':'user','content':'Persistent constraint: preserve exact solve(xs) signature.'})
            if config.restrict_actions: messages.append({'role':'user','content':'Available tools: read, patch, finish. Do not execute diagnostics.'})
            tick=time.perf_counter()
            if fixture: action=fixture_action(task,config,step,source)
            else:
                action,usage=await chat(model,messages); tokens+=usage.get('total_tokens',0)
            tool=action.get('tool'); executed=False; obs={}
            if tool=='read':
                if config.cache_reads and cache is not None: obs=cache
                else:
                    if world:
                        result=await world.read()
                        if result['exit']: raise RuntimeError('read failed')
                        source=result['stdout']
                    obs={'source':source,'sha256':digest(source)}; cache=obs; executed=True
            elif tool=='patch' and action.get('before')==digest(source) and isinstance(action.get('text'),str):
                if world:
                    result=await world.write(action['text'])
                    if result['exit']: raise RuntimeError('write failed')
                source=action['text']; cache=None; executed=True; obs={'applied':True}
            elif tool=='execute' and not config.restrict_actions and isinstance(action.get('code'),str):
                obs=await world.execute(action['code']) if world else {'stdout':'fixture diagnostic'}; cache=None; executed=True
            elif tool!='finish': obs={'error':'invalid or disallowed action'}
            events.append(dict(step=step,action=action,observation=obs,context=[dict(m) for m in messages],executed=executed,elapsed_ms=(time.perf_counter()-tick)*1000))
            messages.extend([{'role':'assistant','content':json.dumps(action)},{'role':'user','content':json.dumps(obs)}])
            if tool=='finish': break
        if world:
            result=await world.read()
            if result['exit']: raise RuntimeError('final read failed')
            source=result['stdout']
    finally:
        if world: await world.close()
    import ast
    try:
        tree=ast.parse(source)
        signature=any(isinstance(n,ast.FunctionDef) and n.name=='solve' and ast.unparse(n.args)=='xs' for n in tree.body)
    except SyntaxError: signature=False
    if fixture: correct=source in (task['correct'],task['correct'].replace('solve(xs)','solve(xs, extra=None)'))
    else:
        verifier=World(image)
        try:
            await verifier.start(source)
            result=await verifier.execute('import json; from solution import solve; print(json.dumps(solve('+repr(task['args'])+')))')
            try: correct=result['exit']==0 and json.loads(result['stdout'])==task['expected']
            except ValueError: correct=False
        finally: await verifier.close()
    return dict(task_id=task['id'],split=task['split'],sample=sample,model=model,harness=asdict(config),initial_sha256=digest(task['initial']),events=events,source=source,correct=correct,signature=signature,tokens=tokens,elapsed_ms=(time.perf_counter()-start)*1000)

def evaluate(row):
    events=row['events']; reads=[e for e in events if e['action']['tool']=='read']
    redundant=sum(a['observation']==b['observation'] for a,b in zip(reads,reads[1:]))
    diagnostics=sum(e['action']['tool']=='execute' and 'unnecessary diagnostic' in e['action'].get('code','') for e in events)
    invalid=sum('error' in e['observation'] for e in events)
    return dict(correctness=int(row['correct']),intent_preservation=int(row['signature']),action_selection=1-(redundant+diagnostics+invalid)/max(1,len(events)),redundant_reads=redundant,diagnostics=diagnostics,invalid_actions=invalid,execution_ms=row['elapsed_ms'],tool_operations=sum(e['executed'] for e in events),tokens=row['tokens'])

def diagnose(records):
    # Deterministic reference meta-controller; candidate changes are allowlisted.
    return Harness(any(not e['intent_preservation'] for e in records),any(e['diagnostics'] for e in records),any(e['redundant_reads'] for e in records))

async def run(args):
    db=sqlite3.connect(args.output)
    db.execute('CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, phase TEXT, raw TEXT, evaluation TEXT, evaluator_version TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY, report TEXT)')
    sem=asyncio.Semaphore(args.workers)
    async def batch(phase, subset, config):
        async def job(t,i):
            async with sem: return await episode(t,config,args.model,args.image,i)
        rows=await asyncio.gather(*(job(t,i) for t in subset for i in range(args.samples)))
        evaluations=[evaluate(r) for r in rows]
        with db:
            db.executemany('INSERT INTO episodes(phase,raw,evaluation,evaluator_version) VALUES (?,?,?,?)',[(phase,json.dumps(r),json.dumps(e),'1') for r,e in zip(rows,evaluations)])
        return evaluations
    all_tasks=list(tasks()); discovery=[t for t in all_tasks if t['split']=='discovery']; heldout=[t for t in all_tasks if t['split']=='heldout']
    baseline=Harness()
    evidence=await batch('discovery_baseline',discovery,baseline)
    candidate=diagnose(evidence)
    if args.meta_model:
        proposal,_=await chat(args.meta_model,[{'role':'system','content':'Diagnose discovery evaluations. Return JSON with exactly three boolean keys: persist_constraints, restrict_actions, cache_reads. Select only justified interventions.'},{'role':'user','content':json.dumps(evidence)}])
        if set(proposal)!=set(asdict(baseline)) or any(type(v)!=bool for v in proposal.values()): raise ValueError('invalid meta-agent proposal')
        candidate=Harness(**proposal)
    def summary(rows): return {k:sum(e[k] for e in rows)/len(rows) for k in rows[0]}
    report={'mode':args.model,'meta_model':args.meta_model,'candidate':asdict(candidate),'discovery_baseline':summary(evidence)}
    report['discovery_candidate']=summary(await batch('discovery_candidate',discovery,candidate))
    # Held-out evidence is never passed to the proposer.
    report['heldout_baseline']=summary(await batch('heldout_baseline',heldout,baseline))
    report['heldout_candidate']=summary(await batch('heldout_candidate',heldout,candidate))
    report['episodes']=4*len(discovery)*args.samples
    a,b=report['heldout_baseline'],report['heldout_candidate']
    report['eligible_for_review']=b['correctness']>=a['correctness'] and b['intent_preservation']>=a['intent_preservation'] and b['action_selection']>=a['action_selection'] and b['tool_operations']<=a['tool_operations'] and (b['intent_preservation']>a['intent_preservation'] or b['action_selection']>a['action_selection'])
    with db: db.execute('INSERT INTO experiments(report) VALUES (?)',(json.dumps(report),))
    db.close(); print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',default='fixture'); p.add_argument('--meta-model'); p.add_argument('--image',default='forgerl-world:local'); p.add_argument('--samples',type=int,default=2); p.add_argument('--workers',type=int,default=4); p.add_argument('--output',default='evolution.sqlite')
    args=p.parse_args()
    if min(args.samples,args.workers)<1: p.error('positive samples and workers required')
    asyncio.run(run(args))
