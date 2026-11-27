"""Source-level candidate experiment with protected host-side evaluation."""
import argparse, asyncio, json, sqlite3, hashlib, time
from pathlib import Path
from .evolution import tasks, evaluate, chat
from .meta import BASELINE, Program, propose
from .world import World
from forge import digest

async def rollout(task, program, model, sample):
    world=World(program.image); events=[]; tokens=0; cache=None
    messages=[{'role':'system','content':'Return JSON action read, patch (before SHA256, text), execute (code), or finish.'},{'role':'user','content':task['goal']+'; preserve exact solve(xs) signature.'}]
    start=time.perf_counter()
    try:
        await world.start(task['initial'])
        for step in range(8):
            context,allowed,use_cache=await program.hooks(messages,{'step':step,'goal':task['goal'],'constraint':'preserve exact solve(xs) signature'})
            context=context+[{'role':'user','content':'Allowed actions: '+', '.join(allowed)}]
            tick=time.perf_counter()
            action,usage=await chat(model,context); tokens+=usage.get('total_tokens',0)
            tool=action.get('tool'); executed=False
            if tool not in allowed: obs={'error':'disallowed action'}
            elif tool=='read':
                if use_cache and cache is not None: obs=cache
                else:
                    r=await world.read()
                    if r['exit']: raise RuntimeError('read failed')
                    obs={'source':r['stdout'],'sha256':digest(r['stdout'])}; cache=obs; executed=True
            elif tool=='patch':
                r=await world.read()
                if r['exit']: raise RuntimeError('read failed')
                if digest(r['stdout'])!=action.get('before'): obs={'error':'stale patch'}
                else:
                    r=await world.write(action.get('text')); cache=None; executed=True; obs={'applied':r['exit']==0}
            elif tool=='execute':
                obs=await world.execute(action['code']); cache=None; executed=True
            else: obs={}
            events.append(dict(action=action,observation=obs,context=context,executed=executed,elapsed_ms=(time.perf_counter()-tick)*1000))
            messages.extend([{'role':'assistant','content':json.dumps(action)},{'role':'user','content':json.dumps(obs)}])
            if tool=='finish': break
        r=await world.read()
        if r['exit']: raise RuntimeError('final read failed')
        source=r['stdout']
    finally: await world.close()
    import ast
    try: signature=any(isinstance(n,ast.FunctionDef) and n.name=='solve' and ast.unparse(n.args)=='xs' for n in ast.parse(source).body)
    except SyntaxError: signature=False
    verifier=World(program.image)
    try:
        await verifier.start(source)
        r=await verifier.execute('import json; from solution import solve; print(json.dumps(solve('+repr(task['args'])+')))')
        try: correct=r['exit']==0 and json.loads(r['stdout'])==task['expected']
        except ValueError: correct=False
    finally: await verifier.close()
    return dict(task_id=task['id'],sample=sample,model=model,harness_sha=program.sha,events=events,source=source,correct=correct,signature=signature,tokens=tokens,elapsed_ms=(time.perf_counter()-start)*1000)

async def run(a):
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(out/'experiment.sqlite')
    db.execute('CREATE TABLE episodes (phase TEXT, task TEXT, sample INTEGER, raw TEXT, evaluation TEXT, PRIMARY KEY(phase,task,sample))')
    sem=asyncio.Semaphore(a.workers)
    async def batch(phase,subset,program):
        async def job(t,i):
            async with sem:
                row=await rollout(t,program,a.model,i); ev=evaluate(row)
                with db: db.execute('INSERT INTO episodes VALUES (?,?,?,?,?)',(phase,t['id'],i,json.dumps(row),json.dumps(ev)))
                return ev
        return await asyncio.gather(*(job(t,i) for t in subset for i in range(a.samples)))
    ts=list(tasks()); discovery=[t for t in ts if t['split']=='discovery']; heldout=[t for t in ts if t['split']=='heldout']
    discovery=discovery[:a.tasks_per_split]; heldout=heldout[:a.tasks_per_split]
    baseline=Program(BASELINE,a.image)
    evidence=await batch('discovery_baseline',discovery,baseline)
    source=await propose(a.meta_model,evidence)
    candidate=Program(source,a.image)
    (out/'baseline.py').write_text(BASELINE); (out/'candidate.py').write_text(source)
    import difflib
    (out/'candidate.diff').write_text(''.join(difflib.unified_diff(BASELINE.splitlines(True),source.splitlines(True),fromfile='baseline.py',tofile='candidate.py')))
    def mean(rows): return {k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}
    report={'model':a.model,'meta_model':a.meta_model,'baseline_sha':baseline.sha,'candidate_sha':candidate.sha,'discovery_baseline':mean(evidence)}
    report['discovery_candidate']=mean(await batch('discovery_candidate',discovery,candidate))
    report['heldout_baseline']=mean(await batch('heldout_baseline',heldout,baseline))
    report['heldout_candidate']=mean(await batch('heldout_candidate',heldout,candidate))
    from .benchmark_stats import paired_report
    report['paired']=paired_report(db)
    report['episodes']=db.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]
    x,y=report['heldout_baseline'],report['heldout_candidate']
    report['eligible_for_review']=y['correctness']>=x['correctness'] and y['intent_preservation']>=x['intent_preservation'] and y['action_selection']>=x['action_selection'] and y['tool_operations']<=x['tool_operations'] and y['execution_ms']<=1.15*x['execution_ms'] and (y['intent_preservation']>x['intent_preservation'] or y['action_selection']>x['action_selection'])
    (out/'report.json').write_text(json.dumps(report,indent=2)); db.close(); print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--meta-model'); p.add_argument('--image',default='forgerl-world:local'); p.add_argument('--tasks-per-split',type=int,default=30); p.add_argument('--samples',type=int,default=2); p.add_argument('--workers',type=int,default=4); p.add_argument('--output',default='source-experiment')
    a=p.parse_args()
    if min(a.samples,a.workers,a.tasks_per_split)<1: p.error('positive samples and workers required')
    asyncio.run(run(a))
