"""Multi-model source ablations with protected evaluation and durable failed episodes."""
import argparse, asyncio, json, sqlite3, time, statistics, difflib
from pathlib import Path
from .research_tasks import suite
from .research_eval import verify,evaluate,semantic_judge
from .meta import Program,BASELINE,propose
from .evolution import chat
from .world import World
from forge import digest

def variants():
    context=BASELINE.replace('return messages','return messages + [{"role":"user","content":state["constraint"]+" Latest requirement: "+state["goal"]}]')
    cache=BASELINE.replace('return False','return True')
    # Keep execution available: useful tests should not be prohibited.
    action=BASELINE.replace('return messages','return messages + [{"role":"user","content":"Read before patching; test relevant behavior; finish only when requirements are satisfied."}]')
    return {'baseline':BASELINE,'context':context,'cache':cache,'action_guidance':action,'combined':context.replace('return False','return True')}

def valid_action(a):
    if not isinstance(a,dict): return False
    t=a.get('tool')
    if t in ('read','finish'): return True
    if t=='patch': return isinstance(a.get('before'),str) and isinstance(a.get('text'),str) and len(a['text'])<=24000
    if t=='execute': return isinstance(a.get('code'),str) and len(a['code'])<=8000
    return False

async def rollout(task,program,model,sample):
    start=time.perf_counter(); events=[]; tokens=0; delivered=False; cache=None; checks=[]; signature=False; status='complete'; error=None
    goal=task['goal']; source=task['initial']
    messages=[{'role':'system','content':'Return one JSON object: read {"tool":"read"}, patch {"tool":"patch","before":"SHA256 from read","text":"complete source"}, execute {"tool":"execute","code":"Python test code"}, finish {"tool":"finish"}. Preserve exact solve(xs) signature.'},{'role':'user','content':goal}]
    world=World(program.image)
    try:
        await world.start(source)
        for step in range(10):
            if step==2:
                goal=task['update']; delivered=True
                messages.append({'role':'user','content':'Requirement update: '+goal})
            context,allowed,use_cache=await program.hooks(messages,{'step':step,'goal':goal,'constraint':'Preserve exact solve(xs) signature.'})
            context=context+[{'role':'user','content':'Allowed tools: '+', '.join(allowed)}]
            tick=time.perf_counter(); executed=False
            try:
                a,usage=await chat(model,context); tokens+=usage.get('total_tokens',0) or 0
            except (ValueError,TypeError,KeyError) as exc:
                a={'tool':'invalid'}; obs={'error':'malformed response: '+type(exc).__name__}
            else:
                if not valid_action(a) or a['tool'] not in allowed:
                    a=a if isinstance(a,dict) else {'tool':'invalid'}
                    obs={'error':'invalid or disallowed action'}
                elif a['tool']=='read':
                    if use_cache and cache is not None: obs=cache
                    else:
                        r=await world.read()
                        if r['exit']: raise RuntimeError('read failed')
                        obs={'source':r['stdout'],'sha256':digest(r['stdout'])}; cache=obs; executed=True
                elif a['tool']=='patch':
                    r=await world.read()
                    if r['exit']: raise RuntimeError('read failed')
                    if digest(r['stdout'])!=a['before']: obs={'error':'stale patch'}
                    else:
                        r=await world.write(a['text']); cache=None; executed=True
                        obs={'applied':True} if r['exit']==0 else {'error':'write failed'}
                elif a['tool']=='execute':
                    obs=await world.execute(a['code']); cache=None; executed=True
                    if obs['exit']: obs['error']='execution failed'
                elif not delivered: obs={'error':'user update pending; continue'}
                else: obs={}
            events.append(dict(step=step,action=a,observation=obs,context=context,executed=executed,elapsed_ms=(time.perf_counter()-tick)*1000))
            messages.extend([{'role':'assistant','content':json.dumps(a)},{'role':'user','content':json.dumps(obs)}])
            if a.get('tool')=='finish' and delivered and 'error' not in obs: break
        else: status='step_limit'
        r=await world.read()
        if r['exit']: raise RuntimeError('final read failed')
        source=r['stdout']
        checks,signature=await verify(source,task,program.image)
    except Exception as exc:
        status='error'; error=type(exc).__name__+': '+str(exc)[:500]
    finally:
        try: await world.close()
        except Exception as exc: status='error'; error='cleanup: '+type(exc).__name__
    return dict(task_id=task['id'],sample=sample,model=model,harness_sha=program.sha,events=events,source=source,checks=checks,signature=signature,update_delivered=delivered,tokens=tokens,elapsed_ms=(time.perf_counter()-start)*1000,status=status,error=error)

async def run(a):
    out=Path(a.output); out.mkdir(parents=True,exist_ok=False)
    db=sqlite3.connect(out/'research.sqlite')
    db.execute('CREATE TABLE episodes(model TEXT, variant TEXT, split TEXT, task TEXT, sample INTEGER, raw TEXT, evaluation TEXT, PRIMARY KEY(model,variant,split,task,sample))')
    sem=asyncio.Semaphore(a.workers); tasks=list(suite()); sources=variants()
    async def batch(model,name,split):
        p=Program(sources[name],a.image)
        async def job(t,i):
            async with sem:
                row=await rollout(t,p,model,i); ev=evaluate(row)
                if a.judge:
                    try: ev['semantic']=await semantic_judge(chat,a.judge,row)
                    except Exception as exc: ev['judge_error']=type(exc).__name__
                with db: db.execute('INSERT INTO episodes VALUES (?,?,?,?,?,?,?)',(model,name,split,t['id'],i,json.dumps(row),json.dumps(ev)))
                return ev
        return await asyncio.gather(*(job(t,i) for t in tasks if t['split']==split for i in range(a.samples)))
    models=list(dict.fromkeys(a.models)); report={'models':models,'discovery_model':models[0],'results':{},'candidate_error':None}
    evidence=await batch(models[0],'baseline','discovery')
    if a.meta_model:
        try: sources['generated']=await propose(a.meta_model,evidence)
        except Exception as exc: report['candidate_error']=type(exc).__name__+': '+str(exc)[:300]
    # Freeze all candidates before any held-out results are collected.
    for name,source in sources.items():
        (out/(name+'.py')).write_text(source)
        (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(BASELINE.splitlines(True),source.splitlines(True))))
    metrics=['correctness','case_pass_rate','intent_preservation','action_selection','execution_ms','tokens','tool_operations','failed']
    for model in models:
        report['results'][model]={}
        for name in sources:
            rows=await batch(model,name,'heldout')
            report['results'][model][name]={k:statistics.mean(r[k] for r in rows) for k in metrics}
        baseline=report['results'][model]['baseline']
        for name,r in report['results'][model].items():
            r['delta']={k:r[k]-baseline[k] for k in metrics}
    report['episodes']=db.execute('SELECT COUNT(*) FROM episodes').fetchone()[0]
    (out/'report.json').write_text(json.dumps(report,indent=2)); db.close(); print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--models',nargs='+',required=True); p.add_argument('--meta-model'); p.add_argument('--judge'); p.add_argument('--image',default='forgerl-world:local'); p.add_argument('--samples',type=int,default=2); p.add_argument('--workers',type=int,default=2); p.add_argument('--output',default='research-run')
    a=p.parse_args()
    if min(a.samples,a.workers)<1: p.error('positive samples and workers required')
    asyncio.run(run(a))
