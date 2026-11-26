import argparse, asyncio, json
from forge import TASKS
from .pipeline import rollout, normalize, SYSTEM
async def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--samples',type=int,default=8); p.add_argument('--workers',type=int,default=4); p.add_argument('--output',default='model-rollouts.jsonl'); p.add_argument('--system-file'); a=p.parse_args()
    if min(a.samples,a.workers)<1: p.error('positive samples/workers required')
    system=open(a.system_file).read() if a.system_file else SYSTEM
    sem=asyncio.Semaphore(a.workers)
    async def job(t,i):
        async with sem:
            try: return await rollout(t,i,a.model,system=system)
            except Exception as e: return {'task_id':t,'sample':i,'error':str(e)}
    rows=normalize(await asyncio.gather(*(job(t[0],i) for t in TASKS for i in range(a.samples))))
    with open(a.output,'w') as f:
        for r in rows: f.write(json.dumps(r)+'\n')
if __name__=='__main__': asyncio.run(main())
