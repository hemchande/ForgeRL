"""Temporal activities and workflow; start workers independently."""
import asyncio, argparse, json
from datetime import timedelta
from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.common import RetryPolicy
with workflow.unsafe.imports_passed_through():
    from .pipeline import rollout, normalize
    from forge import TASKS

@activity.defn
async def run_episode(config:dict)->dict:
    return await rollout(**config)

@workflow.defn
class Batch:
    @workflow.run
    async def run(self, config:dict)->list:
        jobs=config['jobs']; rows=[]; width=config.get('concurrency',4)
        for i in range(0,len(jobs),width):
            results=await asyncio.gather(*(workflow.execute_activity(run_episode,j,start_to_close_timeout=timedelta(minutes=20),retry_policy=RetryPolicy(maximum_attempts=2)) for j in jobs[i:i+width]),return_exceptions=True)
            for j,r in zip(jobs[i:i+width],results): rows.append({'task_id':j['task_id'],'sample':j['sample'],'error':str(r)} if isinstance(r,BaseException) else r)
        return rows

async def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['worker','submit']); p.add_argument('--address',default='localhost:7233'); p.add_argument('--model',default='local-model'); p.add_argument('--samples',type=int,default=8); p.add_argument('--concurrency',type=int,default=4); p.add_argument('--backend',choices=['docker','kubernetes'],default='docker'); p.add_argument('--image',default='forgerl-world:local'); p.add_argument('--namespace',default='default'); p.add_argument('--output',default='distributed.jsonl'); a=p.parse_args()
    if min(a.samples,a.concurrency)<1: p.error('positive samples/concurrency required')
    c=await Client.connect(a.address)
    if a.mode=='worker':
        await Worker(c,task_queue='forgerl',workflows=[Batch],activities=[run_episode],max_concurrent_activities=a.concurrency).run()
    else:
        import uuid
        jobs=[dict(task_id=t[0],sample=i,model=a.model,backend=a.backend,image=a.image,namespace=a.namespace) for t in TASKS for i in range(a.samples)]
        rows=await c.execute_workflow(Batch.run,{'jobs':jobs,'concurrency':a.concurrency},id='forge-'+uuid.uuid4().hex,task_queue='forgerl')
        with open(a.output,'w') as f:
            for r in normalize(rows): f.write(json.dumps(r)+'\n')
if __name__=='__main__': asyncio.run(main())
