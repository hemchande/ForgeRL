"""ForgeRL: synthetic coding rollouts, replay and reward export."""
import argparse, asyncio, hashlib, json, math, os, tempfile, uuid
from pathlib import Path

TASKS = [
 ('divide','Return floating point division.', 'def solve(a,b):\n    return a//b\n', 'def solve(a,b):\n    return a/b\n', [[[5,2],2.5],[[-3,2],-1.5]]),
 ('reverse','Reverse a string.', 'def solve(s):\n    return s\n', 'def solve(s):\n    return s[::-1]\n', [[['abc'],'cba'],[[''],'']]),
 ('unique','Remove duplicates preserving order.', 'def solve(xs):\n    return list(set(xs))\n', 'def solve(xs):\n    return list(dict.fromkeys(xs))\n', [[[[3,1,3,2]],[3,1,2]],[[[]],[]]])
]
def digest(s): return hashlib.sha256(s.encode()).hexdigest()

class World:
    def __init__(self, root, docker): self.root, self.docker = root, docker
    async def execute(self, code):
        name='forge-'+uuid.uuid4().hex
        cmd=['docker','run','--name',name,'--rm','--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','128m','--cpus','1','--pids-limit','32','--user',f'{os.getuid()}:{os.getgid()}','--mount',f'type=bind,src={self.root},dst=/workspace,readonly','-w','/workspace','python:3.12-slim','python','-B','-c',code] if self.docker else ['python','-B','-c',code]
        p=await asyncio.create_subprocess_exec(*cmd,cwd=self.root,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE,env={'PATH':os.environ['PATH']})
        async def drain(stream):
            data=bytearray()
            while chunk:=await stream.read(4096):
                data.extend(chunk[:max(0,8192-len(data))])
            return data.decode(errors='replace')
        readers=[asyncio.create_task(drain(s)) for s in (p.stdout,p.stderr)]
        try:
            await asyncio.wait_for(p.wait(),15)
            out,err=await asyncio.wait_for(asyncio.gather(*readers),2)
            return {'exit':p.returncode,'stdout':out,'stderr':err}
        except asyncio.TimeoutError:
            if p.returncode is None: p.kill(); await p.wait()
            if self.docker:
                cleanup=await asyncio.create_subprocess_exec('docker','rm','-f',name,stdout=asyncio.subprocess.DEVNULL,stderr=asyncio.subprocess.DEVNULL); await cleanup.wait()
            for r in readers: r.cancel()
            return {'timeout':True}
    async def step(self,a):
        f=self.root/'solution.py'
        if a['tool']=='read': return {'text':f.read_text()}
        if a['tool']=='patch':
            if digest(f.read_text())!=a['before']: return {'error':'stale patch'}
            f.write_text(a['text']); return {'sha256':digest(a['text'])}
        if a['tool']=='execute': return await self.execute(a['code'])
        raise ValueError('unknown tool')

async def verify(world,cases):
    results=[]
    for args,expected in cases:
        obs=await world.execute('import json\nfrom solution import solve\nprint(json.dumps(solve(*'+repr(args)+')))')
        if obs.get('timeout') or obs.get('exit') in (125,126,127): raise RuntimeError('verification infrastructure failure or timeout')
        try: passed=obs.get('exit')==0 and json.loads(obs['stdout'])==expected
        except (ValueError,KeyError): passed=False
        results.append(passed)
    return sum(results)/len(results),results

async def episode(task,index,docker,replay=None):
    name,prompt,initial,correct,cases=task
    with tempfile.TemporaryDirectory(prefix='forge-') as d:
        root=Path(d); (root/'solution.py').write_text(initial); world=World(root,docker)
        actions=replay if replay is not None else [{'tool':'read'},{'tool':'execute','code':'from solution import solve; print(solve(*'+repr(cases[0][0])+'))'},{'tool':'patch','before':digest(initial),'text':correct if index%2==0 else initial}]
        trace=[]
        for action in actions: trace.append({'action':action,'observation':await world.step(action)})
        reward,checks=await verify(world,cases)
        return {'schema_version':1,'episode_id':uuid.uuid4().hex,'task_id':name,'prompt':prompt,'initial_sha256':digest(initial),'policy':'scripted-fixture-v1','sample':index,'trace':trace,'reward':reward,'checks':checks,'final_source':(root/'solution.py').read_text(),'backend':'docker' if docker else 'trusted-local','token_logprobs':None}

async def run(args):
    sem=asyncio.Semaphore(args.workers)
    async def job(t,i):
        async with sem:
            try: return await episode(t,i,args.docker)
            except Exception as e: return {'task_id':t[0],'sample':i,'error':str(e)}
    rows=await asyncio.gather(*(job(t,i) for t in TASKS for i in range(args.samples)))
    for t in TASKS:
        group=[r for r in rows if r['task_id']==t[0] and 'reward' in r]; rewards=[r['reward'] for r in group]
        if not rewards: continue
        mean=sum(rewards)/len(rewards); std=math.sqrt(sum((x-mean)**2 for x in rewards)/len(rewards))
        for r in group: r.update(group_id=t[0],group_size=len(group),advantage=(r['reward']-mean)/(std+1e-8))
    Path(args.output).write_text(''.join(json.dumps(r)+'\n' for r in rows)); print(json.dumps({'episodes':len(rows),'valid':sum('reward' in r for r in rows),'output':args.output}))

async def replay(args):
    row=json.loads(Path(args.input).read_text().splitlines()[args.index]); task=next(t for t in TASKS if t[0]==row['task_id']); result=await episode(task,row['sample'],args.docker,[s['action'] for s in row['trace']]); print(json.dumps({'reward_match':result['reward']==row['reward'],'trace_match':result['trace']==row['trace']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    r=sub.add_parser('run'); r.add_argument('--samples',type=int,default=8); r.add_argument('--workers',type=int,default=4); r.add_argument('--output',default='rollouts.jsonl')
    q=sub.add_parser('replay'); q.add_argument('input'); q.add_argument('--index',type=int,default=0)
    for s in (r,q): s.add_argument('--docker',action='store_true'); s.add_argument('--trusted-local',action='store_true')
    a=p.parse_args()
    if a.docker==a.trusted_local: p.error('choose exactly one backend: --docker or --trusted-local')
    if a.command=='run' and (a.samples<1 or a.workers<1): p.error('positive samples/workers required')
    asyncio.run(run(a) if a.command=='run' else replay(a))
