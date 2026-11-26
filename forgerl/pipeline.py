import json, hashlib, uuid, math
from forge import TASKS, digest
from .world import World
from .policy import ChatPolicy

SYSTEM='Return exactly one JSON object: {"tool":"read"}, {"tool":"patch","before":"SHA256 of current source","text":"complete replacement source"}, {"tool":"execute","code":"Python code"}, or {"tool":"finish"}. Work on solution.py in /workspace. No hidden tests are available.'
async def rollout(task_id, sample, model, backend='docker', image='forgerl-world:local', max_steps=12, namespace='default', system=SYSTEM):
    task=next(t for t in TASKS if t[0]==task_id)
    _,prompt,initial,_,cases=task
    world=World(image,backend,namespace); policy=ChatPolicy(model)
    messages=[{'role':'system','content':system},{'role':'user','content':prompt}]; trace=[]
    try:
        await world.start(initial)
        for _ in range(max_steps):
            action,raw,logprobs=await policy.act(messages)
            if not isinstance(action,dict): action={'tool':'invalid'}
            tool=action.get('tool'); obs={}
            if tool=='read':
                obs=await world.read(); obs['sha256']=digest(obs['stdout'])
            elif tool=='patch':
                current=await world.read()
                obs=await world.write(action.get('text')) if current['exit']==0 and digest(current['stdout'])==action.get('before') else {'error':'stale patch'}
            elif tool=='execute' and isinstance(action.get('code'),str): obs=await world.execute(action['code'])
            elif tool!='finish': obs={'error':'invalid action'}
            trace.append({'action':action,'observation':obs,'completion':raw,'logprobs':logprobs})
            messages.extend([{'role':'assistant','content':raw or ''},{'role':'user','content':json.dumps(obs)}])
            if tool=='finish': break
        final=await world.read()
    finally: await world.close()
    if final['exit']: raise RuntimeError('cannot read final source')
    verifier=World(image,backend,namespace); checks=[]
    try:
        await verifier.start(final['stdout'])
        for args,expected in cases:
            obs=await verifier.execute('import json; from solution import solve; print(json.dumps(solve(*'+repr(args)+')))')
            try: checks.append(obs['exit']==0 and json.loads(obs['stdout'])==expected)
            except ValueError: checks.append(False)
    finally: await verifier.close()
    return {'schema_version':2,'episode_id':uuid.uuid4().hex,'task_id':task_id,'sample':sample,'policy':model,'prompt':prompt,'system':system,'initial_sha256':digest(initial),'trace':trace,'reward':sum(checks)/len(checks),'checks':checks,'final_source':final['stdout'],'backend':backend,'termination':'finish' if trace and trace[-1]['action'].get('tool')=='finish' else 'step_limit'}

def normalize(rows):
    groups={}
    for r in rows:
        if 'reward' in r: groups.setdefault((r['task_id'],r['policy'],r.get('system','')),[]).append(r)
    for key,g in groups.items():
        mean=sum(r['reward'] for r in g)/len(g); std=math.sqrt(sum((r['reward']-mean)**2 for r in g)/len(g))
        for r in g: r.update(group_id=hashlib.sha256(repr(key).encode()).hexdigest()[:16],group_size=len(g),advantage=(r['reward']-mean)/(std+1e-8))
    return rows
