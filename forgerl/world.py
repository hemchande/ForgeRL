"""Persistent Docker or Kubernetes world per episode."""
import asyncio, uuid, base64

class World:
    def __init__(self, image='forgerl-world:local', backend='docker', namespace='default'):
        self.name='forge-'+uuid.uuid4().hex; self.image=image; self.backend=backend; self.namespace=namespace
    async def rpc(self,*args,timeout=30):
        p=await asyncio.create_subprocess_exec(*args,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        async def drain(s):
            b=bytearray()
            while chunk:=await s.read(4096): b.extend(chunk[:max(0,32768-len(b))])
            return b.decode(errors='replace')
        readers=[asyncio.create_task(drain(s)) for s in (p.stdout,p.stderr)]
        try:
            await asyncio.wait_for(p.wait(),timeout)
            out,err=await asyncio.wait_for(asyncio.gather(*readers),3)
            return {'exit':p.returncode,'stdout':out,'stderr':err}
        except BaseException:
            if p.returncode is None: p.kill(); await p.wait()
            for r in readers: r.cancel()
            raise
    async def start(self,source):
        if self.backend=='docker':
            r=await self.rpc('docker','run','-d','--name',self.name,'--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','256m','--cpus','1','--pids-limit','64','--tmpfs','/workspace:rw,size=32m,mode=1777','--tmpfs','/tmp:rw,size=16m,mode=1777',self.image)
        else:
            import json
            spec={'apiVersion':'v1','kind':'Pod','metadata':{'name':self.name,'labels':{'app':'forgerl-world'}},'spec':{'automountServiceAccountToken':False,'restartPolicy':'Never','activeDeadlineSeconds':900,'containers':[{'name':'world','image':self.image,'securityContext':{'runAsUser':10001,'runAsNonRoot':True,'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,'capabilities':{'drop':['ALL']},'seccompProfile':{'type':'RuntimeDefault'}},'resources':{'limits':{'cpu':'1','memory':'256Mi'}},'volumeMounts':[{'name':'workspace','mountPath':'/workspace'},{'name':'tmp','mountPath':'/tmp'}]}],'securityContext':{'fsGroup':10001},'volumes':[{'name':'workspace','emptyDir':{'sizeLimit':'32Mi'}},{'name':'tmp','emptyDir':{'sizeLimit':'16Mi'}}]}}
            r=await self.rpc('kubectl','-n',self.namespace,'run',self.name,'--image',self.image,'--restart=Never','--overrides',json.dumps(spec))
            if r['exit']==0: r=await self.rpc('kubectl','-n',self.namespace,'wait','--for=condition=Ready','pod/'+self.name,'--timeout=90s',timeout=100)
        if r['exit']: raise RuntimeError(str(r))
        r=await self.write(source)
        if r['exit']: raise RuntimeError(str(r))
    async def execute(self,code):
        prefix=['docker','exec','-w','/workspace',self.name] if self.backend=='docker' else ['kubectl','-n',self.namespace,'exec',self.name,'--']
        r=await self.rpc(*prefix,'python','-B','-c',code,timeout=20)
        if r['exit'] in (125,126,127): raise RuntimeError(str(r))
        return r
    async def write(self,source):
        if not isinstance(source,str) or len(source)>64000: raise ValueError('invalid source')
        return await self.execute('from pathlib import Path; import base64; Path("/workspace/solution.py").write_bytes(base64.b64decode('+repr(base64.b64encode(source.encode()).decode())+'))')
    async def read(self): return await self.execute('from pathlib import Path; print(Path("/workspace/solution.py").read_text(),end="")')
    async def close(self):
        args=['docker','rm','-f',self.name] if self.backend=='docker' else ['kubectl','-n',self.namespace,'delete','pod',self.name,'--ignore-not-found','--wait=false']
        await self.rpc(*args)
