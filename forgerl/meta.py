"""Candidate source generation and isolated execution of harness hooks."""
import asyncio, base64, hashlib, json
from pathlib import Path
from .world import World
from .evolution import chat

BASELINE = Path(__file__).with_name('harness_program.py').read_text()

REFERENCE = '''def build_context(messages, state):
    return messages + [{"role":"user","content":"Persistent constraint: preserve exact solve(xs) signature."}]

def allowed_actions(state):
    return ["read", "patch", "finish"]

def cache_reads(state):
    return True
'''

async def propose(model, evidence):
    if not model: return REFERENCE
    response,_=await chat(model,[{'role':'system','content':'You are a harness engineer. Return JSON with source and hypothesis. Rewrite only the three Python hooks in the supplied module. Hooks receive JSON state and messages; return JSON-compatible values. No evaluator changes. Preserve function signatures.'},{'role':'user','content':json.dumps({'baseline':BASELINE,'discovery_evidence':evidence})}])
    source=response['source']
    if not isinstance(source,str) or len(source)>16000: raise ValueError('invalid candidate source')
    return source

class Program:
    def __init__(self, source, image):
        self.source=source; self.image=image; self.sha=hashlib.sha256(source.encode()).hexdigest()
    async def hooks(self, messages, state):
        # Separate sandbox: candidate never runs in the evaluator process.
        world=World(self.image)
        try:
            await world.start(self.source)
            payload=base64.b64encode(json.dumps([messages,state]).encode()).decode()
            code='import json,base64; import solution as h; m,s=json.loads(base64.b64decode('+repr(payload)+')); print(json.dumps([h.build_context(m,s),h.allowed_actions(s),h.cache_reads(s)]))'
            result=await world.execute(code)
            if result['exit']: raise RuntimeError('candidate hook failed')
            context,actions,cache=json.loads(result['stdout'])
            if not isinstance(context,list) or len(context)>100 or any(not isinstance(m,dict) or set(m)!={'role','content'} or m['role'] not in ('system','user','assistant') or not isinstance(m['content'],str) for m in context): raise ValueError('invalid context')
            if not isinstance(actions,list) or not set(actions)<= {'read','patch','execute','finish'} or 'finish' not in actions or type(cache)!=bool: raise ValueError('invalid hook output')
            return context,actions,cache
        finally: await world.close()
