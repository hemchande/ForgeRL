"""OpenAI-compatible chat policy; never execute model code on the host."""
import json, os, urllib.request, asyncio

class ChatPolicy:
    def __init__(self, model, endpoint=None):
        self.model=model
        self.endpoint=endpoint or os.getenv('MODEL_URL','http://localhost:8000/v1/chat/completions')
    async def act(self, messages):
        def call():
            body=json.dumps({'model':self.model,'messages':messages,'temperature':0.7,'max_tokens':1024,'logprobs':True}).encode()
            headers={'Content-Type':'application/json'}
            if os.getenv('MODEL_API_KEY'): headers['Authorization']='Bearer '+os.environ['MODEL_API_KEY']
            with urllib.request.urlopen(urllib.request.Request(self.endpoint,body,headers),timeout=90) as response:
                choice=json.load(response)['choices'][0]
            content=choice['message']['content']
            try: action=json.loads(content)
            except (ValueError,TypeError): action={'tool':'invalid'}
            return action,content,choice.get('logprobs')
        return await asyncio.to_thread(call)
