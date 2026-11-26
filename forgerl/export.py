"""Export successful model conversations for SFT; preserve RL traces separately."""
import argparse,json
from pathlib import Path
from .pipeline import SYSTEM

def export(rows):
    for r in rows:
        if r.get('reward')!=1 or r.get('schema_version')!=2: continue
        messages=[{'role':'system','content':r.get('system',SYSTEM)},{'role':'user','content':r['prompt']}]
        for s in r['trace']:
            messages.append({'role':'assistant','content':s['completion'] or ''})
            if s['action'].get('tool')!='finish': messages.append({'role':'user','content':json.dumps(s['observation'])})
        if messages[-1]['role']=='assistant': yield {'messages':messages,'episode_id':r['episode_id']}
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('input'); p.add_argument('output'); a=p.parse_args()
    Path(a.output).write_text(''.join(json.dumps(r)+'\n' for r in export(map(json.loads,Path(a.input).read_text().splitlines()))))
