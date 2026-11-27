"""Task-clustered paired bootstrap; descriptive intervals, not a promotion proof."""
import json, random, statistics

def paired_report(db, draws=2000):
    rows=db.execute("SELECT a.task,a.evaluation,b.evaluation FROM episodes a JOIN episodes b ON a.task=b.task AND a.sample=b.sample WHERE a.phase='heldout_baseline' AND b.phase='heldout_candidate'").fetchall()
    if not rows: raise ValueError('no matched heldout episodes')
    metrics=['correctness','intent_preservation','action_selection','execution_ms','tool_operations','tokens']
    groups={}
    for task,a,b in rows:
        a,b=json.loads(a),json.loads(b)
        groups.setdefault(task,[]).append({k:b[k]-a[k] for k in metrics})
    rng=random.Random(42); report={}
    for k in metrics:
        values=[statistics.mean(r[k] for r in g) for g in groups.values()]
        boot=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(draws))
        report[k]={'delta':statistics.mean(values),'bootstrap_95':[boot[int(.025*draws)],boot[min(draws-1,int(.975*draws))]]}
    return {'pairs':len(rows),'task_clusters':len(groups),'metrics':report,'warning':'Small synthetic task families are correlated; intervals are descriptive, not evidence of broad generalization.'}
