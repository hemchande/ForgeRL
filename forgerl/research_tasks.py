"""Authored synthetic coding tasks with disjoint families and scripted requirement changes."""
def suite():
    specs=[
      ('discovery','dedup','Return unique values preserving first occurrence.','return list(set(xs))','Return unique values preserving LAST occurrence order.',[([2,1,2,3],[1,2,3]),([],[]),([1,2,1],[2,1]),([4,4],[4])]),
      ('discovery','intervals','Merge overlapping [start,end] intervals.','return sorted(xs)','Also merge touching intervals; return sorted lists.',[([[1,3],[3,5]],[[1,5]]),([],[]),([[5,6],[1,2]],[[1,2],[5,6]]),([[1,9],[2,3]],[[1,9]])]),
      ('discovery','ledger','Aggregate [account, amount] records into a dictionary.','return dict(xs)','Omit accounts whose final balance is zero.',[([['a',2],['a',-2]],{}),([],{}),([['a',2],['a',3]],{'a':5}),([['b',-1]],{'b':-1})]),
      ('heldout','toposort','Return a topological ordering of integer nodes from directed edges.','return sorted(set(v for e in xs for v in e))','Choose the smallest available node first; return [] on cycles.',[([[2,1],[3,1]],[2,3,1]),([],[]),([[1,2],[2,1]],[]),([[3,4],[1,4]],[1,3,4])]),
      ('heldout','windows','Return sums of adjacent pairs.','return xs','Use circular adjacent pairs, but return [] for fewer than two items.',[([1,2,3],[3,5,4]),([],[]),([7],[]),([-1,2],[1,1])]),
      ('heldout','inventory','Aggregate [sku, quantity] records.','return dict(xs)','Clamp final quantities to zero and omit zero entries.',[([['a',3],['a',-5]],{}),([],{}),([['b',2],['b',4]],{'b':6}),([['a',-2],['a',5]],{'a':3})])
    ]
    for split,name,goal,bug,update,cases in specs:
        yield dict(id=name,split=split,goal=goal,update=update,initial='def solve(xs):\n    '+bug+'\n',cases=cases)
