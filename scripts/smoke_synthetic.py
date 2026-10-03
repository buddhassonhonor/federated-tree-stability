"""Check the deterministic synthetic generator and primary tree implementation."""
from pathlib import Path
import sys,json
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.run_supplementary_experiments import get_synthetic_data,predict_simple_tree
from src.run_revision_experiments import federated_train_fiti_robust
nodes,_,_,x,y,_=get_synthetic_data(0)
again=get_synthetic_data(0)
assert np.array_equal(nodes[0][0],again[0][0][0])
artifact=np.load(root/'data/synthetic_seed0.npz',allow_pickle=False)
assert np.array_equal(x,artifact['ood_X']) and np.array_equal(y,artifact['ood_y'])
for i,(xi,yi) in enumerate(nodes):
 assert np.array_equal(xi,artifact[f'train_X_{i}']) and np.array_equal(yi,artifact[f'train_y_{i}'])
expected=json.loads((root/'exp/supplementary_results.json').read_text())['results']['Synthetic']['0']
for method,penalty in [('flse',0),('fiti',expected['fiti'][0]['lambda'])]:
 tree=federated_train_fiti_robust(nodes,max_depth=3,lam=penalty)
 pred=predict_simple_tree(tree,x);accuracy=float(np.mean(pred==y))
 target=expected[method][0]['ood_accuracy']
 assert abs(accuracy-target)<1e-12,(method,accuracy,target)
 print(method,'seed 0 OOD accuracy:',accuracy,'matches archived result')
print('Synthetic generator, data and seed-0 trees verified.')
