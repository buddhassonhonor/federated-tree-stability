"""Derive CSV summaries from archived results without rerunning models."""
from pathlib import Path
import csv,json,statistics
root=Path(__file__).resolve().parents[1]
out=root/'results';out.mkdir(exist_ok=True)
data=json.loads((root/'exp/supplementary_results.json').read_text())
with (out/'ood_accuracy.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['dataset','method','n_seeds','mean_ood_accuracy','population_std'])
 for ds,seed_results in sorted(data['results'].items()):
  for method in ['fiti','flse','local','tuned_cart','tuned_rf','tuned_et','tuned_gbdt','tuned_xgb','fedrf_standard','fedrf_invariant','ab_shuffled','ab_rand_group','ab_abs_range','ab_mad']:
   vals=[r[method][0]['ood_accuracy'] for _,r in sorted(seed_results.items(),key=lambda item:int(item[0]))]
   w.writerow([ds,method,len(vals),statistics.mean(vals),statistics.pstdev(vals)])
p=json.loads((root/'exp/priority2_results.json').read_text())
with (out/'client_weighting_voting.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.writer(f);w.writerow(['dataset','method','n_seeds','mean_ood_accuracy','population_std'])
 for ds,methods in p.items():
  for method,record in methods.items():
   vals=record['vals'];w.writerow([ds,method,len(vals),statistics.mean(vals),statistics.pstdev(vals)])
print('Exported accuracy summaries to results/.')
