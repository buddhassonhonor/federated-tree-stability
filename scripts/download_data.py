"""Download the four public input files and check their recorded SHA-256 hashes."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=root/'data');args=parser.parse_args()
records=json.loads((root/'data/sources.json').read_text())['files']
for item in records:
 target=args.data_dir/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():
  raw=target.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=item['sha256']:
   raise RuntimeError(f'Existing file differs from archived input: {target}; preserve it and choose another directory.')
 else:
  with urllib.request.urlopen(item['download_url'],timeout=60) as r:raw=r.read()
  if hashlib.sha256(raw).hexdigest()!=item['sha256']:
   raise RuntimeError(f'Upstream bytes differ from archived input: {item["path"]}')
  target.write_bytes(raw)
 print('Verified',item['path'],len(raw),'bytes')
