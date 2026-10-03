"""Verify the versioned artifact's SHA-256 manifest."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'release-manifest.json').read_text())
for item in manifest['files']:
 p=root/item['path'];assert p.is_file(),item['path']
 raw=p.read_bytes()
 assert len(raw)==item['size'],item['path']
 assert hashlib.sha256(raw).hexdigest()==item['sha256'],item['path']
print('Verified',len(manifest['files']),'files for version',manifest['version'])
