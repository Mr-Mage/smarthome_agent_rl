"""Archive only explicitly named historical transfer files, retaining byte identity."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
archive = ROOT / 'work/archive/transfers'
archive.mkdir(parents=True, exist_ok=True)
if not archive.resolve().is_relative_to(ROOT.resolve()):
    raise RuntimeError('Archive must remain inside the project')
records = []
for name in ('final-source.tar.gz', 'logs-evidence.tar.gz', 'official-source.tar.gz',
             'project-source.tar.gz', 'runs-evidence.tar.gz'):
    source = ROOT / name
    if not source.exists():
        continue
    if not source.resolve().is_relative_to(ROOT.resolve()) or not source.is_file():
        raise RuntimeError('Unexpected transfer path')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    target = archive / name
    if target.exists():
        raise FileExistsError('Never replace an existing archive')
    shutil.move(str(source), str(target))
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Archive identity mismatch')
    records.append({'original': name, 'archived': str(target.relative_to(ROOT)), 'sha256': digest})
receipt = archive / 'archive.json'
if records:
    if receipt.exists():
        raise FileExistsError('Do not overwrite a previous archive receipt')
    receipt.write_text(json.dumps({'files': records, 'deleted_unique_evidence': False}, indent=2), encoding='utf-8')
print(json.dumps({'archived_files': len(records), 'receipt': str(receipt)}))
