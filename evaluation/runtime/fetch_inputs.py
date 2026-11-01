"""Explicit download of pinned public inputs; not executed by tests or imports."""
import hashlib
import urllib.request
import yaml
from common import ROOT, DESIGN


def fetch(url, path, expected=None):
    if path.exists():
        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Existing input hash mismatch; inspect instead of overwriting')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix+'.partial')
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, partial.open('xb') as out:
        for chunk in iter(lambda:response.read(1024*1024), b''):
            h.update(chunk)
            out.write(chunk)
    if expected and h.hexdigest() != expected:
        raise ValueError('Downloaded hash mismatch; partial retained for inspection')
    partial.replace(path)


if __name__ == '__main__':
    cfg = yaml.safe_load((DESIGN/'experiment.yaml').read_text())
    data = cfg['data']['public_binary']
    bert = cfg['bert']
    fetch('https://huggingface.co/datasets/{}/resolve/{}/{}'.format(
        data['repo_id'],data['revision'],data['filename']), ROOT/'data/raw/CFunSet.json', data['expected_sha256'])
    for remote, local in [('vocab.txt','bert_vocab.txt'),('tokenizer_config.json','bert_tokenizer_config.json')]:
        fetch('https://huggingface.co/{}/resolve/{}/{}'.format(bert['repo_id'],bert['revision'],remote),ROOT/'data/raw'/local)
    print('Pinned public inputs saved. Run prepare_data.py to audit and split.')
