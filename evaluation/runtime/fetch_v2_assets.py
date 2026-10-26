"""Public immutable assets only; no access-gate acceptance or model execution."""
import argparse
import concurrent.futures, hashlib, json, os, time
from pathlib import Path
import requests
ROOT=None

def fetch(item):
    dest=ROOT/item['destination'];dest.parent.mkdir(parents=True,exist_ok=True)
    def digest(p):
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        return h.hexdigest()
    if dest.exists():
        if dest.stat().st_size!=item['size'] or (item.get('sha256') and digest(dest)!=item['sha256']):raise ValueError('Existing mismatch '+dest.name)
        return
    part=dest.with_name(dest.name+'.partial')
    for attempt in range(8):
        try:
            start=part.stat().st_size if part.exists() else 0
            if start>item['size']:raise ValueError('Partial larger than expected')
            if start<item['size']:
                url=item['url']+('?resume='+str(start))
                with requests.get(url,headers={'Range':f'bytes={start}-'} if start else {},stream=True,timeout=(30,120)) as r:
                    r.raise_for_status()
                    if start and r.status_code==206:
                        if not r.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                            raise ValueError('Incorrect range boundary')
                        mode='ab'
                    elif r.status_code==200:
                        mode='wb'
                    else:raise ValueError('Unexpected HTTP response')
                    with part.open(mode) as f:
                        for chunk in r.iter_content(1024*1024):f.write(chunk)
            if part.stat().st_size!=item['size']:raise ValueError('size mismatch')
            h=digest(part)
            if item.get('sha256') and h!=item['sha256']:raise ValueError('hash mismatch')
            part.replace(dest)
            print(json.dumps({'complete':item['destination'],'bytes':item['size'],'sha256':h}),flush=True)
            return
        except Exception as e:
            print(json.dumps({'retry':item['destination'],'attempt':attempt,'error':type(e).__name__}),flush=True)
            if attempt==7:raise
            time.sleep(3)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Download pinned public model and manual audio assets; resume partial files.')
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4,choices=range(1,9))
    args=parser.parse_args()
    ROOT=args.output.resolve();ROOT.mkdir(parents=True,exist_ok=True)
    items=json.loads(args.manifest.read_text())
    for item in items:
        destination=(ROOT/item['destination']).resolve()
        if ROOT not in destination.parents:raise ValueError('Unsafe destination')
        if not item.get('sha256'):raise ValueError('Expected SHA256 required for every file')
    items.sort(key=lambda x: (0 if 'manual_annotation' in x['destination'] else 1))
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(fetch,items))
    (ROOT/'downloads_complete.json').write_text(json.dumps({'files':len(items),'completed':True,'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest()}))
