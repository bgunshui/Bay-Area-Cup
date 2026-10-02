import os, sys, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

resolve = sys.argv[1]
out = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv)>3 else 16
r = requests.get(resolve, allow_redirects=False, timeout=60)
r.raise_for_status()
url = r.headers['location']
r = requests.get(url, headers={'Range':'bytes=0-0'}, stream=True, timeout=60)
r.raise_for_status()
total = int(r.headers['content-range'].split('/')[-1])
r.close()
part_dir = out + '.parts'
os.makedirs(part_dir, exist_ok=True)
chunk = (total+n-1)//n
def one(i):
    start=i*chunk; end=min(total-1,(i+1)*chunk-1)
    p=os.path.join(part_dir,f'{i:03d}.part')
    if os.path.exists(p) and os.path.getsize(p)==end-start+1: return i, os.path.getsize(p), 'cached'
    for attempt in range(8):
        try:
            rr=requests.get(url,headers={'Range':f'bytes={start}-{end}'},stream=True,timeout=(30,300))
            if rr.status_code!=206: raise RuntimeError(f'status {rr.status_code}')
            got=0
            with open(p+'.tmp','wb') as f:
                for b in rr.iter_content(8*1024*1024):
                    if b: f.write(b); got+=len(b)
            rr.close()
            if got != end-start+1: raise RuntimeError(f'got {got}, expected {end-start+1}')
            os.replace(p+'.tmp',p); return i,got,'ok'
        except Exception as e:
            try: os.remove(p+'.tmp')
            except OSError: pass
            if attempt==7: raise
            time.sleep(2*(attempt+1))
with ThreadPoolExecutor(max_workers=min(n,8)) as ex:
    fs=[ex.submit(one,i) for i in range(n)]
    for k,f in enumerate(as_completed(fs),1): print(k,'/',n,f.result(),flush=True)
tmp=out+'.tmp'
with open(tmp,'wb') as w:
    for i in range(n):
        with open(os.path.join(part_dir,f'{i:03d}.part'),'rb') as f:
            while True:
                b=f.read(16*1024*1024)
                if not b: break
                w.write(b)
if os.path.getsize(tmp)!=total: raise RuntimeError('final size mismatch')
os.replace(tmp,out)
import shutil; shutil.rmtree(part_dir,ignore_errors=True)
print('done',out,total,flush=True)
