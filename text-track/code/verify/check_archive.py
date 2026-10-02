import hashlib
import os
import tarfile

p = r"D:\文本附件资料.tar"
sz = os.path.getsize(p)
print("exists:", os.path.exists(p))
print("size  : {:,} bytes = {:.2f} GiB".format(sz, sz / 2**30))

t = tarfile.open(p, "r:")
for m in t.getmembers():
    print("  member: {}  size={:,}  bytes".format(m.name, m.size))
t.close()

t = tarfile.open(p, "r:")
ti = [m for m in t.getmembers() if m.name.endswith(".tar")][0]
f = t.extractfile(ti)
h = hashlib.sha256()
left = ti.size
while left > 0:
    b = f.read(min(8 << 20, left))
    if not b:
        break
    h.update(b)
    left -= len(b)
got = h.hexdigest()
exp = "a163a653d2aa05b7bd51ec6016fbc5aed51379f7802fbfbb9a39c231dc691f68"
print("inner image sha256:", got)
print("RESULT:", "MATCH" if got == exp else "MISMATCH")
