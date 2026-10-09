"""Refresh the unpacked materials' SHA256 manifest; never create an archive."""
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.build','__pycache__','node_modules'}

def material_files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file()
        and not (EXCLUDED & set(p.relative_to(ROOT).parts))
        and p.name!='SHA256SUMS.txt' and p.suffix.lower() not in {'.zip','.pyc'})

def main():
    files=material_files()
    lines=[hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix() for p in files]
    (ROOT/'SHA256SUMS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print(f'Updated {len(files)} checksums; no archive created.')

if __name__=='__main__':main()
