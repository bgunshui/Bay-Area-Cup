"""Read-only, standard-library validation for the tracked defense materials."""
from pathlib import Path
from html.parser import HTMLParser
import hashlib
import json
import re
import sys
from update_checksums import ROOT,material_files

def read(name):return (ROOT/name).read_text(encoding='utf-8')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class Deck(HTMLParser):
    def __init__(self):super().__init__();self.slides=[];self.active=False;self.text=[]
    def handle_starttag(self,tag,attrs):
        values=dict(attrs)
        if tag=='section' and 'slide' in values.get('class','').split():
            self.slides.append(values);self.active=True
    def handle_endtag(self,tag):
        if tag=='section':self.active=False
    def handle_data(self,data):
        if self.active:self.text.append(data)

def main():
    html=read('index.html')
    deck=Deck();deck.feed(html)
    assert len(deck.slides)==14,'Expected 14 audience slides'
    ids=[s['data-slide-id'] for s in deck.slides]
    assert len(set(ids))==14,'Slide IDs must be unique'
    match=re.search(r'const SPEAKER_NOTES = (\[[\s\S]*?\]);\s*window.__SPEAKER_NOTES__',html)
    assert match,'Embedded speaker notes missing'
    notes=json.loads(match.group(1))
    assert [n['id'] for n in notes]==ids,'Speaker notes and slides differ'
    seconds=[round(n['minutes']*60) for n in notes]
    assert sum(seconds)==800,'Expected 800 planned seconds'
    assert not any(n.get('autoAdvanceSeconds') for n in notes),'Automatic advance must remain off'
    layouts={s['data-layout'] for s in deck.slides}
    assert len(layouts)>=8 and layouts<={f'S{i:02d}' for i in range(1,23)}
    assert not re.search(r'<(?:script|link|img)[^>]+(?:src|href)=[\"\']https?://',html,re.I),'External runtime asset found'
    speech=read('逐字讲稿.md')
    assert [int(s) for s in re.findall(r'计划：(\d+)秒',speech)]==seconds
    assert re.findall(r'^## \d{2}｜([^\n]+)',speech,re.M)==[s['data-title'] for s in deck.slides]
    spoken=re.findall(r'^## \d{2}[^\n]*\n[\s\S]*?计划：\d+秒[^\n]*\n\n([\s\S]*?)(?=\n\n> 证据提示)',speech,re.M)
    chars=len(re.findall(r'[\u4e00-\u9fffA-Za-z0-9]',''.join(spoken)))
    assert len(spoken)==14 and 2600<=chars<=3000
    _qids=[int(n) for n in re.findall(r'^### Q(\d+)(?:\.|｜)',read('评委问答.md'),re.M)]
    assert _qids==list(range(1,len(_qids)+1))
    audience=''.join(deck.text)
    assert not any(w in audience for w in ['复核','交付','仓库成绩记录','交接确认','证据等级','验收'])

    coverage=json.loads(read('assets/font-coverage.json'))
    supported=set()
    for font in coverage['fonts']:
        assert digest(ROOT/'assets'/font['filename'])==font['sha256'],'Font coverage data is stale'
        supported.update(font['unicode_codepoints'])
    required={ord(c) for c in audience+json.dumps(notes,ensure_ascii=False) if not c.isspace()}
    missing=required-supported
    assert not missing,f'Expand the embedded font subsets for {sorted(missing)}'
    for stem in ['图像候选AUC','文本候选AUC','软标签函数','教师融合权重','图像分量相关性','尺寸先验OOF']:
        assert all((ROOT/'图表'/(stem+suffix)).is_file() for suffix in ['.png','.svg'])

    entries={}
    for line in read('SHA256SUMS.txt').splitlines():
        value,name=line.split('  ',1)
        assert name not in entries,'Duplicate checksum entry'
        entries[name]=value
    files={p.relative_to(ROOT).as_posix():p for p in material_files()}
    assert set(files)==set(entries),'Update SHA256SUMS.txt after adding or removing files'
    assert all(digest(path)==entries[name] for name,path in files.items()),'A material differs from SHA256SUMS.txt'
    assert not list(ROOT.rglob('*.zip')),'Archives do not belong in defense/'

    browser=json.loads(read('验收记录.json'))
    matches=browser.get('inputSha256')==digest(ROOT/'index.html')
    if matches:
        assert browser['passed'] and all(c['ok'] for c in browser['checks'])
        assert not browser['errors'] and not browser['network']
    print(json.dumps(dict(slides=14,seconds=800,layouts=len(layouts),speech_characters=chars,questions=len(re.findall(r'^### Q(?:\d+)(?:\.|｜)',read('评委问答.md'),re.M)),
        checksummed_files=len(files),offline_assets=True,current_html_matches_recorded_browser_test=matches),ensure_ascii=False))
    if not matches:
        print('NOTICE: HTML changed; the historical browser test is not validation of this revision. Recheck layout and interactions before presenting.')

if __name__=='__main__':
    try:main()
    except (AssertionError,ValueError,KeyError) as error:
        print(f'ERROR: {error}',file=sys.stderr)
        sys.exit(1)
