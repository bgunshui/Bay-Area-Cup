"""Rebuild the judge-facing academic deck from the locked Swiss runtime."""
from pathlib import Path
import hashlib
import base64
import json
import re
import sys

OUT = Path(__file__).resolve().parents[1]
PROJECT = OUT.parent
BUILD = OUT / '.build'
SKILL = PROJECT / '.agents/skills/guizang-ppt-skill'
VENDOR = OUT / 'assets'
sys.path.insert(0, str(OUT/'src'))
import core
import image
import text
TIMES = [10,40,45,70,80,60,80,70,60,60,60,75,60,30]
NOTES = json.loads((OUT / 'src/page-plan.json').read_text(encoding='utf-8'))['notes']
PAGES = [f'<section class="slide light" data-slide-id="{n["id"]}"></section>' for n in NOTES]
PAGES, NOTES = core.apply(PAGES, NOTES)
PAGES, NOTES = image.apply(PAGES, NOTES)
PAGES, NOTES = text.apply(PAGES, NOTES)
assert len(PAGES) == len(NOTES) == len(TIMES) == 14

for i, (page, note) in enumerate(zip(PAGES, NOTES)):
    assert f'data-slide-id="{note["id"]}"' in page
    note['minutes'] = TIMES[i]/60
    note.pop('autoAdvanceSeconds', None)
    # Page counts remain in presenter controls; audience pages show content only.
    page = re.sub(r'<div class="r-page-index">[\s\S]*?</div>', '', page)
    PAGES[i] = page

template = (SKILL/'assets/template-swiss.html').read_text(encoding='utf-8')
template = re.sub(r'<title>.*?</title>', '<title>AI生成内容鉴别 · 南枝大学参赛队</title>', template, count=1)
template = re.sub(r'<link[^>]*(?:fonts\.googleapis|fonts\.gstatic)[^>]*>\s*', '', template)
start = template.index('<div id="deck">') + len('<div id="deck">')
end = template.index('<div id="nav">', start)
template = template[:start]+'\n\n'+'\n\n'.join(PAGES)+'\n\n</div>\n\n'+template[end:]
template = re.sub(r'const SPEAKER_NOTES = \[[\s\S]*?\];\s*window\.__SPEAKER_NOTES__ = SPEAKER_NOTES;',
    'const SPEAKER_NOTES = '+json.dumps(NOTES, ensure_ascii=False, indent=2)+';\nwindow.__SPEAKER_NOTES__ = SPEAKER_NOTES;',template,count=1)
template = template.replace('<script src="https://unpkg.com/lucide@latest/dist/umd/lucide.min.js"></script>','<!-- OFFLINE_LUCIDE -->')
template, replacements = re.subn(r'let motion;\s*try \{[\s\S]*?\n\}\s*\nif\(motion\)\{', 'let motion = window.Motion;\nif(motion){', template, count=1)
assert replacements == 1
template = template.replace('<!-- Motion One 动效引擎 (与原模板一致) -->','<!-- OFFLINE_MOTION -->\n<!-- Motion One: locally bundled 11.11.17 -->')
template = template.replace('frame.style.width=`${width}px`;frame.style.height=`${height}px`;',
    'frame.style.width="1920px";frame.style.height="1080px";frame.style.transform=`translate(-50%,-50%) scale(${width/1920})`;')

UNIFY = r'''
/* Conference-talk typography; the user requested a full academic redesign. */
.slide{font-family:Inter,"Noto Sans SC",sans-serif}
.r-title{font-size:clamp(40px,2.916667vw,56px);font-weight:600;line-height:1.15;padding-bottom:0;border-bottom:0}
.r-body{margin-top:4.1vh}
.slide:not([data-slide-id="cover"]):not([data-slide-id="closing"])::after{content:"";position:absolute;left:4.5vw;right:4.5vw;top:15vh;border-top:1px solid #ccd9e1;pointer-events:none}
.r-slide sub,.r-slide sup{font-size:22px;line-height:0}
.im-page .im-math,.r-eq i{font-family:Inter,"Noto Sans SC",sans-serif}
.im-page .im-frame{gap:max(36px,5vh)}
.im-page .im-title{line-height:1.15;padding-bottom:0;border-bottom:0}
.im-page .im-train-grid{grid-template-rows:5.3vh 1.32fr 1fr 1fr 1fr}
.im-page .im-th-count{text-align:center}
.im-page .im-slot{gap:max(14px,1.2vh)}
.im-page .im-ref-panel{gap:max(14px,1vh)}
.im-page .im-corr-panel{gap:max(14px,1.4vh)}
.r-deploy-condition{min-height:3.1em}
.ppt-frame{position:absolute;left:50%;top:50%;max-width:none;max-height:none;transform-origin:center center}
'''
template = template.replace('</style>', core.CSS+'\n'+image.CSS+'\n'+text.CSS+'\n'+UNIFY+'\n</style>',1)
template = template.replace('</head>', '<!-- OFFLINE_FONTS -->\n</head>',1)
BUILD.mkdir(exist_ok=True)
(BUILD/'draft.html').write_text(template,encoding='utf-8',newline='\r\n')
(BUILD/'page-plan.json').write_text(json.dumps(dict(seconds=TIMES,notes=NOTES,
    skill_commit='c91369c449d34755d320a8b81d0734000d99d1ab',
    evidence_commit='c11fa520f2f73d705e83cb3f40a6a214f8cc7f31',revision='2026-10-09'),ensure_ascii=False,indent=2),encoding='utf-8')

def finalize():
    doc=template
    for key, name in [('OFFLINE_LUCIDE','lucide-0.468.0.min.js'),('OFFLINE_MOTION','motion-11.11.17.js')]:
        data=(VENDOR/name).read_text(encoding='utf-8').replace('</script','<\\/script')
        doc=doc.replace('<!-- '+key+' -->','<script>\n'+data+'\n</script>')
    fontcss=[]
    for family, filename in [('Noto Sans SC','NotoSansSC-subset.woff2'),('Inter','Inter-subset.woff2')]:
        encoded=base64.b64encode((VENDOR/filename).read_bytes()).decode('ascii')
        fontcss.append('@font-face{font-family:"'+family+'";font-style:normal;font-weight:200 900;font-display:block;src:url(data:font/woff2;base64,'+encoded+') format("woff2")}')
    doc=doc.replace('<!-- OFFLINE_FONTS -->','<style>\n'+'\n'.join(fontcss)+'\n</style>')
    assert 'OFFLINE_' not in doc and '[必填]' not in doc and '[选填]' not in doc
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)="https?://',doc)
    assert 'await import(' not in doc
    (OUT/'index.html').write_text(doc,encoding='utf-8',newline='\r\n')
    print(json.dumps(dict(slides=len(PAGES),seconds=sum(TIMES),layouts=len(set(re.findall('data-layout="([^"]+)"',''.join(PAGES)))),
        html_bytes=len(doc.encode()),sha256=hashlib.sha256((OUT/'index.html').read_bytes()).hexdigest()),ensure_ascii=False))

if '--final' in sys.argv:
    finalize()
else:
    print('Draft and page plan updated; finalize after embedding the current font subset.')
