"""Rebuild licensed font subsets from local full-font sources for draft.html."""
from pathlib import Path
import argparse
import hashlib
import json
import re

from fontTools import subset
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    args = parser.parse_args()
    draft_path = ROOT / '.build/draft.html'
    draft = draft_path.read_text(encoding='utf-8')
    chars = {ord(c) for c in draft if ord(c) >= 32} | set(range(32, 127))
    audience = draft[draft.index('<div id="deck">'):draft.index('<div id="nav">')]
    notes = re.search(r'const SPEAKER_NOTES = ([\s\S]*?);\s*window.__SPEAKER_NOTES__', draft).group(1)
    needed = {ord(c) for c in re.sub(r'<[^>]*>', '', audience) + notes if not c.isspace()}
    old = json.loads((ROOT / 'assets/font-coverage.json').read_text(encoding='utf-8'))
    records, coverage, supported = [], [], set()
    for family, source, output in [
        ('Noto Sans SC', 'NotoSansSC-variable-full.ttf', 'NotoSansSC-subset.woff2'),
        ('Inter', 'Inter-variable.ttf', 'Inter-subset.woff2'),
    ]:
        font = TTFont(args.source_dir / source)
        previous = next((set(f['unicode_codepoints']) for f in old['fonts'] if f['filename'] == output), set())
        available = (chars | previous) & set(font.getBestCmap())
        weight = next(a for a in font['fvar'].axes if a.axisTag == 'wght')
        font = instantiateVariableFont(font, {'wght': (200, max(200, weight.defaultValue), 900)}, inplace=True)
        options = subset.Options()
        options.flavor = 'woff2'
        options.name_IDs = ['*']
        options.name_legacy = True
        options.name_languages = ['*']
        cutter = subset.Subsetter(options=options)
        cutter.populate(unicodes=available)
        cutter.subset(font)
        font.flavor = 'woff2'
        target = ROOT / 'assets' / output
        font.save(target)
        actual = TTFont(target)
        current = set(actual.getBestCmap())
        supported |= current
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        coverage.append(dict(filename=output, sha256=digest, unicode_codepoints=sorted(current)))
        records.append(dict(family=family, source=source, output=output, bytes=target.stat().st_size,
                            sha256=digest, unicode_count=len(current),
                            axes=[dict(tag=a.axisTag, min=a.minValue, default=a.defaultValue, max=a.maxValue)
                                  for a in actual['fvar'].axes]))
    missing = sorted(needed - supported)
    report = dict(draft_sha256=hashlib.sha256(draft_path.read_bytes()).hexdigest(),
                  coverage_success=not missing, required_characters=len(needed),
                  missing=[dict(codepoint=hex(c), char=chr(c)) for c in missing], fonts=records)
    for name, data in [('font-coverage.json', dict(fonts=coverage)), ('font-subset-report.json', report)]:
        (ROOT / 'assets' / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8', newline='\r\n')
    print(json.dumps(dict(coverage_success=not missing, missing=report['missing'],
                         fonts=[dict(output=r['output'], bytes=r['bytes']) for r in records]), ensure_ascii=False))
    assert not missing, 'Required characters missing from font subsets'


if __name__ == '__main__':
    main()
