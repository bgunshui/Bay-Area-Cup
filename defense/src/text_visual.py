"""Native computation diagrams for the three text-track slides.

Only ``apply`` return values are changed.  Geometry is SVG; every visible
label is HTML so the deck's Chinese fonts and presenter scaling are retained.
The P7 main path is the v2 pseudo-label path, not the later teacher pool.
"""
import html
import math
import re


CSS = r"""
.tv-page{padding:0!important;background:#f8f9fb!important;color:#11283d!important}
.tv-page *{box-sizing:border-box}
.tv-page .tv-paper{width:100vw;height:100vh;padding:7vh 4.5vw 9.6vh;display:flex;flex-direction:column;overflow:hidden;background:#f8f9fb}
.tv-page .tv-title{font-family:Inter,"Noto Sans SC",sans-serif!important;font-size:clamp(40px,2.916667vw,56px)!important;font-weight:600!important;letter-spacing:-.02em;line-height:1.15!important;margin:0!important;color:#11283d}
.tv-page .tv-content{display:grid;grid-template-columns:1.65fr 1fr;gap:3vw;flex:1;min-height:0;margin-top:4.4vh}
.tv-page .tv-panel{display:flex;flex-direction:column;min-width:0;min-height:0}
.tv-page h3{font-size:clamp(24px,1.666667vw,32px)!important;font-weight:600!important;line-height:1.2!important;letter-spacing:0!important;margin:0 0 2.1vh!important;color:#174d6b}
.tv-page .tv-stage{position:relative;flex:1;min-height:0}
.tv-page .tv-lines{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.tv-page .tv-lines path,.tv-page .tv-lines line{vector-effect:non-scaling-stroke}
.tv-page .tv-box{position:absolute;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;gap:1vh;padding:1.2vh .6vw;border:1px solid #becdd7;background:#fff;min-width:0;line-height:1.3}
.tv-page .tv-box strong{font-size:clamp(22px,1.5625vw,30px);font-weight:600;line-height:1.22}
.tv-page .tv-label{font-size:clamp(22px,1.354167vw,26px);font-weight:500;line-height:1.28}
.tv-page .tv-small{font-size:clamp(22px,1.25vw,24px);font-weight:500;line-height:1.3;color:#4c6579}
.tv-page .tv-math{font-family:Inter,"Noto Sans SC",sans-serif;font-size:clamp(22px,1.458333vw,28px);font-weight:500;line-height:1.3;white-space:nowrap;letter-spacing:-.025em}
.tv-page sub{font-size:inherit!important;position:relative;bottom:-.2em;vertical-align:baseline;line-height:inherit!important}
.tv-page .tv-num{font-variant-numeric:tabular-nums}
.tv-page .tv-note{font-size:clamp(22px,1.354167vw,26px);font-weight:500;line-height:1.35;color:#4c6579;margin:2vh 0 0!important;padding-top:1.5vh;border-top:1px solid #c9d4df}
.tv-page .tv-accent{background:#e7eff4;border-color:#91adbe;color:#174d6b}
.tv-page .tv-output{background:#174d6b;border-color:#174d6b;color:#fff}
.tv-page .tv-output .tv-label,.tv-page .tv-output .tv-small{color:#fff}
.tv-page .tv-floating{position:absolute;font-size:clamp(22px,1.354167vw,26px);font-weight:500;line-height:1.25;background:#f8f9fb;color:#4c6579;padding:.1vh .3vw;white-space:nowrap;text-align:center}
.tv-page .tv-model{font-size:clamp(34px,2.604167vw,50px);font-weight:600;line-height:1.05;color:#174d6b}
.tv-page .tv-train .tv-box{width:28%;height:25%}
.tv-page .tv-train .tv-teacher{left:0;top:0}
.tv-page .tv-train .tv-rank{left:36%;top:0}
.tv-page .tv-train .tv-pseudo{left:72%;top:0}
.tv-page .tv-train .tv-hard{left:0;top:43%}
.tv-page .tv-train .tv-loss{left:36%;top:43%}
.tv-page .tv-train .tv-valid{left:72%;top:43%}
.tv-page .tv-train .tv-student{left:36%;top:81%;height:19%}
.tv-page .tv-train .tv-oof{left:72%;top:81%;height:19%}
.tv-page .tv-formula{display:flex;align-items:center;gap:.75vw;font-size:clamp(22px,1.354167vw,26px);font-weight:500;line-height:1.3;white-space:nowrap;margin-bottom:1.2vh;letter-spacing:-.04em}
.tv-page .tv-key{width:1.7vw;min-width:22px;border-top:4px solid #174d6b}
.tv-page .tv-key.weight{border-top:3px dashed #8199ac}
.tv-page .tv-plot{position:relative;flex:1;min-height:0;margin-top:1vh}
.tv-page .tv-plot svg{position:absolute;inset:0;width:100%;height:100%}
.tv-page .tv-tick{position:absolute;font-size:clamp(22px,1.25vw,24px);font-weight:500;line-height:1.1;color:#4c6579;white-space:nowrap}
.tv-page .tv-xtick{transform:translateX(-50%)}
.tv-page .tv-ytick{transform:translate(-100%,-50%)}
.tv-page .tv-point{position:absolute;font-size:clamp(22px,1.25vw,24px);font-weight:600;line-height:1.1;color:#174d6b;background:#f8f9fb;padding:2px 4px;white-space:nowrap}
.tv-page .tv-ensemble-grid{grid-template-columns:1.52fr 1fr}
.tv-page .tv-tree{min-height:0}
.tv-page .tv-base14{left:2%;top:0;width:29.3%;height:21%}
.tv-page .tv-base7{left:52%;top:0;width:29.3%;height:21%}
.tv-page .tv-adapters{position:absolute;top:34%;width:31.3%;display:flex;flex-direction:column;align-items:center;gap:1.1vh}
.tv-page .tv-adapters strong{font-size:clamp(22px,1.5625vw,30px);font-weight:600;line-height:1.2}
.tv-page .tv-adapter-row{display:grid;grid-template-columns:repeat(5,1fr);gap:.35vw;width:100%}
.tv-page .tv-adapter{padding:.85vh 0;background:#dce7ef;border:1px solid #b2c5d3;text-align:center;color:#174d6b;font-size:clamp(22px,1.25vw,24px);font-weight:600;line-height:1}
.tv-page .tv-tree .tv-aggregate{top:66%;height:15%;width:29.3%;padding:.8vh .3vw}
.tv-page .tv-tree .tv-aggregate14{left:2%}
.tv-page .tv-tree .tv-aggregate7{left:52%}
.tv-page .tv-tree .tv-final{left:22%;top:91%;width:56%;height:9%;padding:.2vh}
.tv-page .tv-tree .tv-floating{top:83%;color:#174d6b;font-weight:600}
.tv-page .tv-results{display:flex;flex-direction:column;gap:2.2vh;flex:1;min-height:0}
.tv-page .tv-result{position:relative;padding:1.1vh 0 1.5vh;border-bottom:1px solid #d1dde5;display:flex;align-items:center;justify-content:space-between;gap:1vw}
.tv-page .tv-result-name{font-size:clamp(22px,1.458333vw,28px);font-weight:500;line-height:1.3}
.tv-page .tv-result-value{font-size:clamp(26px,1.979167vw,38px);font-weight:600;line-height:1.2;color:#174d6b;font-variant-numeric:tabular-nums}
.tv-page .tv-result:last-child{border-bottom:3px solid #174d6b}
.tv-page .tv-ensemble-grid .tv-results{gap:1.5vh}
.tv-page .tv-auc-row{display:flex;flex-direction:column;gap:.6vh;padding:.4vh 0 .6vh}
.tv-page .tv-auc-heading{display:flex;justify-content:space-between;align-items:center;gap:1vw}
.tv-page .tv-auc-track{height:2vh;position:relative;border-left:1px solid #bbcbd7;border-right:1px solid #bbcbd7;background:linear-gradient(90deg,transparent calc(50% - .5px),#d4dfe7 calc(50% - .5px),#d4dfe7 calc(50% + .5px),transparent calc(50% + .5px))}
.tv-page .tv-auc-track::before{content:"";position:absolute;left:0;right:0;top:50%;border-top:1px solid #d4dfe7}
.tv-page .tv-auc-line{position:absolute;left:0;top:50%;height:3px;transform:translateY(-50%);background:#8da7b9}
.tv-page .tv-auc-dot{position:absolute;top:50%;width:13px;height:13px;border:2px solid #174d6b;border-radius:50%;background:#f8f9fb;transform:translate(-50%,-50%)}
.tv-page .tv-auc-row:last-of-type .tv-auc-line{background:#174d6b;height:4px}.tv-page .tv-auc-row:last-of-type .tv-auc-dot{background:#174d6b}
.tv-page .tv-auc-axis{position:relative;display:flex;justify-content:space-between;font-size:clamp(22px,1.25vw,24px);color:#567387;margin-top:-1vh}
.tv-page .tv-gain{margin-top:auto;font-size:clamp(22px,1.354167vw,26px);font-weight:500;line-height:1.4;color:#4c6579;padding-top:1vh}
.tv-page .tv-gain strong{color:#174d6b;font-weight:600}
.tv-page .tv-inference-grid{grid-template-columns:1fr 1fr;gap:5vw}
.tv-page .tv-inference-grid .tv-panel+ .tv-panel{border-left:1px solid #c9d4df;padding-left:3vw}
.tv-page .tv-calc .tv-input{top:0;height:17%;padding:.8vh .3vw;gap:.6vh}
.tv-page .tv-calc .tv-op{position:absolute;display:flex;justify-content:center;align-items:center;text-align:center;border:1px solid #90acbd;background:#e7eff4;color:#174d6b;font-size:clamp(26px,1.666667vw,32px);font-weight:550;line-height:1.1}
.tv-page .tv-offline .tv-input{width:28%}
.tv-page .tv-offline .tv-rank-op{top:30%;height:14%;width:20%;border-radius:50%}
.tv-page .tv-offline .tv-rank-out{top:46%;width:20%;padding:0;background:#f8f9fb;color:#174d6b}
.tv-page .tv-calc .tv-sum{width:12%;height:14%;border-radius:50%;left:44%;font-size:clamp(28px,2.1875vw,42px);background:#fff}
.tv-page .tv-offline .tv-sum{top:61%}
.tv-page .tv-offline .tv-final-rank{left:31%;top:83%;width:38%;height:17%;padding:.8vh .3vw;gap:.6vh}
.tv-page .tv-online .tv-input{width:45%}
.tv-page .tv-logit-marks{display:flex;justify-content:center;gap:3px;width:90%;height:1.4vh}
.tv-page .tv-logit-marks i{display:block;flex:1;background:#7798af}
.tv-page .tv-online .tv-mean{top:30%;height:14%;width:45%;padding:.5vh .3vw}
.tv-page .tv-online .tv-sigma{top:54%;width:14%;height:15%;border-radius:50%}
.tv-page .tv-online .tv-prob{top:70%;width:14%;padding:0;background:#f8f9fb;color:#174d6b}
.tv-page .tv-online .tv-sum{top:78%;height:11%}
.tv-page .tv-online .tv-final-score{left:22%;top:92%;width:56%;height:8%;padding:.1vh}
.tv-page .tv-offline .tv-coefficient{top:55%;color:#174d6b;font-weight:600}
.tv-page .tv-online .tv-coefficient{top:76%;color:#174d6b;font-weight:600}
@media(max-width:1450px){
.tv-page .tv-content{gap:2.3vw}
.tv-page .tv-box{padding:1.1vh .35vw;gap:.85vh}
.tv-page .tv-results{gap:1.7vh}
.tv-page .tv-result{padding:.6vh 0 1.4vh}
.tv-page .tv-inference-grid{gap:3.6vw}
.tv-page .tv-inference-grid .tv-panel+ .tv-panel{padding-left:2.4vw}
.tv-page .tv-formula{gap:.5vw;letter-spacing:-.045em}
.tv-page .tv-train .tv-box strong{font-size:22px}
}
"""


def _svg(content, prefix, label):
    return f'''<svg class="tv-lines" viewBox="0 0 1000 1000" preserveAspectRatio="none" role="img" aria-label="{html.escape(label, quote=True)}">
      <defs><marker id="{prefix}-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#8199ac"/></marker></defs>
      <g fill="none" stroke="#8199ac" stroke-width="2.4" marker-end="url(#{prefix}-arrow)">{content}</g>
    </svg>'''


def _curves():
    soft = lambda rank: 1 / (1 + math.exp(-4 * (rank - .5)))
    weight = lambda rank: min(1, 4 * abs(soft(rank) - .5))
    a = .5 - math.log(3) / 4
    b = 1 - a
    ranks = sorted({j / 200 for j in range(201)} | {a, .5, b})
    points = lambda f: " ".join(f"{70+540*r:.4f},{340-310*f(r):.4f}" for r in ranks)
    return f'''<div class="tv-plot">
      <svg viewBox="0 0 650 390" preserveAspectRatio="none" role="img" aria-label="确定性映射函数示意，横轴为目标集秩分位值r，实线为sigmoid软标签，虚线为置信权重">
        <path d="M70 30V340H610" stroke="#8199ac" stroke-width="2" fill="none"/>
        <path d="M70 30H610M70 185H610M340 30V340" stroke="#d4dfe7" stroke-width="1.5" fill="none"/>
        <path d="M{70+540*a:.4f} 30V340M{70+540*b:.4f} 30V340" stroke="#b8c7d5" stroke-dasharray="5 7" fill="none"/>
        <polyline points="{points(weight)}" stroke="#8199ac" stroke-width="3.5" stroke-dasharray="8 6" fill="none" stroke-linejoin="round"/>
        <polyline points="{points(soft)}" stroke="#174d6b" stroke-width="4" fill="none" stroke-linejoin="round"/>
      </svg>
      <span class="tv-tick tv-ytick" style="left:8.9%;top:7.692%">1</span>
      <span class="tv-tick tv-ytick" style="left:8.9%;top:47.436%">.5</span>
      <span class="tv-tick tv-ytick" style="left:8.9%;top:87.179%">0</span>
      <span class="tv-tick" style="left:2%;top:0">y, w</span>
      <span class="tv-tick tv-xtick" style="left:10.769%;top:90%">0</span>
      <span class="tv-tick tv-xtick" style="left:52.308%;top:90%">.5</span>
      <span class="tv-tick tv-xtick" style="left:93.846%;top:90%">1</span>
      <span class="tv-tick" style="right:1%;top:97%">r</span>
      <span class="tv-point" style="left:13%;top:76%">.1192</span>
      <span class="tv-point" style="right:6%;top:16%">.8808</span>
    </div>'''


def _training():
    paths = '''<path d="M280 125H355"/><path d="M640 125H715"/>
      <path d="M860 250V335H500V425"/><path d="M280 555H355"/>
      <path d="M500 680V805"/><path d="M640 905H683V555H715"/>
      <path d="M360 905H330V555H355" stroke-dasharray="6 5"/>
      <path d="M860 680V805"/>'''
    return f'''<div class="tv-paper">
      <h2 class="tv-title" data-anim="head">文本：目标域自训练</h2>
      <div class="tv-content" data-anim="up">
        <div class="tv-panel">
          <h3>软监督 → 14B学生</h3>
          <div class="tv-stage tv-train loop-diagram">
            {_svg(paths, 'tv7', '文本v2标签数据流：上一代检测分数转为目标集秩和软标签；800条硬标签侧路汇入加权BCE；LoRA学生经200条验证选择checkpoint，OOF用于融合选择')}
            <div class="tv-box tv-teacher"><strong>上一代检测分数</strong><span class="tv-small">20,000目标文本</span></div>
            <div class="tv-box tv-rank"><strong>目标集排序</strong><span class="tv-math">r = rank / (N+1)</span><span class="tv-small">并列取平均秩</span></div>
            <div class="tv-box tv-pseudo tv-accent"><strong>软标签 y</strong><span class="tv-label">置信权重 w</span></div>
            <div class="tv-box tv-hard"><strong>官方硬标签</strong><span class="tv-label">每折800条</span><span class="tv-small">权重 = 2</span></div>
            <div class="tv-box tv-loss tv-accent"><strong>加权BCE</strong><span class="tv-math">mean(w · BCE)</span></div>
            <div class="tv-box tv-valid"><strong>每折200条验证</strong><span class="tv-small">AUC选checkpoint</span></div>
            <div class="tv-box tv-student tv-output"><strong>LoRA学生</strong><span class="tv-small">适配器＋分类头</span></div>
            <span class="tv-floating" style="left:53%;top:71.5%">梯度更新</span><span class="tv-floating" style="left:22%;top:73%">预测</span>
            <div class="tv-box tv-oof"><strong>五折OOF</strong><span class="tv-small">选择融合方案</span></div>
          </div>
          <p class="tv-note">冻结语言基座；以目标域软监督适配检测任务。</p>
        </div>
        <div class="tv-panel">
          <h3>排序 → 标签与权重</h3>
          <div class="tv-formula"><span class="tv-key"></span><span>y = σ[4(r − 0.5)]</span></div>
          <div class="tv-formula"><span class="tv-key weight"></span><span>w = min(1, 4|y − 0.5|)</span></div>
          {_curves()}
          <p class="tv-note">中点弱监督，排序两端权重饱和；<br>曲线表示公式映射。</p>
        </div>
      </div>
    </div>'''


def _adapter_group(left, name):
    blocks = ''.join(f'<span class="tv-adapter">{i}</span>' for i in range(5))
    return f'''<div class="tv-adapters" style="left:{left}%"><strong>{name}</strong>
      <div class="tv-adapter-row" aria-label="{name}五折适配器0至4">{blocks}</div>
      <span class="tv-small">5个adapter</span></div>'''


def _ensemble():
    paths = '''<path d="M166.5 210V327"/><path d="M666.5 210V266H500V327"/><path d="M666.5 266H833.5V327"/>
      <path d="M166.5 570V650"/><path d="M500 570V610H666.5V650"/><path d="M833.5 570V610H666.5"/>
      <path d="M166.5 810V870H500V904"/><path d="M666.5 810V870H500"/>'''
    candidates = [('早期融合候选', '.978284'), ('14B自训练候选', '.979964'), ('14B双种子', '.980896'), ('14B五折', '.981069'), ('14B＋7B', '.981301')]
    rows = ''
    for name, auc in candidates:
        position = (float(auc) - .978) / .004 * 100
        rows += f'<div class="tv-auc-row"><div class="tv-auc-heading"><span class="tv-result-name">{name}</span><span class="tv-result-value">0{auc}</span></div><div class="tv-auc-track"><i class="tv-auc-line" style="width:{position:.4f}%"></i><i class="tv-auc-dot" style="left:{position:.4f}%"></i></div></div>'
    return f'''<div class="tv-paper">
      <h2 class="tv-title" data-anim="head">文本：两基座，15个适配器</h2>
      <div class="tv-content tv-ensemble-grid" data-anim="up">
        <div class="tv-panel">
          <h3>共享基座 · 适配器轮换</h3>
          <div class="tv-stage tv-tree">
            {_svg(paths, 'tv8', '两个共享Qwen2.5基座；14B p4五个adapter，7B r3a和r3b各五个adapter；14B五个logit平均后sigmoid，7B十个logit平均后sigmoid，最后九比一融合')}
            <div class="tv-box tv-base14 tv-accent"><span class="tv-small">Qwen2.5</span><span class="tv-model">14B</span></div>
            <div class="tv-box tv-base7 tv-accent"><span class="tv-small">Qwen2.5</span><span class="tv-model">7B</span></div>
            {_adapter_group(1, '主分支')}{_adapter_group(34.3, '辅助 A')}{_adapter_group(67.7, '辅助 B')}
            <div class="tv-box tv-aggregate tv-aggregate14"><span class="tv-math">p<sub>14</sub> = σ(mean 5)</span></div>
            <div class="tv-box tv-aggregate tv-aggregate7"><span class="tv-math">p<sub>7</sub> = σ(mean 10)</span></div>
            <span class="tv-floating" style="left:25%">0.9</span><span class="tv-floating" style="left:57%">0.1</span>
            <div class="tv-box tv-final tv-output"><span class="tv-math">0.9p<sub>14</sub> ＋ 0.1p<sub>7</sub></span></div>
          </div>
          <p class="tv-note">两个基座常驻；每条文本执行5＋5＋5次前向。</p>
        </div>
        <div class="tv-panel">
          <h3>候选方案 · 复赛AUC</h3>
          <div class="tv-results" role="img" aria-label="五个文本候选的复赛AUC，局部坐标0.978至0.982">{rows}<div class="tv-auc-axis"><span>0.978</span><span>0.980</span><span>0.982</span></div><div class="tv-gain">双种子未超过五折组合。<br>最终组合增量 <strong>＋0.000232</strong></div></div>
        </div>
      </div>
    </div>'''


def _offline():
    paths = '''<path d="M140 170V295"/><path d="M500 170V295"/><path d="M860 170V295"/>
      <path d="M140 440V535L443 667"/><path d="M500 490V605"/><path d="M860 440V535L557 667"/>
      <path d="M500 750V825"/>'''
    branches = [('0', '4', '14B主分支', 'r<sub>14</sub>'), ('36', '40', '7B辅助 A', 'r<sub>a</sub>'), ('72', '76', '7B辅助 B', 'r<sub>b</sub>')]
    nodes = ''.join(f'''<div class="tv-box tv-input" style="left:{x}%"><strong>{model}</strong><span class="tv-small">分支预测</span></div>
      <div class="tv-op tv-rank-op" style="left:{opx}%">rank</div><span class="tv-floating tv-rank-out tv-math" style="left:{opx}%">{r}</span>''' for x, opx, model, r in branches)
    return f'''<div class="tv-stage tv-calc tv-offline">
      {_svg(paths, 'tv9off', '离线路径：三个分支预测分别rank，按0.9、0.05、0.05加权，再对整批做最终rank')}
      {nodes}<div class="tv-op tv-sum">＋</div>
      <span class="tv-floating tv-coefficient" style="left:27%;top:59%">0.9</span><span class="tv-floating tv-coefficient" style="left:47%">0.05</span><span class="tv-floating tv-coefficient" style="left:67%;top:59%">0.05</span>
      <div class="tv-box tv-final-rank tv-output"><strong>最终rank</strong><span class="tv-small">离线检测分数</span></div>
    </div>'''


def _online():
    paths = '''<path d="M225 170V295"/><path d="M775 170V295"/>
      <path d="M225 440V535"/><path d="M775 440V535"/>
      <path d="M225 735V770L443 840"/><path d="M775 735V770L557 840"/>
      <path d="M500 890V915"/>'''
    leftmarks = '<i></i>' * 5
    rightmarks = '<i></i>' * 10
    return f'''<div class="tv-stage tv-calc tv-online">
      {_svg(paths, 'tv9on', '在线路径：14B五个logit先平均，7B十个logit先平均；各做一次sigmoid后以0.9和0.1相加')}
      <div class="tv-box tv-input" style="left:0"><strong>14B · 5个logit</strong><div class="tv-logit-marks" aria-hidden="true">{leftmarks}</div></div>
      <div class="tv-box tv-input" style="left:55%"><strong>7B · 10个logit</strong><div class="tv-logit-marks" aria-hidden="true">{rightmarks}</div></div>
      <div class="tv-box tv-mean" style="left:0"><span class="tv-math">μ<sub>14</sub> = mean(5)</span></div>
      <div class="tv-box tv-mean" style="left:55%"><span class="tv-math">μ<sub>7</sub> = mean(10)</span></div>
      <div class="tv-op tv-sigma" style="left:15.5%">σ</div><div class="tv-op tv-sigma" style="left:70.5%">σ</div>
      <span class="tv-floating tv-prob tv-math" style="left:15.5%">p<sub>14</sub></span><span class="tv-floating tv-prob tv-math" style="left:70.5%">p<sub>7</sub></span>
      <span class="tv-floating tv-coefficient" style="left:35%">0.9</span><span class="tv-floating tv-coefficient" style="left:58%">0.1</span>
      <div class="tv-op tv-sum">＋</div><div class="tv-box tv-final-score tv-output"><span class="tv-math">0.9p<sub>14</sub> ＋ 0.1p<sub>7</sub></span></div>
    </div>'''


def _inference():
    return f'''<div class="tv-paper">
      <h2 class="tv-title" data-anim="head">文本：批排名与单条推理</h2>
      <div class="tv-content tv-inference-grid duo-compare" data-anim="up">
        <div class="tv-panel duo-half"><h3>整批目标集合 · 秩空间</h3>{_offline()}</div>
        <div class="tv-panel duo-half"><h3>单条输入 · logit空间</h3>{_online()}</div>
      </div>
      <p class="tv-note">聚合空间不同，不保证同排序；分支预测已包含各自逐折聚合。在线上限512 tokens。</p>
    </div>'''


def _replace(previous, body):
    match = re.match(r"\s*(<section\b[^>]*>)", previous)
    if not match:
        raise ValueError("Text diagram requires an existing slide section")
    tag = re.sub(r'\bclass="[^"]*"', 'class="slide light tv-page"', match.group(1), count=1)
    return tag + body + '</section>'


def apply(pages, notes):
    """Keep stable IDs, titles, layout registrations, and planned durations."""
    updated, cuecards = list(pages), [dict(n) for n in notes]
    definitions = [('text-self-training', _training), ('text-ensemble', _ensemble), ('text-offline-service', _inference)]
    for sid, make in definitions:
        hits = [i for i, slide in enumerate(updated) if f'data-slide-id="{sid}"' in slide]
        if len(hits) != 1 or cuecards[hits[0]].get('id') != sid:
            raise ValueError('Unexpected text slide or note: ' + sid)
        updated[hits[0]] = _replace(updated[hits[0]], make())
    cuecards[6]['purpose'] = '沿v2标签生成、硬标签侧路、LoRA训练和验证解释目标域自训练'
    cuecards[6]['talk'] = [
        '从左上沿实线看已确认的v2标签链：mk_pseudo_v2读取上一代submit_v2_p2AD.csv，先排名，再按sigmoid生成软标签。p43默认标签与日志指向v2；后续43候选中六路非零教师属于不同代际，不能直接接到这条标签链。',
        '每折800条官方硬标签以权重2汇入加权BCE；20,000目标文本的软标签以置信权重w汇入。L=mean(w·BCE)，不是除以权重总和。冻结语言基座，更新LoRA和分类头。',
        '右侧实线y=σ[4(r−0.5)]，端点约0.1192与0.8808；虚线w=min(1,4|y−0.5|)，中点0，两端饱和到1。图是确定性映射，权重不是标签正确概率。',
        '学生对每折200条验证样本预测，由验证AUC选checkpoint；五折OOF用于融合选择。这不是新的独立测试集。',
        '主图表示已确认的14B学生v2链路，不表示所有最终adapter都用同一教师或经历同样轮次。转导训练效果限于已使用的目标集合，未见域需要另测。',
    ]
    cuecards[7]['purpose'] = '沿共享基座、三组五折adapter和九比一汇合解释在线结构'
    cuecards[8]['purpose'] = '沿操作符计算图比较整批秩空间和单条logit空间的聚合顺序'
    return updated, cuecards
