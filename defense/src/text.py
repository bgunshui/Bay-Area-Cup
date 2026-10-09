"""Text P7–P9 revision. Only apply() return values are changed; no file I/O."""
import html
import math
import re

SOURCE_SHA = "c11fa520f2f73d705e83cb3f40a6a214f8cc7f31"
PROVENANCE = {
    "teachers": ("text-track/evidence/验证输出/p4_report.json",
                 "text-track/docs/01_技术方案报告.md:148-164"),
    "soft": "text-track/code/fusion/mk_pseudo_v2.py:15-18",
    "confidence": "text-track/code/training/p43_14b_p2.py:105-107",
    "data_and_selection": "text-track/docs/01_技术方案报告.md:24-26,62-80",
    "candidate_auc": "text-track/evidence/关键实验结果.md:9-18",
    "public_claim_withdrawn": "text-track/docs/历史-早期23分支系统结果报告.md:45-49",
    "offline": "text-track/code/fusion/p97_p4_7b_sweep.py:12-38",
    "online": "text-track/code/docker_submit/server_dual_v4.py:206-233",
}
SAT_LOW = 0.5 - math.log(3) / 4
SAT_HIGH = 1 - SAT_LOW
UNIFORM_MEAN_WEIGHT = 2 * SAT_LOW + 2 * math.log(math.cosh(math.log(3) / 2))

CSS = r"""
.tx-page{padding:0!important;background:#f8f9fb!important;color:#11283d!important}
.tx-page .tx-paper{box-sizing:border-box;width:100vw!important;height:100vh!important;max-height:none!important;padding:7vh 4.5vw 9.26vh!important;background:#f8f9fb!important;color:#11283d!important;display:flex!important;flex-direction:column!important;overflow:hidden}
.tx-page *{box-sizing:border-box}
.tx-page .tx-title{font-family:Inter,"Noto Sans SC",sans-serif!important;font-size:clamp(40px,2.91667vw,56px)!important;line-height:1.15!important;font-weight:600!important;letter-spacing:-.02em!important;margin:0!important;color:#11283d!important}
.tx-page .tx-content{display:grid;grid-template-columns:1fr 1fr;gap:4vw;flex:1;min-height:0;margin-top:4.1vh;align-items:stretch}
.tx-page .tx-panel{display:flex;flex-direction:column;min-width:0;min-height:0}
.tx-page h3{font-family:Inter,"Noto Sans SC",sans-serif!important;font-size:clamp(24px,1.6667vw,32px)!important;line-height:1.2!important;font-weight:600!important;letter-spacing:0!important;margin:0 0 2.2vh!important;color:#174d6b!important}
.tx-page p,.tx-page .tx-label{font-size:clamp(22px,1.4583vw,28px)!important;line-height:1.35!important;font-weight:500!important;margin:0!important}
.tx-page .tx-detail{font-size:clamp(22px,1.3542vw,26px)!important;line-height:1.32!important;font-weight:500!important;color:#4c6579!important}
.tx-page .tx-num{font-variant-numeric:tabular-nums;letter-spacing:-.025em}
.tx-page .tx-note{margin-top:2.1vh!important;padding-top:1.7vh;border-top:1px solid #c9d4df;font-size:clamp(22px,1.3542vw,26px)!important;line-height:1.32!important;font-weight:500!important;color:#4c6579!important}
.tx-page .tx-em{color:#174d6b;font-weight:650}
.tx-page .tx-loop{position:relative;flex:1;min-height:0}
.tx-page .tx-loop-svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.tx-page .tx-loop-node{position:absolute;width:39%;height:32%;padding:1.6vh 1.1vw;background:#e9eff4;display:flex;flex-direction:column;justify-content:center;gap:.9vh}
.tx-page .tx-loop-node strong{font-size:clamp(24px,1.5625vw,30px);line-height:1.15;font-weight:600}
.tx-page .tx-loop-node .tx-loop-big{font-size:clamp(28px,2.0833vw,40px);line-height:1.12;color:#174d6b;font-weight:600}
.tx-page .tx-loop-node:nth-of-type(1){left:0;top:0}
.tx-page .tx-loop-node:nth-of-type(2){right:0;top:0}
.tx-page .tx-loop-node:nth-of-type(3){right:0;bottom:0}
.tx-page .tx-loop-node:nth-of-type(4){left:0;bottom:0}
.tx-page .tx-loop-core{position:absolute;left:36%;top:43%;width:28%;text-align:center;font-size:clamp(22px,1.3542vw,26px);font-weight:550;line-height:1.28;color:#4c6579}
.tx-page .tx-formula{display:flex;align-items:center;gap:1vw;font-size:clamp(22px,1.4583vw,28px);line-height:1.3;font-weight:550;margin-bottom:1vh;white-space:nowrap}
.tx-page .tx-key{display:inline-block;width:2.2vw;min-width:23px;height:0;border-top:4px solid #174d6b}
.tx-page .tx-key.conf{border-top:3px dashed #8199ac}
.tx-page .tx-plot{position:relative;flex:1;min-height:0;margin-top:1vh}
.tx-page .tx-plot svg{position:absolute;inset:0;width:100%;height:100%}
.tx-page .tx-tick{position:absolute;font-size:clamp(22px,1.25vw,24px);font-weight:550;line-height:1.1;color:#4c6579;white-space:nowrap}
.tx-page .tx-xtick{transform:translateX(-50%)}
.tx-page .tx-ytick{transform:translate(-100%,-50%)}
.tx-page .tx-point{position:absolute;font-size:clamp(22px,1.25vw,24px);font-weight:600;line-height:1.1;white-space:nowrap;color:#174d6b;background:#f8f9fb;padding:2px 5px}
.tx-page .tx-tree{position:relative;flex:1;min-height:0;margin-bottom:2.4vh}
.tx-page .tx-tree-svg{position:absolute;inset:0;width:100%;height:100%}
.tx-page .tx-base{position:absolute;top:0;width:27%;height:24%;background:#e9eff4;display:flex;flex-direction:column;justify-content:center;align-items:center;gap:.5vh}
.tx-page .tx-base .tx-model{font-size:clamp(28px,2.1875vw,42px);line-height:1.1;font-weight:600;color:#174d6b}
.tx-page .tx-base14{left:2%}.tx-page .tx-base7{left:53.75%}
.tx-page .tx-adapter-group{position:absolute;top:49%;width:31%;display:flex;flex-direction:column;align-items:center;gap:1.1vh}
.tx-page .tx-adapter-group:nth-of-type(3){left:0}.tx-page .tx-adapter-group:nth-of-type(4){left:34.5%}.tx-page .tx-adapter-group:nth-of-type(5){left:69%}
.tx-page .tx-group-name{font-size:clamp(24px,1.5625vw,30px);font-weight:600;line-height:1.15}
.tx-page .tx-adapter-row{display:grid;grid-template-columns:repeat(5,1fr);gap:.35vw;width:100%}
.tx-page .tx-adapter{padding:.65vh 0;text-align:center;background:#d9e5ee;color:#174d6b;font-size:clamp(22px,1.25vw,24px);line-height:1;font-weight:600}
.tx-page .tx-group-detail{text-align:center;font-size:clamp(22px,1.25vw,24px);font-weight:500;line-height:1.2;color:#4c6579}
.tx-page .tx-weight-labels{display:flex;justify-content:space-between;gap:2vw;font-size:clamp(22px,1.4583vw,28px);font-weight:550;line-height:1.2;margin-bottom:1.2vh}
.tx-page .tx-weight-bar{height:3.3vh;min-height:23px;width:100%}
.tx-page .tx-results{display:flex;flex-direction:column;gap:0}
.tx-page .tx-result{display:flex;align-items:center;justify-content:space-between;gap:1vw;padding:1.65vh 0;border-bottom:1px solid #d4dfe7}
.tx-page .tx-result:first-child{padding-top:0}
.tx-page .tx-result .tx-result-name{font-size:clamp(22px,1.4583vw,28px);font-weight:500;line-height:1.25}
.tx-page .tx-result .tx-result-value{font-size:clamp(26px,1.9792vw,38px);font-weight:600;line-height:1.15;color:#174d6b;font-variant-numeric:tabular-nums}
.tx-page .tx-candidate-limit{margin-top:auto;padding-top:2.2vh;font-size:clamp(22px,1.3542vw,26px);line-height:1.32;font-weight:500;color:#4c6579}
.tx-page .tx-infer-grid{grid-template-columns:1fr 1px 1fr;gap:3vw}
.tx-page .tx-infer-grid .vrule{position:static;width:1px;height:100%;background:#c9d4df;transform:none}
.tx-page .tx-lane{display:flex;flex-direction:column;gap:1.4vh;min-width:0;min-height:0}
.tx-page .tx-lane h3{margin-bottom:1.5vh!important}
.tx-page .tx-step{background:#e9eff4;padding:1.7vh 1.3vw;text-align:center;display:flex;flex-direction:column;justify-content:center;gap:.6vh;min-width:0}
.tx-page .tx-step .tx-step-main{font-size:clamp(24px,1.5625vw,30px);line-height:1.26;font-weight:550}
.tx-page .tx-step-output{background:#174d6b;color:#fff}
.tx-page .tx-step-output .tx-step-main{color:#fff}
.tx-page .tx-down{height:2vh;min-height:16px;display:flex;justify-content:center;align-items:center;flex:0 0 auto}
.tx-page .tx-down svg{width:2vw;height:100%;min-width:22px}
.tx-page .tx-pair{display:grid;grid-template-columns:1fr 1fr;gap:1vw}
.tx-page .tx-pair .tx-step{padding-left:.4vw;padding-right:.4vw}
.tx-page .tx-math{font-size:clamp(24px,1.5625vw,30px);line-height:1.3;font-weight:550;white-space:nowrap}
.tx-page .tx-math sub{font-size:inherit;position:relative;bottom:-.18em;vertical-align:baseline}
.tx-page sub{font-size:inherit;position:relative;bottom:-.18em;vertical-align:baseline}
@media(max-width:1450px){
.tx-page .tx-content{gap:3vw;margin-top:3.7vh}
.tx-page .tx-loop-node{padding:1.2vh .7vw;gap:.7vh}
.tx-page .tx-infer-grid{gap:2.4vw}
.tx-page .tx-lane{gap:1.1vh}
.tx-page .tx-step{padding:1.35vh .85vw}
.tx-page .tx-result{padding:1.35vh 0}
.tx-page .tx-formula{gap:.7vw;letter-spacing:-.025em}
}
"""


def soft(rank):
    return 1.0 / (1.0 + math.exp(-4.0 * (rank - 0.5)))


def confidence(rank):
    return min(1.0, max(0.0, 4.0 * abs(soft(rank) - 0.5)))


def _curves():
    # Exact plot transform: x = 70+540*r; y = 340-310*v.
    # Include analytic saturation abscissae, so the plateau knees are not
    # artifacts of a coarse sample grid. Text stays in HTML, never SVG.
    ranks = sorted({j / 200 for j in range(201)} | {SAT_LOW, 0.5, SAT_HIGH})
    pts = lambda fn: " ".join(f"{70+540*r:.4f},{340-310*fn(r):.4f}" for r in ranks)
    sat1, sat2 = 70 + 540 * SAT_LOW, 70 + 540 * SAT_HIGH
    return f'''<div class="tx-plot">
      <svg viewBox="0 0 650 390" preserveAspectRatio="none" role="img" aria-label="软标签sigmoid与饱和V形置信权重的确定性函数，横轴排序分位值0到1">
        <path d="M70 30V340H610" stroke="#8199ac" stroke-width="2" fill="none"/>
        <path d="M70 30H610M70 185H610M340 30V340" stroke="#d4dfe7" stroke-width="1.5" fill="none"/>
        <path d="M{sat1:.4f} 30V340M{sat2:.4f} 30V340" stroke="#b8c7d5" stroke-dasharray="5 7" fill="none"/>
        <polyline points="{pts(confidence)}" stroke="#8199ac" stroke-width="4" stroke-dasharray="9 6" fill="none" stroke-linejoin="round"/>
        <polyline points="{pts(soft)}" stroke="#174d6b" stroke-width="4" fill="none" stroke-linejoin="round"/>
        <rect x="65.5" y="{340-310*soft(0)-4.5:.4f}" width="9" height="9" fill="#174d6b"/>
        <rect x="605.5" y="{340-310*soft(1)-4.5:.4f}" width="9" height="9" fill="#174d6b"/>
      </svg>
      <span class="tx-tick tx-ytick" style="left:8.9%;top:7.692%">1</span>
      <span class="tx-tick tx-ytick" style="left:8.9%;top:47.436%">0.5</span>
      <span class="tx-tick tx-ytick" style="left:8.9%;top:87.179%">0</span>
      <span class="tx-tick" style="left:3%;top:0">y, w</span>
      <span class="tx-tick tx-xtick" style="left:10.769%;top:89.9%">0</span>
      <span class="tx-tick tx-xtick" style="left:{sat1/650*100:.4f}%;top:89.9%">.2253</span>
      <span class="tx-tick tx-xtick" style="left:52.308%;top:89.9%">.5</span>
      <span class="tx-tick tx-xtick" style="left:{sat2/650*100:.4f}%;top:89.9%">.7747</span>
      <span class="tx-tick tx-xtick" style="left:93.846%;top:89.9%">1</span>
      <span class="tx-tick" style="right:1%;top:97%">r</span>
      <span class="tx-point" style="left:13%;top:74.5%">.1192</span>
      <span class="tx-point" style="right:6.4%;top:20%">.8808</span>
    </div>'''


def _training_page():
    return f'''<div class="canvas-card tx-paper">
      <h2 class="tx-title" data-anim="head">文本：目标域自训练</h2>
      <div class="tx-content loop-diagram" data-anim="up">
        <div class="tx-panel loop-steps">
          <h3>教师 → 目标域 → 学生</h3>
          <div class="tx-loop">
            <svg class="tx-loop-svg loop-svg" viewBox="0 0 1000 500" preserveAspectRatio="none" aria-hidden="true">
              <path d="M390 80H605M805 165V335M610 420H395M195 335V165" stroke="#8199ac" stroke-width="2.5" fill="none"/>
              <path d="M587 70L607 80L587 90M795 315L805 335L815 315M413 410L393 420L413 430M185 185L195 165L205 185" stroke="#8199ac" stroke-width="2.5" fill="none"/>
            </svg>
            <div class="tx-loop-node"><strong>教师筛选</strong><span class="tx-loop-big">43 → 6</span><span class="tx-detail">非零权重分支</span></div>
            <div class="tx-loop-node"><strong>目标域预测</strong><span class="tx-loop-big">20,000</span><span class="tx-detail">无标签文本</span></div>
            <div class="tx-loop-node"><strong>软标签训练</strong><span class="tx-detail">伪标签按w加权<br>硬标签权重2.0</span></div>
            <div class="tx-loop-node"><strong>五折验证</strong><span class="tx-detail">800训练＋200验证<br>学生可更新教师</span></div>
            <div class="tx-loop-core">目标域<br>自适应</div>
          </div>
          <p class="tx-note">有效教师：ft 50% · robust 35% · llm 15%<br>公共检测器11个候选，<span class="tx-em">有效权重0</span></p>
        </div>
        <div class="tx-panel">
          <h3>软标签与置信权重</h3>
          <div class="tx-formula"><span class="tx-key"></span><span>y(r) = σ[4(r − 0.5)]</span></div>
          <div class="tx-formula"><span class="tx-key conf"></span><span>w(r) = min(1, 4|y(r) − 0.5|)</span></div>
          {_curves()}
          <p class="tx-note">L = mean[wᵢ · BCE(zᵢ, yᵢ)]<br>中点权重0，排序两端饱和到1。</p>
        </div>
      </div>
    </div>'''


def _adapter_group(left, group):
    blocks = "".join(f'<span class="tx-adapter">{j}</span>' for j in range(5))
    return f'''<div class="tx-adapter-group" style="left:{left}%">
      <span class="tx-group-name">{group}</span><div class="tx-adapter-row" aria-label="五折适配器0至4">{blocks}</div>
      <span class="tx-group-detail">5个适配器</span>
    </div>'''


def _ensemble_page():
    candidates = [
        ("多分支教师", "0.978284"),
        ("14B自训练候选", "0.979964"),
        ("双14B seed", "0.980896"),
        ("14B p4五折", "0.981069"),
        ("14B＋7B", "0.981301"),
    ]
    rows = "".join(f'<div class="tx-result"><span class="tx-result-name">{n}</span><span class="tx-result-value">{v}</span></div>' for n, v in candidates)
    return f'''<div class="canvas-card tx-paper">
      <h2 class="tx-title" data-anim="head">文本：两基座，15个适配器</h2>
      <div class="tx-content" data-anim="up">
        <div class="tx-panel">
          <h3>基座复用 · 五折集成</h3>
          <div class="tx-tree">
            <svg class="tx-tree-svg" viewBox="0 0 1000 400" preserveAspectRatio="none" aria-hidden="true">
              <path d="M155 96V188M672.5 96V147H500V188M672.5 147H845V188" stroke="#8199ac" stroke-width="2.5" fill="none"/>
              <path d="M145 170L155 190L165 170M490 170L500 190L510 170M835 170L845 190L855 170" stroke="#8199ac" stroke-width="2.5" fill="none"/>
            </svg>
            <div class="tx-base tx-base14"><span class="tx-detail">Qwen2.5</span><span class="tx-model">14B</span></div>
            <div class="tx-base tx-base7"><span class="tx-detail">Qwen2.5</span><span class="tx-model">7B</span></div>
            {_adapter_group(0,"p4")}{_adapter_group(34.5,"r3a")}{_adapter_group(69,"r3b")}
          </div>
          <div class="tx-weight-labels"><span>14B <strong class="tx-em">90%</strong></span><span>7B两组共 <strong class="tx-em">10%</strong></span></div>
          <svg class="tx-weight-bar" viewBox="0 0 1000 40" preserveAspectRatio="none" role="img" aria-label="最终融合参数，14B权重90%，7B两组合计10%">
            <rect width="900" height="40" fill="#174d6b"/><rect x="900" width="100" height="40" fill="#a7bacb"/>
          </svg>
          <p class="tx-note">两个基座常驻，逐adapter切换。<br>每条文本：<span class="tx-em">5＋5＋5次前向</span>。</p>
        </div>
        <div class="tx-panel">
          <h3>候选方案结果 · 复赛 AUC</h3>
          <div class="tx-results">{rows}</div>
          <div class="tx-candidate-limit">双seed未超过p4五折。<br>最终组合增量 <span class="tx-em">＋0.000232</span>。</div>
        </div>
      </div>
    </div>'''


def _arrow():
    return '''<div class="tx-down" aria-hidden="true"><svg viewBox="0 0 40 25"><path d="M20 0V20M10 10L20 22L30 10" stroke="#8199ac" stroke-width="2.5" fill="none"/></svg></div>'''


def _inference_page():
    arrow = _arrow()
    return f'''<div class="canvas-card tx-paper">
      <h2 class="tx-title" data-anim="head">文本：批排名与单条推理</h2>
      <div class="tx-content tx-infer-grid duo-compare" data-anim="up">
        <div class="tx-lane duo-half">
          <h3>整批目标集合 · 秩空间</h3>
          <div class="tx-step"><span class="tx-step-main">目标集合预测</span><span class="tx-detail">14B p4 ／ 7B r3a、r3b</span></div>{arrow}
          <div class="tx-step"><span class="tx-step-main">各分支排名百分位</span><span class="tx-math">r<sub>14</sub>，r<sub>a</sub>，r<sub>b</sub></span></div>{arrow}
          <div class="tx-step"><span class="tx-step-main">先在排名空间加权</span><span class="tx-math">0.9r<sub>14</sub> ＋ 0.05r<sub>a</sub> ＋ 0.05r<sub>b</sub></span></div>{arrow}
          <div class="tx-step tx-step-output"><span class="tx-step-main">整批rank → 检测分数</span></div>
        </div>
        <span class="vrule" aria-hidden="true"></span>
        <div class="tx-lane duo-half">
          <h3>单条输入 · logit空间</h3>
          <div class="tx-pair"><div class="tx-step"><span class="tx-step-main">14B · 5个logit</span></div><div class="tx-step"><span class="tx-step-main">7B · 10个logit</span></div></div>{arrow}
          <div class="tx-pair"><div class="tx-step"><span class="tx-step-main">μ<sub>14</sub> = mean(5)</span></div><div class="tx-step"><span class="tx-step-main">μ<sub>7</sub> = mean(10)</span></div></div>{arrow}
          <div class="tx-pair"><div class="tx-step"><span class="tx-step-main">p<sub>14</sub> = σ(μ<sub>14</sub>)</span></div><div class="tx-step"><span class="tx-step-main">p<sub>7</sub> = σ(μ<sub>7</sub>)</span></div></div>{arrow}
          <div class="tx-step tx-step-output"><span class="tx-step-main">0.9p<sub>14</sub> ＋ 0.1p<sub>7</sub></span></div>
          <p class="tx-detail">7B两组共10个logit先平均，<br>再做<span class="tx-em">一次sigmoid</span>。</p>
        </div>
      </div>
      <p class="tx-note">聚合空间不同，<span class="tx-em">不保证同排序</span>；sigmoid分数仍需校准。在线输入默认截断至512 tokens。</p>
    </div>'''


def _section(previous, sid, title, layout, recipe, body):
    match = re.match(r"\s*(<section\b[^>]*>)", previous)
    if not match or f'data-slide-id="{sid}"' not in match.group(1):
        raise ValueError("Unexpected existing slide: " + sid)
    tag = match.group(1)
    for name, value in (
        ("class", "slide light tx-page"), ("data-title", title),
        ("data-layout", layout), ("data-animate", recipe),
    ):
        attr = name + '="' + html.escape(value, quote=True) + '"'
        if re.search(r"\b" + name + r'="[^"]*"', tag):
            tag = re.sub(r"\b" + name + r'="[^"]*"', attr, tag, count=1)
        else:
            tag = tag[:-1] + " " + attr + ">"
    return tag + body + "</section>"


def apply(pages, notes):
    """Return copies with only the three text IDs replaced; CSS is injected by parent."""
    if len(pages) != len(notes):
        raise ValueError("Page/note lengths differ")
    revised_pages, revised_notes = list(pages), [dict(n) for n in notes]
    definitions = [
        ("text-self-training", "文本：目标域自训练", "S14", "loop-form", 80, _training_page,
         "解释有效教师、目标域适应和标签置信权重",
         [
             "教师候选池43个，按家族权重乘成员权重，仅6个带权：robust 0.35；7B官方s2/s3各0.25；4B hidden mean 0.075；qwen pair和4B hidden last各0.0375。public的11个候选有效权重全为0。",
             "1,000条有标签样例分五折；每折800条训练、200条验证，另加入全量20,000条无标签目标文本。这属于转导自训练，目标集合效果不能直接外推到未见目标域。",
             "r为并列平均秩除以N＋1；软标签y=σ[4(r−0.5)]，理论端点0.119203和0.880797。置信权重w=min(1,4|y−0.5|)，中心0、两端1；饱和阈值0.22534693和0.77465307。这是确定性公式图，权重不是标签正确概率。",
             "无并列且按连续均匀r近似时，饱和区占45.0694%，平均权重0.73837593；20,000条的权重总和约14767.52，比800×2约9.23倍。它不是统计有效样本量ESS，也不是实测训练收益，因此这些推导只作问答补充。",
             "验证AUC选择checkpoint，OOF AUC选择融合。闭环表示可迭代方法；未宣称全部最终adapter完整经历相同三轮。旧公共检测器泛化最强说法已被目标集反馈否定，后续主教师public权重为0；不能据候选池数量宣传泛化。",
         ], "下一页说明最终两种基座与15个适配器如何组合。"),
        ("text-ensemble", "文本：两基座，15个适配器", "S04", "six-cells", 70, _ensemble_page,
         "展示真实模型树、最终融合权重与候选结果",
         [
             "Qwen2.5-14B-Instruct的p4一组五折共5个adapter；Qwen2.5-7B-Instruct的r3a和r3b各5个。两个基座共享、共15个adapter，不是7B十折交叉验证。",
             "基座常驻，每条文本轮换adapter执行5＋5＋5次前向。最终14B权重0.9，7B两组合计0.1；它们是融合参数，不代表各分支独立因果贡献。",
             "候选LB AUC分别为多分支教师0.978284、14B自训练0.979964、双14B seed 0.980896、14B p4五折0.981069、最终14B＋7B 0.981301。早期0.9615的模型身份未明确，页面不纳入该点。",
             "双seed低于p4五折；最终比p4高0.000232。这些候选结果未固定所有变量，也缺配对显著性检验，不据此宣称多seed或异构融合总会更好。",
             "最终成绩来自队伍成绩口径，尚无官方标签用于独立重算；15个adapter是配置结构，实际服务还需要断言加载数量、设备和预测方向。这里没有宣称已完成最终容器验收。",
         ], "结构相同也不保证计算链相同，下面分解两条融合路径。"),
        ("text-offline-service", "文本：批排名与单条推理", "S08", "duo-mirror", 60, _inference_page,
         "解释整批秩空间融合与单条logit融合为何不等价",
         [
             "整批路径将14B p4和7B两组目标集合预测各自转成排名百分位；7B两组等权，以0.9 r14＋0.05 ra＋0.05 rb融合，再对整批做最终rank。",
             "14B训练脚本的目标集合预测先对逐折预测分别rank、按折平均、再rank。先rank再平均一般不等于先平均再rank；页面用分支预测简写这些内层步骤，不能擅自替换成rank(raw mean)。",
             "在线14B五个logit共同平均，7B两组共十个logit共同平均；两侧各做一次sigmoid，再按0.9与0.1加权。7B不是两组各sigmoid后再平均。服务内部sigmoid对logit作−30到30夹取。",
             "排名空间与logit空间不同，不保证同排序。给定20,000行输出的排序相关约0.9998697006，样本对排序变化218571对、约0.10929%；这不是分类错误率或实际AUC损失，缺真标签不能推得服务同分。",
             "在线输入显式截断为最多512 tokens，长文本尾部可能被忽略；检测分数未经概率校准。目标域、长度与并发评测需另外设计。",
         ], "接着看两个任务的整体检测效果与适用边界。"),
    ]
    for sid, title, layout, recipe, seconds, builder, purpose, talk, transition in definitions:
        hits = [i for i, page in enumerate(pages) if f'data-slide-id="{sid}"' in page]
        if len(hits) != 1:
            raise ValueError("Expected exactly one page: " + sid)
        i = hits[0]
        if notes[i].get("id") != sid:
            raise ValueError("Note/page ID mismatch: " + sid)
        revised_pages[i] = _section(pages[i], sid, title, layout, recipe, builder())
        revised_notes[i].update(
            title=title, section="文本检测", purpose=purpose, minutes=seconds / 60,
            talk=talk, transition=transition)
        revised_notes[i].pop("autoAdvanceSeconds", None)
    return revised_pages, revised_notes
