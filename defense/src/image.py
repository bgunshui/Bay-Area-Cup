"""Image-track scientific diagrams for slides 4--6.

Facts are pinned to c11fa520f2f73d705e83cb3f40a6a214f8cc7f31.
CSV Spearman values were independently recomputed with average ranks.
Only page indices 3/4/5 and their note wording are replaced.
"""

from html import escape


CSS = r"""
.im-page{background:#f8f9fb!important;color:#11283d!important}
.im-page .im-frame{height:100vh;box-sizing:border-box;padding:7vh 4.5vw 9vh;display:flex;flex-direction:column;gap:3.5vh;overflow:hidden}
.im-page .im-title{margin:0;font-size:clamp(40px,2.916667vw,56px);font-weight:600;line-height:1.25;letter-spacing:-.025em;color:#11283d;padding-bottom:2.4vh;border-bottom:2px solid #11283d;flex:none}
.im-page .im-content{flex:1;min-height:0;display:flex;flex-direction:column;gap:2.1vh;font-size:clamp(22px,1.458333vw,28px);line-height:1.35}
.im-page .im-svg-arrow{display:block;width:100%;height:3.3vh;min-height:24px;max-height:38px}
.im-page .im-math{font-family:'Cambria Math','STIX Two Math','Times New Roman',serif;font-size:clamp(24px,1.666667vw,32px);font-variant-numeric:tabular-nums;line-height:1.35;white-space:nowrap}
.im-page sub,.im-page sup{font-size:22px;line-height:0;position:relative;vertical-align:baseline}
.im-page sub{bottom:-.25em}.im-page sup{top:-.4em}
.im-page .im-train-grid{flex:1;min-height:0;display:grid;grid-template-columns:22fr 3fr 24fr 3fr 38fr 3fr 7fr;grid-template-rows:4.3vh 1.32fr 1fr 1fr 1fr;gap:.8vh 0}
.im-page .im-th{font-size:clamp(24px,1.5625vw,30px);font-weight:500;border-bottom:2px solid #174d6b;padding-bottom:1vh;align-self:end}
.im-page .im-src{border-left:3px solid #9eb6c5;background:#eef2f6;padding:1.2vh .9vw;display:flex;flex-direction:column;justify-content:center;gap:.6vh;min-width:0}
.im-page .im-src strong{font-size:clamp(24px,1.5625vw,30px);font-weight:500}
.im-page .im-src span{font-size:clamp(22px,1.354167vw,26px);color:#465f72;line-height:1.35}
.im-page .im-model-cell{padding:1.1vh .6vw;display:flex;flex-direction:column;justify-content:center;gap:.65vh;border-bottom:1px solid #ced9e2;min-width:0}
.im-page .im-model-name{font-size:clamp(24px,1.5625vw,30px);font-weight:500;line-height:1.25;white-space:nowrap}
.im-page .im-model-spec{font-size:clamp(22px,1.354167vw,26px);color:#465f72;white-space:nowrap}
.im-page .im-arrow-cell{display:flex;align-items:center;padding:0 .25vw}
.im-page .im-seeds{display:grid;grid-template-columns:repeat(2,1fr);gap:.65vh .65vw;padding:1.1vh .5vw;align-content:center;min-width:0;border-bottom:1px solid #ced9e2}
.im-page .im-seed{font-size:clamp(22px,1.354167vw,26px);padding:.7vh .6vw;background:#fff;border:1px solid #c7d5df;white-space:nowrap;font-variant-numeric:tabular-nums;text-align:center;line-height:1.25}
.im-page .im-one-adapter{padding:1.1vh .5vw;display:flex;align-items:center;border-bottom:1px solid #ced9e2}
.im-page .im-one-adapter .im-seed{width:100%;box-sizing:border-box}
.im-page .im-count{display:flex;flex-direction:column;align-items:center;justify-content:center;color:#174d6b;border-bottom:1px solid #ced9e2;font-size:clamp(42px,3.125vw,60px);font-weight:500;line-height:1}
.im-page .im-count span{font-size:22px;font-weight:400;margin-top:.7vh}
.im-page .im-common{border-top:2px solid #174d6b;padding-top:1.6vh;display:flex;align-items:center;justify-content:space-between;gap:2vw;min-width:0}
.im-page .im-common strong{font-weight:500;font-size:clamp(24px,1.5625vw,30px)}
.im-page .im-common span{font-size:clamp(22px,1.354167vw,26px);color:#465f72}
.im-page .im-calc-layout{flex:1;min-height:0;display:grid;grid-template-columns:70fr 3fr 27fr;gap:1vw}
.im-page .im-calc-main{display:flex;flex-direction:column;min-height:0;gap:.9vh}
.im-page .im-calc-intro{display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid #174d6b;padding-bottom:1.25vh;font-size:clamp(24px,1.5625vw,30px)}
.im-page .im-slots{display:grid;grid-template-columns:1.4fr 1fr 1fr;gap:1.1vw;min-height:0}
.im-page .im-slot{background:#fff;border-top:4px solid #174d6b;border-bottom:1px solid #c7d5df;padding:1vh .9vw;display:flex;flex-direction:column;gap:.8vh;min-width:0}
.im-page .im-slot h3{margin:0;font-size:clamp(17px,1.1979vw,23px);font-weight:500;line-height:1.25}
.im-page .im-member-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.5vw}
.im-page .im-member{background:#eef2f6;padding:.6vh 0;text-align:center;font-size:clamp(22px,1.354167vw,26px);line-height:1.2;white-space:nowrap}
.im-page .im-member-grid .im-supervised{grid-column:1/-1;color:#174d6b;background:#e1ebf1}
.im-page .im-member-pair{display:grid;grid-template-columns:1fr 1fr;gap:.5vw}
.im-page .im-slot .im-math{font-size:clamp(22px,1.458333vw,28px);text-align:center;margin-top:auto;line-height:1.5}
.im-page .im-slot-spec{color:#465f72;font-size:clamp(16px,1.1458vw,22px);text-align:center;line-height:1.4}
.im-page .im-merge{height:3.1vh;min-height:22px;flex:none;width:100%}
.im-page .im-base-to-z{min-height:16px;flex:1;width:100%;position:relative}
.im-page .im-base{background:#e8eff4;border:1px solid #aabfce;text-align:center;padding:1vh .7vw;font-size:clamp(26px,1.875vw,36px);line-height:1.5}
.im-page .im-base .im-math{font-size:clamp(26px,1.875vw,36px)}
.im-page .im-prior-mix{display:grid;grid-template-columns:1.15fr 3vw 1fr;gap:1vw;align-items:center}
.im-page .im-prior{border-top:2px solid #a2b6c6;padding:.9vh .7vw;background:#fff;display:flex;flex-direction:column;gap:.5vh;text-align:center}
.im-page .im-prior strong{font-size:clamp(24px,1.5625vw,30px);font-weight:500}
.im-page .im-prior span{font-size:clamp(22px,1.354167vw,26px);color:#465f72;line-height:1.35}
.im-page .im-z{background:#174d6b;color:#fff;text-align:center;padding:2vh .7vw;line-height:1.6}
.im-page .im-z .im-math{font-size:clamp(26px,1.875vw,36px)}
.im-page .im-z span{display:block;font-size:clamp(22px,1.354167vw,26px);margin-top:.5vh;color:#e7eff4}
.im-page .im-final-arrow{display:flex;align-items:flex-end;padding-bottom:4.8vh}
.im-page .im-ref-panel{border-left:1px solid #c7d5df;padding-left:1.7vw;display:flex;flex-direction:column;min-width:0;gap:1vh}
.im-page .im-ref-panel h3{margin:0;font-size:clamp(26px,1.770833vw,34px);font-weight:500;line-height:1.3}
.im-page .im-ref-sub{font-size:clamp(22px,1.354167vw,26px);color:#465f72;line-height:1.35}
.im-page .im-ref-plot{position:relative;flex:1;min-height:23vh;max-height:37vh}
.im-page .im-plot-svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.im-page .im-ref-y{position:absolute;left:-.2vw;top:44%;transform:rotate(-90deg);transform-origin:left top;font-size:clamp(22px,1.354167vw,26px);white-space:nowrap}
.im-page .im-ref-x{position:absolute;bottom:0;left:18%;right:0;text-align:center;font-size:clamp(22px,1.354167vw,26px);white-space:nowrap}
.im-page .im-ref-query{position:absolute;left:70%;top:82%;font-size:clamp(22px,1.354167vw,26px);font-family:'Cambria Math','Times New Roman',serif}
.im-page .im-ref-result{position:absolute;left:19%;top:21%;font-size:clamp(22px,1.354167vw,26px);font-family:'Cambria Math','Times New Roman',serif;color:#174d6b}
.im-page .im-frac{display:inline-flex;flex-direction:column;align-items:center;vertical-align:middle;line-height:1.12;font-size:clamp(22px,1.354167vw,26px)}
.im-page .im-frac>span:first-child{border-bottom:1px solid currentColor;padding:0 .2em .15em}.im-page .im-frac>span:last-child{padding-top:.12em}
.im-page .im-ref-formula{text-align:center;padding:1.2vh 0;border-top:1px solid #c7d5df;white-space:nowrap}
.im-page .im-output{margin-top:auto;padding:1.65vh .7vw;border:2px solid #174d6b;text-align:center;background:#fff}
.im-page .im-output .im-math{font-size:clamp(26px,1.770833vw,34px)}
.im-page .im-techline{margin:0;font-size:clamp(22px,1.354167vw,26px);color:#465f72;line-height:1.35}
.im-page .im-eval-grid{flex:1;min-height:0;display:grid;grid-template-columns:1.35fr 1fr;gap:3vw}
.im-page .im-auc-panel{display:flex;flex-direction:column;min-width:0;gap:1.4vh}
.im-page .im-panel-head{font-size:clamp(26px,1.770833vw,34px);font-weight:500;margin:0;line-height:1.3}
.im-page .im-auc-plot{position:relative;flex:1;min-height:32vh;max-height:52vh}
.im-page .im-y-tick{position:absolute;left:0;width:8.2%;text-align:right;transform:translateY(-50%);font-size:clamp(22px,1.354167vw,26px);color:#465f72;font-variant-numeric:tabular-nums}
.im-page .im-point-label{position:absolute;transform:translate(-50%,-100%);font-size:clamp(24px,1.5625vw,30px);font-weight:500;white-space:nowrap;font-variant-numeric:tabular-nums;color:#174d6b;background:#f8f9fb;padding:0 .2vw}
.im-page .im-x-tick{position:absolute;top:89%;transform:translateX(-50%);text-align:center;font-size:clamp(22px,1.354167vw,26px);color:#465f72;white-space:nowrap}
.im-page .im-auc-gain{border-top:2px solid #174d6b;padding:1.6vh 0;display:flex;align-items:center;justify-content:space-between;gap:1vw;font-size:clamp(22px,1.458333vw,28px);font-variant-numeric:tabular-nums}
.im-page .im-auc-gain strong{font-size:clamp(30px,2.1875vw,42px);font-weight:500;color:#174d6b;white-space:nowrap}
.im-page .im-diagnostics{border-left:1px solid #c7d5df;padding-left:2.2vw;display:flex;flex-direction:column;gap:2.7vh;min-width:0}
.im-page .im-corr-panel{display:flex;flex-direction:column;gap:1.4vh;min-height:0}
.im-page .im-corr-axis{display:flex;justify-content:space-between;margin-left:0;font-size:22px;color:#465f72}
.im-page .im-corr-row{display:flex;flex-direction:column;gap:.45vh}
.im-page .im-corr-label{display:flex;justify-content:space-between;align-items:center;gap:.6vw;font-size:clamp(22px,1.354167vw,26px);line-height:1.3;white-space:nowrap}
.im-page .im-corr-label strong{font-weight:500;font-variant-numeric:tabular-nums;color:#174d6b}
.im-page .im-corr-bar{width:100%;height:2.4vh;min-height:16px;display:block}
.im-page .im-prior-panel{border-top:1px solid #c7d5df;padding-top:1.9vh;flex:1;min-height:0;display:flex;flex-direction:column;gap:1.1vh}
.im-page .im-prior-plot{position:relative;flex:1;min-height:18.7vh;max-height:25vh}
.im-page .im-prior-tick{position:absolute;left:0;width:12.8%;text-align:right;transform:translateY(-50%);font-size:22px;color:#465f72;font-variant-numeric:tabular-nums}
.im-page .im-prior-value{position:absolute;transform:translate(-50%,-100%);font-size:clamp(22px,1.354167vw,26px);font-variant-numeric:tabular-nums;color:#174d6b;background:#f8f9fb;padding:0 .1vw;white-space:nowrap}
.im-page .im-prior-x{position:absolute;top:86%;transform:translateX(-50%);font-size:22px;color:#465f72;text-align:center;white-space:nowrap}
"""


def _arrow(cls="im-svg-arrow"):
    return (f'<svg class="{cls}" viewBox="0 0 40 32" aria-hidden="true">'
            '<path d="M2 16H34M26 8L35 16L26 24" fill="none" '
            'stroke="#174d6b" stroke-width="2.4"/></svg>')


def _page(sid, title, layout, body):
    return (f'<section class="slide light im-page" data-slide-id="{sid}" '
            f'data-layout="{layout}" data-animate="academic-image" '
            f'data-title="{escape(title, quote=True)}">'
            f'<div class="im-frame"><h2 class="im-title">{title}</h2>'
            f'<div class="im-content">{body}</div></div></section>')


def _training():
    arrow = _arrow()
    body = f"""
    <div class="im-train-grid">
      <div class="im-th" style="grid-column:1;grid-row:1">训练来源</div>
      <div class="im-th" style="grid-column:3;grid-row:1">冻结骨干</div>
      <div class="im-th" style="grid-column:5;grid-row:1">LoRA＋分类头</div>
      <div class="im-th im-th-count" style="grid-column:7;grid-row:1">数量</div>

      <div class="im-src" style="grid-column:1;grid-row:2"><strong>转导蒸馏</strong><span>20,000目标软标签<br>1,000样例硬标签（w=4）</span></div>
      <div class="im-arrow-cell" style="grid-column:2;grid-row:2">{arrow}</div>
      <div class="im-model-cell" style="grid-column:3;grid-row:2"><div class="im-model-name">CLIP ViT-H-14</div><div class="im-model-spec">224 · r=32 · α=32</div></div>
      <div class="im-arrow-cell" style="grid-column:4;grid-row:2">{arrow}</div>
      <div class="im-seeds" style="grid-column:5;grid-row:2"><div class="im-seed">20261003 · 2轮</div><div class="im-seed">20261101 · 4轮</div><div class="im-seed">20261102 · 4轮</div><div class="im-seed">20261103 · 4轮</div></div>
      <div class="im-arrow-cell" style="grid-column:6;grid-row:2">{arrow}</div>
      <div class="im-count" style="grid-column:7;grid-row:2">4<span>分量</span></div>

      <div class="im-src" style="grid-column:1;grid-row:3/5"><strong>独立监督</strong><span>Tiny-GenImage 28.8k<br>样例800训练／200验证</span></div>
      <div class="im-arrow-cell" style="grid-column:2;grid-row:3">{arrow}</div>
      <div class="im-model-cell" style="grid-column:3;grid-row:3"><div class="im-model-name">CLIP ViT-bigG-14</div><div class="im-model-spec">224 · r=16 · α=32</div></div>
      <div class="im-arrow-cell" style="grid-column:4;grid-row:3">{arrow}</div>
      <div class="im-seeds" style="grid-column:5;grid-row:3"><div class="im-seed">20261004 · 2轮</div><div class="im-seed">20261005 · 2轮</div></div>
      <div class="im-arrow-cell" style="grid-column:6;grid-row:3">{arrow}</div>
      <div class="im-count" style="grid-column:7;grid-row:3">2<span>分量</span></div>

      <div class="im-arrow-cell" style="grid-column:2;grid-row:4">{arrow}</div>
      <div class="im-model-cell" style="grid-column:3;grid-row:4"><div class="im-model-name">CLIP ViT-bigG-14</div><div class="im-model-spec">384 · r=32 · α=64</div></div>
      <div class="im-arrow-cell" style="grid-column:4;grid-row:4">{arrow}</div>
      <div class="im-one-adapter" style="grid-column:5;grid-row:4"><div class="im-seed">384分辨率的独立训练分量</div></div>
      <div class="im-arrow-cell" style="grid-column:6;grid-row:4">{arrow}</div>
      <div class="im-count" style="grid-column:7;grid-row:4">1<span>分量</span></div>

      <div class="im-src" style="grid-column:1;grid-row:5"><strong>独立监督</strong><span>GenImage++ 78k<br>＋官方样例</span></div>
      <div class="im-arrow-cell" style="grid-column:2;grid-row:5">{arrow}</div>
      <div class="im-model-cell" style="grid-column:3;grid-row:5"><div class="im-model-name">SigLIP2-so400m</div><div class="im-model-spec">384 · r=16 · α=32</div></div>
      <div class="im-arrow-cell" style="grid-column:4;grid-row:5">{arrow}</div>
      <div class="im-one-adapter" style="grid-column:5;grid-row:5"><div class="im-seed">纯监督适配分量</div></div>
      <div class="im-arrow-cell" style="grid-column:6;grid-row:5">{arrow}</div>
      <div class="im-count" style="grid-column:7;grid-row:5">1<span>分量</span></div>
    </div>
    <div class="im-common"><strong>骨干冻结，更新低秩参数与分类头</strong><span>JPEG增强 · 各分量独立预处理</span><div class="im-math">W′ = W ＋ (α/r)BA</div></div>
    """
    return _page("image-training", "图像训练：转导学生与监督分量", "S05", body)


def _ecdf_geometry():
    # Function-form illustration; these steps are not an empirical ROC curve.
    return """
    <div class="im-ref-plot" role="img" aria-label="固定参考分布的ECDF阶梯函数示意">
      <svg class="im-plot-svg" viewBox="0 0 420 340" preserveAspectRatio="none" aria-hidden="true">
        <path d="M65 25V280H395" fill="none" stroke="#738e9f" stroke-width="2"/>
        <path d="M65 263H109V226H152V192H201V148H247V107H294V66H341V35H393" fill="none" stroke="#174d6b" stroke-width="3.6"/>
        <path d="M294 66V280M65 66H294" fill="none" stroke="#789aac" stroke-width="1.6" stroke-dasharray="6 6"/>
        <circle cx="294" cy="66" r="5.6" fill="#174d6b"/>
      </svg>
      <div class="im-ref-y">百分位</div><div class="im-ref-x">原始分／融合分</div>
      <div class="im-ref-query">x</div><div class="im-ref-result">F<sub>ref</sub>(x)</div>
    </div>
    """


def _fusion():
    arrow = _arrow()
    body = f"""
    <div class="im-calc-layout">
      <div class="im-calc-main">
        <div class="im-calc-intro"><strong>8个分量输出</strong><div class="im-math">s<sub>i</sub> → p<sub>i</sub> = F<sub>i</sub>(s<sub>i</sub>)</div></div>
        <div class="im-slots">
          <div class="im-slot"><h3>主槽 · 3/4</h3><div class="im-slot-spec">4个CLIP-H＋1个SigLIP2</div><div class="im-math">m<sub>H</sub> = (H<sub>1</sub>＋H<sub>2</sub>＋H<sub>3</sub>＋H<sub>4</sub>＋S) / 5</div></div>
          <div class="im-slot"><h3>bigG224槽 · 1/7</h3><div class="im-slot-spec">bigG@224 × 2</div><div class="im-math">m<sub>B</sub> = (B<sub>1</sub>＋B<sub>2</sub>) / 2</div></div>
          <div class="im-slot"><h3>bigG@384槽 · 3/28</h3><div class="im-slot-spec">1个bigG@384</div><div class="im-math">m<sub>384</sub> = B<sub>384</sub></div></div>
        </div>
        <svg class="im-merge" viewBox="0 0 1000 70" preserveAspectRatio="none" aria-hidden="true"><path d="M225 2V26H500M600 2V26H500M870 2V26H500V61M490 49L500 62L510 49" fill="none" stroke="#174d6b" stroke-width="2.2"/></svg>
        <div class="im-base"><span class="im-math">base = (3/4)m<sub>H</sub> ＋ (1/7)m<sub>B</sub> ＋ (3/28)m<sub>384</sub></span></div>
        <div class="im-base-to-z"><svg viewBox="0 0 1000 40" preserveAspectRatio="none" aria-hidden="true" style="position:absolute;inset:0;width:100%;height:100%"><path d="M760 0V39" fill="none" stroke="#174d6b" stroke-width="2.2" vector-effect="non-scaling-stroke"/></svg><svg viewBox="0 0 24 18" aria-hidden="true" style="position:absolute;left:76%;bottom:0;width:24px;height:18px;transform:translateX(-50%)"><path d="M3 4L12 15L21 4" fill="none" stroke="#174d6b" stroke-width="2.2"/></svg></div>
        <div class="im-prior-mix">
          <div class="im-prior"><strong>尺寸先验 prior</strong><span>0.4×桶ECDF ＋ 0.6×GBM ECDF<br>→ 先验ECDF</span></div>
          {arrow}
          <div class="im-z"><div class="im-math">z = 0.8base ＋ 0.2prior</div><span>先验在后融合注入</span></div>
        </div>
      </div>
      <div class="im-final-arrow">{arrow}</div>
      <div class="im-ref-panel"><h3>固定参考ECDF</h3><div class="im-ref-sub">N = 20,000 · 函数示意</div>{_ecdf_geometry()}<div class="im-ref-formula"><span class="im-math">F<sub>ref</sub>(x) = </span><span class="im-frac"><span>lo ＋ hi ＋ 1</span><span>2(N ＋ 1)</span></span></div><div class="im-output"><div class="im-math">score = F<sub>final</sub>(z)</div></div></div>
    </div>
    <p class="im-techline">每次查询使用固定参考分布；尺寸先验依赖尺寸与文件属性分布。</p>
    """
    return _page("image-ecdf", "图像融合：三槽计算与逐图ECDF", "S17", body)


def _auc_curve():
    values = [.908114, .916175, .917360, .918887]
    names = ["基线", "基底与权重", "多种子", "监督互补"]
    xs = [155, 385, 615, 845]
    low, high, top, bottom = .905, .920, 75, 510
    yy = lambda v: bottom - (v-low)/(high-low)*(bottom-top)
    lines, labels = [], []
    for v in [.905, .910, .915, .920]:
        y = yy(v)
        lines.append(f'<line x1="95" x2="950" y1="{y:.3f}" y2="{y:.3f}" stroke="#d3dfe7" stroke-width="1.4"/>')
        labels.append(f'<span class="im-y-tick" style="top:{y/6:.4f}%">{v:.3f}</span>')
    points = " ".join(f"{x},{yy(v):.3f}" for x,v in zip(xs,values))
    marks = [f'<circle cx="{x}" cy="{yy(v):.3f}" r="7.8" fill="#174d6b" stroke="#f8f9fb" stroke-width="3"/>' for x,v in zip(xs,values)]
    for i,(x,v,name) in enumerate(zip(xs,values,names)):
        # First point: place its label below/right to clear the steep segment
        # and the vertical axis, while retaining the actual point position.
        label_x = 20.5 if i == 0 else x/10
        label_y = yy(v)/6+12 if i == 0 else yy(v)/6-3
        labels.append(f'<span class="im-point-label" style="left:{label_x:.2f}%;top:{label_y:.4f}%">{v:.6f}</span>')
        labels.append(f'<span class="im-x-tick" style="left:{x/10:.2f}%">{name}</span>')
    return ('<div class="im-auc-plot" role="img" aria-label="四种融合方案复赛AUC为0.908114、0.916175、0.917360、0.918887">'
            '<svg class="im-plot-svg" viewBox="0 0 1000 600" preserveAspectRatio="none" aria-hidden="true">'
            + "".join(lines) + '<path d="M95 75V510H950" fill="none" stroke="#738e9f" stroke-width="2"/>'
            + f'<polyline points="{points}" fill="none" stroke="#174d6b" stroke-width="4" stroke-linejoin="round"/>'
            + "".join(marks) + '</svg>' + "".join(labels) + '</div>')


def _correlations():
    pairs = [("CLIP-H六对（均值）", .9600891535596247),
             ("CLIP-H主跑 ↔ SigLIP2", .7913757727208824),
             ("CLIP-H主跑 ↔ bigG384", .7091440666347133)]
    rows = []
    for label,value in pairs:
        width = 1000 * value
        rows.append(f'<div class="im-corr-row"><div class="im-corr-label"><span>{label}</span><strong>{value:.4f}</strong></div><svg class="im-corr-bar" viewBox="0 0 1000 30" preserveAspectRatio="none" aria-hidden="true"><rect width="1000" height="30" fill="#e5ebf0"/><rect width="{width:.4f}" height="30" fill="#789cab"/></svg></div>')
    return ('<div class="im-corr-panel"><h3 class="im-panel-head">分量秩相关系数 ρ</h3>'
            '<div class="im-corr-axis"><span>0</span><span>0.5</span><span>1.0</span></div>'
            + "".join(rows) + '</div>')


def _prior_oof():
    values = [.818892, .819480, .824702]
    names = ["桶统计", "GBM", "0.4/0.6融合"]
    xs = [155, 325, 500]
    yy = lambda v: 172 - (v-.810)/.020*142
    lines, labels, dots = [], [], []
    for value in [.810, .820, .830]:
        y=yy(value)
        lines.append(f'<line x1="90" x2="560" y1="{y:.3f}" y2="{y:.3f}" stroke="#d3dfe7" stroke-width="1.2"/>')
        labels.append(f'<span class="im-prior-tick" style="top:{y/2.2:.4f}%">{value:.3f}</span>')
    for x,v,name in zip(xs,values,names):
        dots.append(f'<circle cx="{x}" cy="{yy(v):.3f}" r="6.5" fill="#174d6b"/>')
        labels.append(f'<span class="im-prior-value" style="left:{x/6:.4f}%;top:{yy(v)/2.2-4:.4f}%">{v:.6f}</span>')
        labels.append(f'<span class="im-prior-x" style="left:{x/6:.4f}%">{name}</span>')
    return ('<div class="im-prior-panel"><h3 class="im-panel-head">尺寸先验：5折OOF AUC</h3>'
            '<div class="im-prior-plot" role="img" aria-label="桶统计OOF AUC0.818892、GBM0.819480、0.4桶与0.6GBM融合0.824702">'
            '<svg class="im-plot-svg" viewBox="0 0 600 220" preserveAspectRatio="none" aria-hidden="true">'
            + "".join(lines) + '<path d="M90 30V172H560" fill="none" stroke="#738e9f" stroke-width="1.7"/>'
            + "".join(dots) + '</svg>' + "".join(labels) + '</div></div>')


def _effects():
    body = f"""
    <div class="im-eval-grid">
      <div class="im-auc-panel"><h3 class="im-panel-head">融合方案的复赛AUC</h3>{_auc_curve()}<div class="im-auc-gain"><span>0.917360 → 0.918887</span><strong>＋0.001527</strong></div></div>
      <div class="im-diagnostics">{_correlations()}{_prior_oof()}</div>
    </div>
    """
    return _page("image-evidence", "图像效果：融合性能与分量互补", "S08", body)


def apply(pages, notes):
    """Keep the slide IDs, 70/80/60-second allocations and all other pages."""
    if len(pages) < 6 or len(notes) < 6:
        raise ValueError("image.apply requires six or more pages and notes")
    pages[3], pages[4], pages[5] = _training(), _fusion(), _effects()
    updates = {
        3: dict(title="图像训练：转导学生与监督分量",
                purpose="沿数据源、骨干、适配分量的方向解释8路结构",
                talk=[
                    "CLIP-H使用20,000目标图像软标签，加1,000官方样例硬标签，硬标签权重4。",
                    "主跑2轮，另外三个种子4轮；它们共享教师和骨干，种子改变可训参数初始化及增强。主跑种子为脚本默认20261003。",
                    "bigG的224视角有两个种子，384视角另有一个分量；Tiny-GenImage与官方样例提供监督，800训练、200验证。",
                    "SigLIP2使用GenImage++与官方样例监督；全方案冻结骨干，只更新LoRA与分类头。"
                ], transition="下面把8路输出按三槽公式汇合。"),
        4: dict(title="图像融合：三槽计算与逐图ECDF",
                purpose="说明8分量归3槽、prior侧路和固定参考查询",
                talk=[
                    "先对每个分量做ECDF。主槽包含4路CLIP-H和SigLIP2，五者等权；bigG224两路等权，bigG384独占一槽。",
                    "槽权重为3/4、1/7、3/28。三槽直接求base；base不会再经过中间ECDF。",
                    "尺寸先验由0.4桶ECDF与0.6GBM ECDF混合后再做先验ECDF。z等于0.8base加0.2prior，最后才查询最终ECDF。",
                    "右侧画的是ECDF函数形式，参考表有20,000项。输出为排序百分位，不能直接当已校准概率；尺寸先验依赖分布。"
                ], transition="效果页同时看融合AUC与分量的排序差异。"),
        5: dict(title="图像效果：融合性能与分量互补",
                purpose="对照方案AUC、最终成员的秩相关和先验OOF表现",
                talk=[
                    "四种方案的复赛AUC为0.908114、0.916175、0.917360和0.918887，最后两者相差0.001527。基底、权重和训练轮数的变化没有全部解耦，不能把全部差值归给一个因素。",
                    "右上是最终分量输出的Spearman相关，CLIP-H六对平均0.9601；主跑与SigLIP2为0.7914，与bigG384为0.7091。低相关表明排序差异，仍需足够的检测强度才可能帮助融合。",
                    "右下在1,000官方样例上做5折OOF，桶统计0.818892、GBM0.819480、两者融合0.824702。它衡量尺寸与文件属性信号，不是未见生成器的鲁棒性成绩。"
                ], transition="接下来进入文本检测模型与训练策略。")
    }
    for index, update in updates.items():
        notes[index].update(update)
    return pages, notes
