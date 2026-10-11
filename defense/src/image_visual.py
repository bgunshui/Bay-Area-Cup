"""Native image-track architecture and ECDF data-flow figures.

This layer replaces only slides 4 and 5.  Text stays in HTML; SVG contains
geometry only.  The LoRA figure illustrates a linear layer within the frozen
backbone, rather than adding a second network after the backbone.
"""

from html import escape


CSS = r"""
.slide.iv-page{background:#f8f9fb;color:#11283d}
.iv-shell{position:absolute;inset:7vh 4.5vw 9vh;display:flex;flex-direction:column;min-height:0}
.iv-title{font-size:clamp(40px,2.916667vw,56px);font-weight:600;line-height:1.15;letter-spacing:-.025em;margin:0}
.iv-body{flex:1;min-height:0;margin-top:5vh;display:flex;flex-direction:column;gap:1.4vh;font-size:clamp(22px,1.354167vw,26px);line-height:1.25}
.iv-body strong{font-weight:550}.iv-body .iv-muted{color:#567387}
.iv-section-head{display:flex;align-items:center;justify-content:space-between;gap:1.5vw;min-height:3.4vh;flex:none}
.iv-section-head h3{font-size:clamp(24px,1.666667vw,32px);font-weight:550;line-height:1.25;margin:0}
.iv-legend{display:flex;gap:1.4vw;align-items:center;font-size:clamp(22px,1.25vw,24px);color:#567387;white-space:nowrap}
.iv-legend span{display:flex;gap:.5vw;align-items:center}.iv-legend i{width:14px;height:14px;display:inline-block;background:#d5dfe5;border:1px solid #7f98a8;font-style:normal}
.iv-legend span:last-child i{background:#174d6b;border-color:#174d6b}
.iv-training{position:relative;height:31vh;min-height:218px;flex:none}
.iv-lines{position:absolute;inset:0;width:100%;height:100%;overflow:visible;pointer-events:none}
.iv-node{position:absolute;box-sizing:border-box;display:flex;align-items:center;justify-content:center;flex-direction:column;text-align:center;padding:.7vh .65vw;gap:.5vh;border:1.5px solid #bcced9;background:#fff;min-width:0}
.iv-node strong{font-size:clamp(24px,1.5625vw,30px);line-height:1.2;white-space:nowrap}
.iv-node span{font-size:clamp(22px,1.25vw,24px);line-height:1.2;white-space:nowrap}
.iv-soft{left:0;top:11%;width:24%;height:34%;border-left:4px solid #174d6b}
.iv-hard{left:0;top:64%;width:24%;height:34%;border-left:4px solid #7d97a8}
.iv-backbone-frame{position:absolute;left:29%;top:0;width:41%;height:100%;border:1px dashed #aabdc9;background:#f0f4f7;box-sizing:border-box}
.iv-backbone-label{position:absolute;left:29%;top:0;width:41%;height:13%;display:flex;align-items:center;justify-content:center;font-size:clamp(24px,1.5625vw,30px);font-weight:550;color:#174d6b;white-space:nowrap}
.iv-frozen{left:37%;top:19%;width:22%;height:26%;background:#e1e8ed;border-color:#9eb3c0}
.iv-lora{left:37%;top:65%;width:22%;height:26%;background:#174d6b;border-color:#174d6b;color:#fff}
.iv-lora span{color:#edf4f8}
.iv-sum{position:absolute;left:64%;top:56%;width:clamp(30px,2.5vw,48px);height:clamp(30px,2.5vw,48px);transform:translate(-50%,-50%);border:2px solid #174d6b;background:#f8f9fb;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:clamp(26px,1.875vw,36px);line-height:1;color:#174d6b;box-sizing:border-box}
.iv-head{left:74%;top:38%;width:14%;height:36%;border:2px solid #174d6b;background:#fff}
.iv-loss{left:92%;top:38%;width:8%;height:36%;border-color:#174d6b;background:#174d6b;color:#fff;padding:.7vh .2vw}
.iv-loss strong{font-size:clamp(24px,1.5625vw,30px)}
.iv-supervised-head{padding-top:.5vh;border-top:1px solid #ccd9e1;margin-top:.5vh}
.iv-supervised-head .iv-muted{font-size:clamp(22px,1.25vw,24px);white-space:nowrap}
.iv-supervised{display:grid;grid-template-columns:repeat(3,1fr);gap:1.5vw;flex:1;min-height:19vh;max-height:25vh}
.iv-super-branch{position:relative;border-top:3px solid #174d6b;background:#fff;padding:1vh 1.1vw;display:flex;flex-direction:column;justify-content:space-between;gap:.4vh;box-sizing:border-box;min-width:0}
.iv-super-title{display:flex;align-items:center;justify-content:space-between;gap:.8vw}
.iv-super-title strong{font-size:clamp(24px,1.5625vw,30px);font-weight:550;line-height:1.2;white-space:nowrap}
.iv-count{font-size:clamp(26px,2.1875vw,42px);font-weight:500;color:#174d6b;line-height:1.1;white-space:nowrap}
.iv-data-source{font-size:clamp(22px,1.25vw,24px);line-height:1.1;color:#567387;white-space:nowrap}
.iv-down{height:1.9vh;min-height:14px;width:100%;display:block;flex:none}
.iv-super-model{border:1px solid #b4c8d4;background:#edf3f6;padding:.6vh .4vw;text-align:center;font-size:clamp(22px,1.25vw,24px);line-height:1.2;white-space:nowrap}
.iv-super-spec{font-size:clamp(22px,1.25vw,24px);line-height:1.1;color:#567387;text-align:center;white-space:nowrap}
.iv-techline{display:flex;justify-content:space-between;align-items:center;gap:1.5vw;padding-top:1.3vh;border-top:1px solid #ccd9e1;min-height:3.2vh;flex:none;font-size:clamp(22px,1.25vw,24px);line-height:1.25;color:#567387}
.iv-techline .iv-eq{color:#174d6b;white-space:nowrap}
.iv-fusion-layout{position:relative;flex:1;min-height:0;display:grid;grid-template-columns:68fr 3fr 29fr}
.iv-flow-main{display:flex;flex-direction:column;gap:1.1vh;min-height:0}
.iv-stage-head{position:relative;height:3.7vh;min-height:28px;flex:none;color:#174d6b;font-size:clamp(22px,1.354167vw,26px);font-weight:550;white-space:nowrap}
.iv-stage-head span{position:absolute;top:0;line-height:1.25}
.iv-stage-head span:nth-child(1){left:0}.iv-stage-head span:nth-child(2){left:29%}.iv-stage-head span:nth-child(3){left:55%}
.iv-eight{position:relative;height:39vh;min-height:292px;flex:1}
.iv-component,.iv-ecdf{position:absolute;transform:translateY(-50%);height:9.6%;box-sizing:border-box;display:flex;align-items:center;justify-content:center;border:1px solid #bfced8;background:#fff;font-size:clamp(22px,1.25vw,24px);line-height:1;white-space:nowrap}
.iv-component{left:0;width:24%;font-weight:500}.iv-ecdf{left:29%;width:13%;color:#174d6b;background:#edf3f6;border-color:#aac2d1}
.iv-component.iv-comp-sig{border-color:#7898ad;background:#edf3f6;color:#174d6b}
.iv-slot{position:absolute;left:55%;width:40%;box-sizing:border-box;background:#fff;border:1.5px solid #9ab5c6;border-left:4px solid #174d6b;padding:.8vh .9vw;display:flex;flex-direction:column;justify-content:center;gap:1vh;min-width:0}
.iv-slot-head{display:flex;justify-content:space-between;gap:.4vw;align-items:center;font-size:clamp(24px,1.5625vw,30px);line-height:1.15;font-weight:550;white-space:nowrap}
.iv-weight{font-size:clamp(28px,2.1875vw,42px);font-weight:500;line-height:1;color:#174d6b;white-space:nowrap}
.iv-slot span{font-size:clamp(22px,1.25vw,24px);line-height:1.25;color:#567387;white-space:nowrap}
.iv-slot-main{top:2.5%;height:57.5%}.iv-slot-pair{top:64%;height:22%}.iv-slot-one{top:87.5%;height:12.5%;padding:.25vh .9vw}
.iv-slot-pair .iv-slot-head,.iv-slot-one .iv-slot-head{font-size:clamp(22px,1.354167vw,26px)}
.iv-slot-pair .iv-weight,.iv-slot-one .iv-weight{font-size:clamp(24px,1.5625vw,30px)}
.iv-slot-pair{gap:.2vh;padding:.35vh .9vw}.iv-slot-one{justify-content:center}
.iv-member-graphic{display:flex;gap:.4vw;align-items:center;width:100%;height:4vh;min-height:24px;margin:.3vh 0}
.iv-member-graphic i{display:block;flex:1;height:100%;background:#c7d7e1;border:1px solid #a3bccb;box-sizing:border-box}.iv-member-graphic i:last-child{background:#174d6b;border-color:#174d6b}
.iv-lower{position:relative;flex:none;height:18vh}
.iv-lower::before{content:"";position:absolute;left:76.5%;top:-1.1vh;height:1.1vh;border-left:2px solid #174d6b}
.iv-base,.iv-prior,.iv-z{position:absolute;box-sizing:border-box;display:flex;align-items:center;justify-content:center;flex-direction:column;gap:.5vh;padding:.7vh .7vw;text-align:center;border:1px solid #a8c0cf;white-space:nowrap}
.iv-base{left:55%;top:0;width:43%;height:5.7vh;background:#edf3f6;font-size:clamp(24px,1.5625vw,30px)}
.iv-prior{left:0;top:0;width:44%;height:13.3vh;background:#fff;border-left:3px solid #7c9caf;gap:.7vh}
.iv-prior strong{font-size:clamp(24px,1.458333vw,28px);font-weight:550;line-height:1.2}
.iv-prior span{font-size:clamp(22px,1.25vw,24px);line-height:1.2;color:#567387}
.iv-z{left:55%;bottom:0;width:43%;height:7.5vh;background:#174d6b;color:#fff;border-color:#174d6b;font-size:clamp(24px,1.458333vw,28px)}
.iv-z strong{font-weight:550;font-size:clamp(22px,1.25vw,24px);line-height:1}
.iv-ref{grid-column:3;border-left:1px solid #ccd9e1;padding-left:1.5vw;display:flex;flex-direction:column;gap:1vh;min-width:0;min-height:0}
.iv-ref h3{font-size:clamp(24px,1.666667vw,32px);line-height:1.25;font-weight:550;margin:0;white-space:nowrap}
.iv-ref-context{font-size:clamp(22px,1.25vw,24px);color:#567387;line-height:1.25;white-space:nowrap}
.iv-ecdf-plot{position:relative;flex:1;min-height:22vh;margin-top:.8vh;margin-bottom:.8vh}
.iv-ecdf-plot svg{position:absolute;inset:0;width:100%;height:100%;overflow:visible}
.iv-plot-y{position:absolute;left:0;top:39%;transform:rotate(-90deg);transform-origin:0 0;white-space:nowrap;font-size:clamp(22px,1.25vw,24px);color:#567387}
.iv-plot-x{position:absolute;left:19%;right:0;bottom:0;text-align:center;white-space:nowrap;font-size:clamp(22px,1.25vw,24px);color:#567387}
.iv-plot-q{position:absolute;left:67%;top:81%;font-size:clamp(22px,1.25vw,24px);color:#174d6b;white-space:nowrap}
.iv-plot-f{position:absolute;left:19%;top:18%;font-size:clamp(22px,1.25vw,24px);color:#174d6b;white-space:nowrap;background:#f8f9fb;padding:0 .25vw}
.iv-query{border-top:1px solid #ccd9e1;padding-top:1.1vh;margin-bottom:auto;font-size:clamp(22px,1.25vw,24px);color:#567387;line-height:1.3}
.iv-output-chain{height:7.5vh;min-height:54px;display:grid;grid-template-columns:1.15fr 20px 1fr;gap:.4vw;flex:none}
.iv-output-node{display:flex;align-items:center;justify-content:center;flex-direction:column;border:1.5px solid #174d6b;background:#fff;padding:.5vh .15vw;box-sizing:border-box;gap:.4vh;text-align:center;white-space:nowrap;font-size:clamp(22px,1.25vw,24px);line-height:1.1}
.iv-output-node:last-child{background:#174d6b;color:#fff}
.iv-output-node strong{font-weight:550;font-size:clamp(22px,1.25vw,24px)}
.iv-right{width:100%;height:100%;min-width:0;display:block}
.iv-cross-arrow{position:absolute;left:66.65%;width:5.05%;height:3vh;bottom:2.25vh;pointer-events:none}
.iv-footnote{font-size:clamp(22px,1.25vw,24px);line-height:1.3;color:#567387;margin:0;padding-top:1.1vh;border-top:1px solid #ccd9e1;flex:none}
"""


def _page(sid, title, layout, body):
    return (
        f'<section class="slide light iv-page" data-slide-id="{sid}" '
        f'data-title="{escape(title, quote=True)}" data-layout="{layout}" '
        'data-animate="academic-image">'
        f'<div class="iv-shell"><h2 class="iv-title">{title}</h2>'
        f'<div class="iv-body">{body}</div></div></section>'
    )


def _arrow_svg(cls="iv-down", direction="down"):
    shape = "M12 0V20M6 14L12 21L18 14" if direction == "down" else "M0 12H20M14 6L21 12L14 18"
    return (
        f'<svg class="{cls}" viewBox="0 0 24 24" aria-hidden="true">'
        f'<path d="{shape}" fill="none" stroke="#174d6b" stroke-width="2"/></svg>'
    )


def _training():
    branches = [
        ("CLIP bigG · 224", "× 2", "Tiny-GenImage · 28.8k", "r=16 · α=32"),
        ("CLIP bigG · 384", "× 1", "Tiny-GenImage · 28.8k", "r=32 · α=64"),
        ("SigLIP2 · 384", "× 1", "GenImage++ · 78k", "r=16 · α=32"),
    ]
    cards = []
    for model, count, data, spec in branches:
        cards.append(
            '<div class="iv-super-branch">'
            f'<div class="iv-super-title"><strong>{model}</strong><span class="iv-count">{count}</span></div>'
            f'<div class="iv-data-source">{data}</div>'
            + _arrow_svg()
            + '<div class="iv-super-model">冻结骨干＋LoRA＋分类头</div>'
            f'<div class="iv-super-spec">{spec}</div></div>'
        )
    body = r"""
    <div class="iv-section-head"><h3>目标域转导训练 · 4个CLIP-H学生</h3>
      <div class="iv-legend"><span><i></i>冻结参数</span><span><i></i>可训练参数</span></div>
    </div>
    <div class="iv-training" role="img" aria-label="目标图像软标签与官方硬标签混合训练CLIP-H；骨干内冻结线性层与低秩旁路并行，更新LoRA与分类头，使用加权BCE">
      <div class="iv-backbone-frame"></div>
      <svg class="iv-lines" viewBox="0 0 1800 320" preserveAspectRatio="none" aria-hidden="true">
        <defs><marker id="iv-train-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#174d6b"/></marker></defs>
        <g fill="none" stroke="#7d97a8" stroke-width="2.1" vector-effect="non-scaling-stroke">
          <path d="M432 89.6H486V179.2H576"/>
          <path d="M432 259.2H486V179.2"/>
          <path d="M576 179.2H612V102.4H666" marker-end="url(#iv-train-arrow)"/>
          <path d="M1062 102.4H1152V164"/>
        </g>
        <g fill="none" stroke="#174d6b" stroke-width="2.4" vector-effect="non-scaling-stroke">
          <path d="M612 179.2V249.6H666" marker-end="url(#iv-train-arrow)"/>
          <path d="M1062 249.6H1152V195"/>
          <path d="M1180 179.2H1332" marker-end="url(#iv-train-arrow)"/>
          <path d="M1584 179.2H1656" marker-end="url(#iv-train-arrow)"/>
        </g>
      </svg>
      <div class="iv-node iv-soft"><strong>目标图像 20,000</strong><span>教师软标签</span></div>
      <div class="iv-node iv-hard"><strong>官方样例 1,000</strong><span>硬标签 · 权重4</span></div>
      <div class="iv-backbone-label">CLIP-H · 224</div>
      <div class="iv-node iv-frozen"><strong>冻结权重 W</strong><span>骨干内线性层</span></div>
      <div class="iv-node iv-lora"><strong>低秩旁路 LoRA</strong><span>A → B · r=32</span></div>
      <div class="iv-sum">＋</div>
      <div class="iv-node iv-head"><strong>分类头</strong><span>可训练</span></div>
      <div class="iv-node iv-loss"><span>加权</span><strong>BCE</strong></div>
    </div>
    <div class="iv-section-head iv-supervised-head"><h3>独立监督训练 · 4个互补分量</h3><span class="iv-muted">官方样例：800训练 / 200验证</span></div>
    <div class="iv-supervised">""" + "".join(cards) + r"""</div>
    <div class="iv-techline"><span>JPEG增强 · 监督支路水平翻转</span><span class="iv-eq">W′ = W ＋ (α/r)BA</span><span>8个分量 = 4 ＋ 2 ＋ 1 ＋ 1</span></div>
    """
    return _page("image-training", "图像训练：转导学生与监督分量", "S05", body)


def _eight_components():
    members = [
        ("CLIP-H · 1", False), ("CLIP-H · 2", False),
        ("CLIP-H · 3", False), ("CLIP-H · 4", False),
        ("SigLIP2", True), ("bigG224 · 1", False),
        ("bigG224 · 2", False), ("bigG384", False),
    ]
    rows, paths = [], []
    for i, (label, siglip) in enumerate(members):
        y = (i + .5) * 12.5
        sy = y * 4.8
        cls = " iv-comp-sig" if siglip else ""
        rows.append(f'<div class="iv-component{cls}" style="top:{y:.2f}%">{label}</div>')
        rows.append(f'<div class="iv-ecdf" style="top:{y:.2f}%">ECDF</div>')
        paths.append(f'<path d="M240 {sy:.2f}H290" marker-end="url(#iv-eight-arrow)"/>')
        paths.append(f'<path d="M420 {sy:.2f}H490"/>')
    body = r"""
    <div class="iv-eight" role="img" aria-label="八个模型各自查询ECDF；四个CLIP-H与SigLIP2五路等权进入主槽，两个bigG224等权进入第二槽，bigG384进入第三槽，三槽以3/4、1/7、3/28加权">
      <svg class="iv-lines" viewBox="0 0 1000 480" preserveAspectRatio="none" aria-hidden="true">
        <defs><marker id="iv-eight-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#174d6b"/></marker></defs>
        <g fill="none" stroke="#7898ad" stroke-width="1.7" vector-effect="non-scaling-stroke">"""
    body += "".join(paths)
    body += r"""
          <path d="M490 30V270M490 150H550" marker-end="url(#iv-eight-arrow)"/>
          <path d="M490 330V390M490 360H550" marker-end="url(#iv-eight-arrow)"/>
          <path d="M490 450H550" marker-end="url(#iv-eight-arrow)"/>
        </g>
        <g fill="none" stroke="#174d6b" stroke-width="2.1" vector-effect="non-scaling-stroke">
          <path d="M950 150H992V480H765"/>
          <path d="M950 360H992"/>
          <path d="M950 450H992"/>
        </g>
      </svg>""" + "".join(rows) + r"""
      <div class="iv-slot iv-slot-main"><div class="iv-slot-head">主槽 <b class="iv-weight">3/4</b></div><span>4个CLIP-H ＋ SigLIP2</span><div class="iv-member-graphic" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></div><span>五路百分位等权平均</span></div>
      <div class="iv-slot iv-slot-pair"><div class="iv-slot-head">bigG224 <b class="iv-weight">1/7</b></div><span>两路百分位等权平均</span></div>
      <div class="iv-slot iv-slot-one"><div class="iv-slot-head">bigG384 <b class="iv-weight">3/28</b></div></div>
    </div>"""
    return body


def _reference_panel():
    return r"""
    <div class="iv-ref">
      <h3>固定参考分布查询</h3>
      <div class="iv-ref-context">N = 20,000 · 函数示意</div>
      <div class="iv-ecdf-plot" role="img" aria-label="ECDF阶梯函数示意：单个输入x按固定参考分布查找百分位F(x)，并非实验曲线">
        <svg viewBox="0 0 400 350" preserveAspectRatio="none" aria-hidden="true">
          <path d="M68 28V292H390" fill="none" stroke="#7898ad" stroke-width="1.7"/>
          <path d="M68 277H108V242H149V206H191V162H232V121H273V76H317V45H388" fill="none" stroke="#174d6b" stroke-width="3.2"/>
          <path d="M273 76V292M68 76H273" fill="none" stroke="#7898ad" stroke-width="1.5" stroke-dasharray="6 5"/>
          <circle cx="273" cy="76" r="5.5" fill="#174d6b"/>
        </svg>
        <div class="iv-plot-y">百分位</div><div class="iv-plot-x">原始分／融合分</div>
        <div class="iv-plot-q">x</div><div class="iv-plot-f">F(x)</div>
      </div>
      <div class="iv-query">单图分数 x → 查表 → 百分位<br>参考表固定，无需实时批排名</div>
      <div class="iv-output-chain">
        <div class="iv-output-node"><strong>最终ECDF</strong><span>查询 F(z)</span></div>""" + _arrow_svg("iv-right", "right") + r"""
        <div class="iv-output-node"><strong>单图分数</strong><span>[0, 1]</span></div>
      </div>
    </div>
    """


def _fusion():
    lower = r"""
    <div class="iv-lower">
      <svg class="iv-lines" viewBox="0 0 1000 200" preserveAspectRatio="none" aria-hidden="true">
        <defs><marker id="iv-lower-arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#174d6b"/></marker></defs>
        <g fill="none" stroke="#174d6b" stroke-width="2.1" vector-effect="non-scaling-stroke">
          <path d="M765 0V5" marker-end="url(#iv-lower-arrow)"/>
          <path d="M765 65V112" marker-end="url(#iv-lower-arrow)"/>
          <path d="M440 76H490V157H550" marker-end="url(#iv-lower-arrow)"/>
        </g>
      </svg>
      <div class="iv-base"><strong>base · 三槽加权和</strong></div>
      <div class="iv-prior"><strong>尺寸先验 prior</strong><span>0.4 桶 ＋ 0.6 GBM</span><span>各查ECDF → 混合 → ECDF</span></div>
      <div class="iv-z"><strong>融合分 z</strong><span>0.8 base ＋ 0.2 prior</span></div>
    </div>
    """
    body = (
        '<div class="iv-fusion-layout"><div class="iv-flow-main">'
        '<div class="iv-stage-head"><span>8个分量输出</span><span>逐路转换</span><span>3槽聚合与权重</span></div>'
        + _eight_components() + lower + '</div>' + _reference_panel()
        + _arrow_svg("iv-cross-arrow", "right")
        + '</div><p class="iv-footnote">输出为排序百分位；尺寸先验依赖尺寸与文件属性分布。</p>'
    )
    return _page("image-ecdf", "图像融合：三槽计算与逐图ECDF", "S17", body)


def apply(pages, notes):
    """Replace two diagrams without changing the 14-slide / 800-second plan."""
    if len(pages) < 5 or len(notes) < 5:
        raise ValueError("image_visual.apply requires five or more pages and notes")
    pages[3], pages[4] = _training(), _fusion()
    notes[3].update(
        purpose="先沿上方转导训练结构讲低秩旁路，再对照下方三个监督来源",
        talk=[
            "上方左侧是20,000张目标图像的教师软标签，与1,000张官方样例的硬标签混合；硬标签权重4。",
            "中间放大的是CLIP-H骨干内线性层的LoRA适配示意。冻结W，与低秩A、B旁路的更新相加，只更新LoRA与分类头，使用加权BCE训练。",
            "CLIP-H主跑种子20261003训练2轮，三个追加种子20261101/02/03各4轮。不能把主跑与追加分量说成只改种子、训练预算完全相同。",
            "下方三个监督支路：bigG224两个分量和bigG384一个分量使用Tiny-GenImage 28.8k；SigLIP2使用GenImage++78k，各叠加官方800训练、200验证。",
            "所有分量冻结骨干并更新LoRA及分类头；图中并行路径是线性层适配示意，不表示在骨干后串接额外LoRA网络。"
        ],
        transition="训练得到八个分量后，下一页沿每条箭头看单张图怎样得到最终分数。",
    )
    notes[4].update(
        purpose="沿八路ECDF、三个槽、尺寸先验与最终查询解释单图算法",
        talk=[
            "从左侧八路输出开始，每个分量各自查固定参考ECDF，将原始分转成同一百分位尺度。",
            "五条上路进入主槽：四个CLIP-H与一个SigLIP2等权。两条bigG224等权进入第二槽，bigG384单独占第三槽；三槽权重分别3/4、1/7、3/28。",
            "沿槽右侧的汇合线下到base，直接得到加权和；这里不插入额外base ECDF。左下尺寸先验由0.4桶ECDF与0.6GBM ECDF混合，再做先验ECDF。",
            "两路在下方融合为z=0.8base+0.2prior，再向右查询最终ECDF，得到单图分数。",
            "右侧阶梯图仅示意查表函数，不是ROC或新增实验。固定参考表各有20,000项，按并列平均秩处理，F(x)=(lo+hi+1)/(2(N+1))；输出为排序百分位，尺寸先验依赖分布。"
        ],
        transition="接着用融合AUC与分量间排序差异，分析这些分支的实际作用。",
    )
    return pages, notes
