"""System overview: separate training signals and single-sample inference."""
import re

CSS = r'''
.fv-map{position:relative;flex:1;min-height:0;font-size:clamp(22px,1.35vw,26px);line-height:1.3}
.fv-map *{box-sizing:border-box}
.fv-wires{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:visible}
.fv-node{position:absolute;background:#fff;border:1px solid #c5d4df;padding:1.25vh .9vw;display:flex;flex-direction:column;justify-content:center;gap:.75vh}
.fv-node strong{font-size:clamp(24px,1.6vw,31px);line-height:1.18;font-weight:550;color:#174d6b}
.fv-node small{font-size:clamp(22px,1.25vw,24px);line-height:1.25;color:#567387}
.fv-node.fv-dark{background:#174d6b;border:0;color:#fff}.fv-dark strong{color:#fff}
.fv-model{background:#edf2f5;border:0;border-left:3px solid #174d6b}
.fv-label{position:absolute;font-size:clamp(22px,1.35vw,26px);font-weight:550;color:#567387;line-height:1.2}
.fv-stage{position:absolute;top:0;color:#174d6b;font-weight:550;font-size:clamp(24px,1.6vw,31px)}
.fv-train{background:transparent;border:1px dashed #8ba3b4;flex-direction:row;align-items:center;gap:1vw;padding:.8vh 1vw}
.fv-train strong{font-size:clamp(22px,1.35vw,26px)}
.fv-track-label{position:absolute;left:0;display:flex;gap:.65vw;align-items:center;color:#174d6b;font-weight:550}
.fv-icon{width:36px;height:36px;stroke-width:1.5;flex:none}
.fv-input{align-items:center;text-align:center;padding:.6vh .4vw;background:#edf2f5;border:0}
.fv-input .fv-icon{width:56px;height:56px;margin-bottom:.4vh}
.fv-subnodes{display:flex;gap:.4vw;align-items:center}.fv-subnodes span{background:#edf2f5;padding:.9vh .7vw;font-size:clamp(22px,1.25vw,24px);text-align:center;white-space:nowrap}
.fv-output{align-items:center;text-align:center;padding:.7vh .3vw}
.fv-legend{display:flex;align-items:center;gap:2vw;border-top:1px solid #ccd9e1;padding-top:1.5vh;margin-top:1.7vh;font-size:clamp(22px,1.25vw,24px);color:#567387}
.fv-key{display:flex;gap:.6vw;align-items:center}.fv-key i{width:3vw;border-top:2px solid #174d6b}.fv-key.train i{border-top:2px dashed #8ba3b4}
.fv-cover{display:flex;flex-direction:column;gap:4vh;border-left:1px solid #ccd9e1;padding-left:3vw}
.fv-cover h3{color:#174d6b;margin-bottom:1.6vh}
.fv-cover-lane{display:grid;grid-template-columns:1fr .15fr 1.45fr .15fr .7fr;gap:.8vw;align-items:center;font-size:clamp(22px,1.3vw,25px)}
.fv-cover-block{padding:2.5vh .75vw;background:#edf2f5;display:flex;flex-direction:column;gap:1vh;text-align:center;justify-content:center;min-height:18vh}
.fv-cover-block strong{font-size:clamp(30px,2.2vw,42px);font-weight:450;color:#174d6b}
.fv-cover-block.output{background:#174d6b;color:white}.fv-cover-arr{height:25px;width:100%}
'''


def box(x,y,w,h,head,detail='',cls=''):
    return f'<div class="fv-node {cls}" style="left:{x}%;top:{y}%;width:{w}%;height:{h}%"><strong>{head}</strong>{"<small>"+detail+"</small>" if detail else ""}</div>'


def system_page():
    # Geometric connectors only. All labels are selectable HTML text.
    paths='''<path d="M100 212H130M285 212H308V151H330M308 212H330M308 212V305H330M570 151H600V212H625M570 228H600M570 305H600V212M865 212H910"/>
      <path d="M100 519H130M285 519H310V476H330M310 519V581H330M570 476H600V519H625M570 581H600V519M865 519H910"/>
      <path d="M745 301V273"/>
      <path d="M465 112V119M465 427V434" stroke-dasharray="7 6" stroke="#8ba3b4"/>'''
    content='<div class="fv-map system-diagram" role="group" aria-label="双赛道全景架构，训练信号用虚线，推理数据流用实线，图像与文本独立评分">'
    for x,t in [(13,'预处理'),(33,'模型与适配'),(62.5,'分数融合'),(91,'输出')]:
        content+=f'<div class="fv-stage" style="left:{x}%">{t}</div>'
    content+='''<svg class="fv-wires" viewBox="0 0 1000 700" preserveAspectRatio="none" aria-hidden="true"><defs><marker id="fv-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="#174d6b"/></marker></defs><g fill="none" stroke="#174d6b" stroke-width="2" marker-end="url(#fv-arr)">'''+paths+'</g><path d="M0 355H1000" stroke="#ccd9e1" stroke-width="1"/></svg>'
    content+=box(13,7,44,9,'目标域蒸馏 ＋ 公开监督','LoRA适配','fv-train')
    content+=box(13,52,44,9,'目标域软标签 ＋ 官方标注','五折训练','fv-train')
    content+='''<div class="fv-node fv-input" style="left:0;top:23%;width:10%;height:16%"><i data-lucide="image" class="fv-icon"></i><span>单张图像</span></div>'''
    content+=box(13,23,15.5,16,'224 / 384','分量独立<br>归一化')
    content+=box(33,17,24,9,'CLIP-H × 4','', 'fv-model')
    content+=box(33,28,24,9,'bigG × 3','224 × 2 / 384 × 1','fv-model')
    content+=box(33,39,24,9,'SigLIP2 × 1','', 'fv-model')
    content+=box(62.5,23,24,16,'逐分量ECDF','三槽加权 → 最终ECDF')
    content+=box(62.5,43,24,6,'尺寸与文件属性先验','', 'fv-train')
    content+=box(91,23,9,16,'图像<br>分数','', 'fv-output fv-dark')
    content+='''<div class="fv-node fv-input" style="left:0;top:68%;width:10%;height:16%"><i data-lucide="file-text" class="fv-icon"></i><span>单段文本</span></div>'''
    content+=box(13,68,15.5,16,'分词','最多512<br>tokens')
    content+=box(33,62,24,12,'Qwen2.5 · 14B','5个适配器','fv-model')
    content+=box(33,77,24,12,'Qwen2.5 · 7B','两组共10个适配器','fv-model')
    content+=box(62.5,68,24,16,'分基座均值 → sigmoid','14B / 7B按9:1融合')
    content+=box(91,68,9,16,'文本<br>分数','', 'fv-output fv-dark')
    content+='</div><div class="fv-legend"><span class="fv-key"><i></i>单样本推理数据流</span><span class="fv-key train"><i></i>训练与辅助信号</span><span>两条检测链独立训练、独立输出</span></div>'
    return content


def cover():
    arr='<svg class="fv-cover-arr" viewBox="0 0 30 24" aria-hidden="true"><path d="M0 12H26M18 4L27 12L18 20" fill="none" stroke="#567387" stroke-width="2"/></svg>'
    result='<div class="fv-cover">'
    for label,input_,n,models in [('图像检测','图像','8','CLIP-H / bigG<br>SigLIP2'),('文本检测','文本','15','Qwen2.5<br>14B / 7B')]:
        result+=f'<div><h3>{label}</h3><div class="fv-cover-lane"><div class="fv-cover-block">{input_}<span>输入</span></div>{arr}<div class="fv-cover-block"><strong>{n}</strong><span>{"检测分量" if n=="8" else "适配器"}</span><span>{models}</span></div>{arr}<div class="fv-cover-block output">检测<br>分数</div></div></div>'
    return result+'</div>'


def apply(pages, notes):
    import core
    pages[2]=core.page('dual-pipelines','两条独立检测链，共同的方法主线','S17',system_page(),3,recipe='system-diagram')
    # Preserve title, team signature and the cover's existing two-column grid.
    # The cover has nested branch divs; use the known source interval.
    if 'r-cover-figure' in pages[0]:
        left=pages[0].index('<div class="r-cover-figure">')
        pages[0]=pages[0][:left]+cover()+'</div></div></section>'
    # Correct two extrapolated throughput statements; retain measured HTTP conditions.
    pages[11]=pages[11].replace('<br>批量提交：单图8分量约0.07秒','').replace('<br>批量提交：单条平均约0.09到0.14秒','')
    for i in [2,11]:
        notes[i]=dict(notes[i])
    notes[2].update(purpose='通过全景架构区分训练与推理，并沿数据流讲解两个独立输出',talk=[
        '实线连接单样本输入、预处理、模型分支、融合与独立输出；虚线代表训练和辅助信号。',
        '图像8个分量来自CLIP-H四路、bigG三路及SigLIP2一路，进入ECDF和三槽融合。',
        '在线文本两基座各先平均logit再sigmoid，然后9:1融合；整批排名路径在第9页单独说明。'])
    notes[11]['talk']=[x for x in notes[11]['talk'] if not ('0.07' in x or '0.09' in x or '0.14' in x)]
    notes[11]['talk'].append('图像批量8分量0.07秒是单骨干探针外推；文本0.09到0.14秒含训练后评估等阶段，不能作为最终完整系统纯推理耗时。')
    return pages,notes
