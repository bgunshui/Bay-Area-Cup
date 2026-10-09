from html import escape

CSS = r'''
:root{--r-ink:#11283d;--r-blue:#174d6b;--r-mid:#567387;--r-pale:#edf2f5;--r-line:#ccd9e1;--r-paper:#f8f9fb;--accent:#174d6b;--accent-rgb:23,77,107;--ink:#11283d;--paper:#f8f9fb;--text-primary:#11283d;--border-subtle:#ccd9e1;--sans:Inter,"Noto Sans SC",sans-serif;--sans-zh:"Noto Sans SC",Inter,sans-serif}
.slide.r-slide{background:var(--r-paper);color:var(--r-ink)}
.r-shell{position:absolute;inset:7vh 4.5vw 9vh;display:flex;flex-direction:column;min-height:0}
.r-title{font-size:clamp(40px,3vw,58px);font-weight:600;line-height:1.25;letter-spacing:-.025em;margin:0;padding-bottom:2.4vh;border-bottom:2px solid var(--r-ink);flex:none}
.r-body{flex:1;min-height:0;margin-top:3.5vh;display:flex;flex-direction:column}
.r-slide h3{font-size:clamp(26px,1.8vw,35px);line-height:1.25;font-weight:550;margin:0}
.r-slide p{font-size:clamp(22px,1.4vw,28px);line-height:1.55;margin:0}
.r-muted{color:var(--r-mid)}.r-num{font-family:Inter,"Noto Sans SC",sans-serif;font-variant-numeric:tabular-nums;letter-spacing:-.035em}
.r-caption{font-size:clamp(22px,1.25vw,25px);line-height:1.45;color:var(--r-mid)}
.r-eq{font-family:Inter,"Noto Sans SC",sans-serif;font-size:clamp(26px,2vw,38px);line-height:1.6;font-weight:400;letter-spacing:-.01em}
.r-eq i{font-family:Georgia,"Noto Sans SC",serif;font-style:italic}
.r-rule{border-top:1px solid var(--r-line)}
.r-page-index{position:absolute;right:4.5vw;bottom:8vh;color:var(--r-mid);font-family:Inter,sans-serif;font-size:22px;font-variant-numeric:tabular-nums}
.r-cover-layout{display:grid;grid-template-columns:1.22fr 1fr;gap:6vw;height:100%;align-items:center}
.r-cover-title{font-size:clamp(60px,5vw,96px);font-weight:550;line-height:1.17;letter-spacing:-.035em;margin:0}
.r-cover-subtitle{font-size:clamp(28px,2.1vw,40px);font-weight:400;line-height:1.4;margin-top:4vh}
.r-cover-team{font-size:clamp(24px,1.65vw,32px);margin-top:11vh}
.r-cover-figure{border-left:1px solid var(--r-line);padding-left:3vw;display:grid;grid-template-rows:1fr 1fr;gap:6vh;align-items:center}
.r-cover-branch{padding-top:2.6vh;border-top:3px solid var(--r-blue)}
.r-cover-branch h3{font-size:clamp(28px,2vw,38px);margin-bottom:2vh}
.r-cover-count{font-size:clamp(56px,5vw,96px);color:var(--r-blue);line-height:1;font-weight:450}
.r-cover-count span{font-size:clamp(22px,1.35vw,26px);color:var(--r-mid);letter-spacing:0;margin-left:1vw}
.r-cover-branch p{margin-top:2vh}
.r-problem-grid{display:grid;grid-template-columns:1.12fr 1fr;gap:5vw;flex:1;min-height:0}
.r-data-panel{display:flex;flex-direction:column;justify-content:center;gap:3vh}
.r-data-numbers{display:grid;grid-template-columns:1fr 1fr;gap:2.4vw;align-items:end}
.r-data-numbers strong{font-size:clamp(46px,4.2vw,80px);font-weight:450;display:block;line-height:1.1}
.r-data-numbers span{display:block;font-size:clamp(22px,1.5vw,29px);margin-top:1.5vh}
.r-data-bar{height:7vh;display:flex;gap:.25vw}.r-data-bar>span:first-child{width:4.7619048%;background:var(--r-ink)}.r-data-bar>span:last-child{flex:1;background:#a3bbcb}
.r-task-equation{border-top:1px solid var(--r-line);padding-top:3vh}
.r-problem-steps{display:grid;grid-template-rows:repeat(3,1fr);border-left:1px solid var(--r-line);padding-left:3vw}
.r-problem-step{display:grid;grid-template-columns:3vw 1fr;gap:1.2vw;align-content:center;border-bottom:1px solid var(--r-line)}
.r-problem-step:last-child{border-bottom:0}.r-problem-step>span{font-size:clamp(28px,2vw,38px);color:var(--r-mid)}
.r-problem-step p{margin-top:max(14px,1.4vh)}
.r-pipeline-grid{display:grid;grid-template-columns:1fr 1fr;gap:5vw;flex:1;min-height:0;margin-top:0}
.r-pipeline{position:relative;display:flex;flex-direction:column;gap:1.4vh;padding:2vh 0}
.r-pipeline+ .r-pipeline{border-left:1px solid var(--r-line);padding-left:4vw}
.r-pipeline h3{margin-bottom:1vh}.r-pipeline-node{border:1px solid var(--r-line);background:#fff;padding:2.1vh 1.5vw}
.r-pipeline-node strong{display:block;font-size:clamp(24px,1.65vw,32px);font-weight:550;line-height:1.25}
.r-pipeline-node p{margin-top:1vh}.r-pipeline-node.r-fill{border-color:var(--r-blue);background:var(--r-blue);color:#fff}
.r-arrow{text-align:center;font-size:clamp(24px,1.5vw,30px);line-height:1;color:var(--r-mid)}
.r-pipeline-end{font-size:clamp(30px,2.3vw,44px);font-weight:500;text-align:center;padding:1.5vh}
.r-results{display:grid;grid-template-columns:1fr 1fr;gap:5vw;flex:1;min-height:0}
.r-result{display:flex;flex-direction:column;justify-content:center;padding:3vh 0}.r-result+ .r-result{border-left:1px solid var(--r-line);padding-left:4vw}
.r-result-label{font-size:clamp(26px,1.85vw,36px);font-weight:550;display:flex;justify-content:space-between}
.r-result-metric{font-size:clamp(65px,6vw,115px);font-weight:450;color:var(--r-blue);line-height:1.15;margin-top:2vh}
.r-result-chart{position:relative;margin-top:3vh}.r-result-axis{display:flex;justify-content:space-between;font-size:22px;color:var(--r-mid)}
.r-result-track{height:3.8vh;background:#e1e9ee;margin-top:1.5vh}.r-result-track>div{height:100%;background:var(--r-blue)}
.r-result-method{margin-top:4vh!important;padding-top:3vh;border-top:1px solid var(--r-line)}
.r-result-caption{padding-top:2.8vh;margin-top:2.5vh;border-top:1px solid var(--r-line);font-size:clamp(22px,1.4vw,28px)}
.r-robust{display:grid;grid-template-columns:1.26fr 1fr;gap:4vw;flex:1;min-height:0}
.r-stress-flow{display:grid;grid-template-rows:repeat(3,1fr);gap:1.6vh}
.r-stress-row{display:grid;grid-template-columns:1fr auto 1.15fr;gap:1.1vw;align-items:center;border-bottom:1px solid var(--r-line);padding:1.5vh 0}
.r-stress-row:last-child{border-bottom:0}.r-stress-row p{margin-top:max(14px,1.1vh)}
.r-stress-method{background:var(--r-pale);border-left:3px solid var(--r-blue);padding:2.2vh 1.3vw}
.r-generalization{border-left:1px solid var(--r-line);padding-left:3vw;display:flex;flex-direction:column;gap:2.6vh}
.r-risk-axes{display:grid;grid-template-columns:1fr 1fr;gap:2vh 1.5vw}
.r-risk-axis{padding:2vh 0;border-top:1px solid var(--r-line);font-size:clamp(24px,1.65vw,32px)}
.r-limit{margin-top:auto;background:var(--r-ink);color:#fff;padding:2.5vh 1.5vw;font-size:clamp(22px,1.45vw,28px);line-height:1.6}
.r-deploy{display:grid;grid-template-columns:1fr 1fr;gap:5vw;flex:1;min-height:0}
.r-deploy-col{display:flex;flex-direction:column}.r-deploy-col+ .r-deploy-col{padding-left:4vw;border-left:1px solid var(--r-line)}
.r-time{font-size:clamp(64px,5vw,96px);font-weight:450;line-height:1.1;margin-top:2vh;color:var(--r-blue)}
.r-time span{font-size:clamp(24px,1.5vw,30px);letter-spacing:0;margin-left:.8vw}
.r-deploy-condition{margin-top:1.8vh!important}
.r-resident{margin-top:3vh;padding:2.3vh 1.4vw;border-top:2px solid var(--r-blue);background:var(--r-pale);font-size:clamp(24px,1.65vw,32px)}
.r-forward-grid{display:flex;gap:.35vw;margin-top:2vh;height:5vh}.r-forward-grid>span{flex:1;background:#174d6b}.r-forward-grid>span.aux{background:#8caabd}
.r-deploy-desc{margin-top:2vh!important}.r-deploy-bottom{margin-top:auto;border-top:1px solid var(--r-line);padding-top:2vh}
.r-application{display:flex;flex-direction:column;flex:1;min-height:0}
.r-app-flow{display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;gap:1vw;align-items:center;flex:1}
.r-app-node{min-height:26vh;display:flex;flex-direction:column;border-top:3px solid var(--r-blue);padding:2.7vh 1.3vw;background:#fff}
.r-app-node .r-app-symbol{font-size:clamp(46px,3.6vw,68px);line-height:1;color:var(--r-mid);font-weight:400;margin-bottom:3vh}
.r-app-node p{margin-top:2.5vh;padding-top:0}
.r-app-logic{display:grid;grid-template-columns:1fr 1fr;gap:4vw;align-items:center;border-top:1px solid var(--r-line);padding-top:3vh}
.r-app-logic p{border-left:3px solid var(--r-blue);padding-left:1.8vw}
.r-close{display:grid;grid-template-columns:1.05fr 1fr;gap:7vw;height:100%;align-items:center}
.r-close-statement{font-size:clamp(52px,4vw,78px);font-weight:500;line-height:1.32;letter-spacing:-.035em}
.r-close-statement span{display:block;color:var(--r-blue);margin-top:2vh}.r-close-claims{display:grid;grid-template-rows:repeat(3,1fr);gap:2.6vh}
.r-close-claim{display:grid;grid-template-columns:2.8vw 1fr;gap:1.5vw;padding:2.5vh 0;border-top:1px solid var(--r-line)}
.r-close-claim>span{font-size:clamp(24px,1.8vw,34px);color:var(--r-mid)}.r-close-claim p{margin-top:max(14px,1.4vh)}
.r-qa{position:absolute;left:4.5vw;bottom:8vh;font-size:clamp(24px,1.6vw,31px);color:var(--r-mid)}
#hint{opacity:0!important}#hint:hover,#hint:focus-within{opacity:1!important}
#nav{bottom:3.4vh;gap:6px;opacity:.4}#nav:hover{opacity:1}
#nav .dot{border-radius:0!important;background:#b5c6d1!important}#nav .dot.active{background:var(--r-blue)!important}
body.ppt-presenter-mode .r-page-index{display:none}
'''


def page(sid, title, layout, content, number, recipe='duo-mirror', cover=False):
    if cover:
        shell = content
    else:
        shell = f'<h2 class="r-title">{title}</h2><div class="r-body">{content}</div>'
    return (f'<section class="slide light r-slide" data-slide-id="{sid}" data-title="{escape(title,quote=True)}" '
            f'data-layout="{layout}" data-animate="{recipe}"><div class="r-shell">{shell}</div>'
            f'<div class="r-page-index">{number:02} / 14</div></section>')


def apply(pages, notes):
    pages[0] = page('cover','AI生成内容鉴别','S03','''
    <div class="r-cover-layout">
      <div><h1 class="r-cover-title">AI生成内容<br>鉴别</h1><div class="r-cover-subtitle">异构表征与目标域自训练</div><div class="r-cover-team">南枝大学参赛队</div></div>
      <div class="r-cover-figure">
        <div class="r-cover-branch"><h3>图像检测</h3><div class="r-cover-count r-num">8<span>LoRA分量</span></div><p>CLIP-H · bigG · SigLIP2<br>百分位融合与尺寸先验</p></div>
        <div class="r-cover-branch"><h3>文本检测</h3><div class="r-cover-count r-num">15<span>LoRA适配器</span></div><p>Qwen2.5-14B / 7B<br>软标签自训练与分支融合</p></div>
      </div>
    </div>''',1,cover=True)

    pages[1] = page('shared-challenges','少标签与分布变化下的检测任务','S13','''
    <div class="r-problem-grid">
      <div class="r-data-panel">
        <h3>每条赛道的数据条件</h3>
        <div class="r-data-numbers"><div><strong class="r-num">1,000</strong><span>官方标注样例</span></div><div><strong class="r-num">20,000</strong><span>无标签目标样本</span></div></div>
        <div class="r-data-bar" aria-label="标注样例与无标签目标样本数量之比为1比20"><span></span><span></span></div>
        <p class="r-muted">利用预训练特征与目标域软监督</p>
        <div class="r-task-equation"><div class="r-eq"><i>f</i><sub>θ</sub>(<i>x</i>) → <i>s</i>(<i>x</i>) ∈ [0, 1]</div><p>单张图片或单段文本 → 连续检测分数</p></div>
      </div>
      <div class="r-problem-steps">
        <div class="r-problem-step"><span class="r-num">01</span><div><h3>少标签学习</h3><p>预训练骨干与LoRA适配</p></div></div>
        <div class="r-problem-step"><span class="r-num">02</span><div><h3>目标域变化</h3><p>软标签自训练与互补监督</p></div></div>
        <div class="r-problem-step"><span class="r-num">03</span><div><h3>单样本推理</h3><p>固定参考分布与独立评分</p></div></div>
      </div>
    </div>''',2,recipe='three-forces')

    pages[2] = page('dual-pipelines','两条独立检测链，共同的方法主线','S08','''
    <div class="r-pipeline-grid duo-compare">
      <div class="r-pipeline"><h3>图像 · 视觉表征</h3>
        <div class="r-pipeline-node"><strong>图像预处理</strong><p>分量各自的输入分辨率与归一化</p></div><div class="r-arrow">↓</div>
        <div class="r-pipeline-node"><strong>3类骨干 · 8个LoRA分量</strong><p>转导学生 ＋ 独立监督分量</p></div><div class="r-arrow">↓</div>
        <div class="r-pipeline-node r-fill"><strong>分量ECDF → 三槽融合<br>尺寸先验 → 最终ECDF</strong></div>
        <div class="r-pipeline-end">图像检测分数</div>
      </div>
      <div class="r-pipeline"><h3>文本 · 语言表征</h3>
        <div class="r-pipeline-node"><strong>分词与长度处理</strong><p>服务输入上限512 tokens</p></div><div class="r-arrow">↓</div>
        <div class="r-pipeline-node"><strong>2个基座 · 3组五折适配器</strong><p>14B为主，7B两组为辅</p></div><div class="r-arrow">↓</div>
        <div class="r-pipeline-node r-fill"><strong>分组聚合 → 9:1融合</strong></div>
        <div class="r-pipeline-end">文本检测分数</div>
      </div>
    </div>''',3)

    pages[9] = page('results-and-reliability','双赛道检测结果','S20','''
    <div class="r-results">
      <div class="r-result"><div class="r-result-label"><span>图像赛道</span><span>AUC</span></div><div class="r-result-metric r-num">0.918887</div>
        <div class="r-result-chart"><div class="r-result-axis"><span>0</span><span>0.5</span><span>1.0</span></div><div class="r-result-track"><div style="width:91.8887%"></div></div></div>
        <p class="r-result-method">异构视觉骨干与多种子学生<br>三槽百分位融合 ＋ 尺寸先验</p>
      </div>
      <div class="r-result"><div class="r-result-label"><span>文本赛道</span><span>AUC</span></div><div class="r-result-metric r-num">0.981301</div>
        <div class="r-result-chart"><div class="r-result-axis"><span>0</span><span>0.5</span><span>1.0</span></div><div class="r-result-track"><div style="width:98.1301%"></div></div></div>
        <p class="r-result-method">目标域软标签自训练<br>14B主分支 ＋ 7B辅助集成</p>
      </div>
    </div>
    <div class="r-result-caption">评价对象：复赛目标集合的离线检测排序</div>''',10,recipe='stacked-ledger')

    pages[10] = page('robustness-coverage','多场景与对抗鲁棒性：机制和适用范围','S16','''
    <div class="r-robust">
      <div class="r-stress-flow">
        <div class="r-stress-row"><div><h3>图像后处理</h3><p>压缩 · 方向变化</p></div><span class="r-arrow">→</span><div class="r-stress-method"><h3>JPEG与翻转增强</h3><p>覆盖训练阶段的内容扰动</p></div></div>
        <div class="r-stress-row"><div><h3>文本表达变化</h3><p>润色 · 改写</p></div><span class="r-arrow">→</span><div class="r-stress-method"><h3>改写训练探索</h3><p>结合目标域软标签学习</p></div></div>
        <div class="r-stress-row"><div><h3>监督来源差异</h3><p>骨干 · 数据 · 种子</p></div><span class="r-arrow">→</span><div class="r-stress-method"><h3>强度与互补性共同选型</h3><p>增加分量不保证提升</p></div></div>
      </div>
      <div class="r-generalization"><h3>泛化评估的扩展方向</h3><div class="r-risk-axes"><div class="r-risk-axis">未见生成器</div><div class="r-risk-axis">扰动强度</div><div class="r-risk-axis">领域与语言</div><div class="r-risk-axis">输入长度</div></div>
        <p class="r-muted">尺寸先验依赖规格分布<br>转导训练依赖目标集合<br>长度截断可能遗漏尾部线索</p>
        <div class="r-limit">当前结果覆盖竞赛目标域；鲁棒性收益需由独立分组实验检验。</div>
      </div>
    </div>''',11,recipe='brief-grid')

    img_blocks=''.join('<span></span>' for _ in range(8))
    txt_blocks=''.join('<span'+(' class="aux"' if i>=5 else '')+'></span>' for i in range(15))
    pages[11] = page('efficiency-deployment','检测效率与资源配置','S21',f'''
    <div class="r-deploy">
      <div class="r-deploy-col"><h3>图像推理</h3><div class="r-time r-num">0.236<span>秒／张</span></div><p class="r-deploy-condition">GPU容器 · 200张顺序HTTP · 47.2秒</p>
        <div class="r-resident">3类骨干 · 4个共享模型实例</div><div class="r-forward-grid" aria-label="每张图片依次执行8个分量">{img_blocks}</div><p class="r-deploy-desc">8个分量串行前向 → ECDF融合</p>
        <div class="r-deploy-bottom"><p>骨干与LoRA：fp16 · 分类头：fp32<br>模型加载约15.1秒</p></div>
      </div>
      <div class="r-deploy-col"><h3>文本推理</h3><div class="r-time r-num">0.68<span>秒／条</span></div><p class="r-deploy-condition">宿主GPU · 5条短输入（含空串）<br>HTTP均值</p>
        <div class="r-resident">14B与7B基座常驻</div><div class="r-forward-grid" aria-label="每条文本5个14B加10个7B适配器串行前向">{txt_blocks}</div><p class="r-deploy-desc">15个adapter轮换 → 9:1融合</p>
        <div class="r-deploy-bottom"><p>CUDA路径：bf16 · 上限512 tokens<br>模型启动约70.1秒</p></div>
      </div>
    </div>''',12,recipe='tech-spec')

    pages[12] = page('innovation-application','从检测分数到内容审核辅助','S19','''
    <div class="r-application"><div class="r-app-flow four-cards">
      <div class="r-app-node"><div class="r-app-symbol">01</div><h3>内容输入</h3><p>图像与文本<br>独立入口</p></div><div class="r-arrow">→</div>
      <div class="r-app-node"><div class="r-app-symbol">02</div><h3>模型评分</h3><p>提取生成特征<br>连续检测分数</p></div><div class="r-arrow">→</div>
      <div class="r-app-node"><div class="r-app-symbol">03</div><h3>阈值与分流</h3><p>校准业务阈值<br>安排审核顺序</p></div><div class="r-arrow">→</div>
      <div class="r-app-node"><div class="r-app-symbol">04</div><h3>人工审核</h3><p>结合场景规则<br>形成最终判断</p></div>
    </div><div class="r-app-logic"><div class="r-eq"><i>s</i>(<i>x</i>) ≥ <i>τ</i> → 重点审核</div><p>应用设想：内容平台与素材筛查<br>业务数据测试 → 阈值校准 → 分布监控</p></div></div>''',13,recipe='four-cards')

    pages[13] = page('closing','核心结论','S10','''
    <div class="r-close"><div class="r-close-statement">异构表征<br>目标域学习<span>单样本检测</span></div>
      <div class="r-close-claims">
        <div class="r-close-claim"><span>01</span><div><h3>分别建模图像与文本</h3><p>视觉骨干与语言基座提供检测表征</p></div></div>
        <div class="r-close-claim"><span>02</span><div><h3>软监督与分数级融合</h3><p>利用目标域样本，组合强度与互补性</p></div></div>
        <div class="r-close-claim"><span>03</span><div><h3>面向实际审核场景</h3><p>资源配置、业务阈值与人工判断</p></div></div>
      </div>
    </div>''',14,recipe='split-statement',cover=True).replace('<div class="r-page-index">','<div class="r-qa">感谢各位评委，欢迎提问</div><div class="r-page-index">')

    talks = {
      0: ['图像与文本两条独立检测链。','异构表征与目标域自训练。','南枝大学参赛队。'],
      1: ['每条赛道1,000标注样例与20,000无标签目标样本。','少标签下借助预训练模型，LoRA适配检测任务。','目标域学习与单样本推理共同设计。'],
      2: ['图像3类骨干、8个LoRA分量，百分位融合。','文本14B与7B、三组五折适配器、9:1融合。','两种模态独立训练、独立输出。'],
      9: ['复赛离线图像AUC0.918887，文本AUC0.981301。','AUC衡量正负样本排序能力，不是准确率。','图像通过互补视觉分量，文本通过自训练与主辅融合。'],
      10:['图像采用JPEG与翻转增强，文本有改写训练探索。','分量选择同时考虑强度和互补性，增加模型不保证提高。','目标域、尺寸分布、长度截断限制泛化范围。','跨生成器及强度分组尚没有完整独立评测。'],
      11:['图像200张顺序HTTP共47.2秒，均值0.236秒/张。','文本5条短输入含空串，HTTP均值0.68秒/条。','共享基座、轮换适配器、串行请求保护状态。','两组计时条件不同；文本为宿主GPU测量，不能视为最终容器验证。'],
      12:['检测分数用于审核排序与风险分流。','业务阈值需按误报成本与业务样本校准。','内容平台、素材筛查为应用设想；通过试点与分布监控评估。'],
      13:['异构表征分别处理图像与文本。','软标签和分数融合利用目标域信息。','单样本推理结合阈值与人工审核。'],
    }
    for i,talk in talks.items():
        notes[i]['talk'] = talk
    for i in (0,1,2,9,10,11,12,13):
        start=pages[i].find('data-title="')+12
        notes[i]['title']=pages[i][start:pages[i].find('"',start)]
    return pages,notes
