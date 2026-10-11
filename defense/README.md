# 双赛道答辩演示

面向评委的14页学术汇报，计划800秒（13分20秒）。内容涵盖图像与文本检测的架构、训练、数据处理、实验效果、鲁棒性范围、资源配置和应用路径。

在本机双击 [湾区杯决赛-南枝大学.html](湾区杯决赛-南枝大学.html)，使用Edge或Chrome播放。字体、图标与运行资源已内联，断网可用。快捷操作及双屏设置见 [演示说明](演示说明.md)。

## 材料入口

| 文件 | 用途 |
|---|---|
| [逐字讲稿](逐字讲稿.md) | 逐页口播、时长与转场 |
| [评委问答](评委问答.md) | 首答、追问补充及后台依据 |
| [证据索引](证据索引.md) | 关键数值的固定版本来源 |
| [图表说明](图表说明.md)、[图表目录](图表/) | 图示含义、数据和PNG／SVG导出 |
| [预览总览](预览总览.png) | 14页缩略图 |
| [总体架构预览](图表/总体架构预览.png) | 图像与文本两条独立检测链的完整数据流 |
| [本轮可视化检查](可视化检查_20261011.md) | 2026-10-11的两种画布、核心图与演讲者复测范围 |
| [验收记录](验收记录.md) | 原始浏览器与技能检查结果 |
| [制作与版本记录](制作与版本记录.json) | 技术基线、资源及入库说明 |

技术来源固定在 `c11fa520f2f73d705e83cb3f40a6a214f8cc7f31`，便于查询每项数字的依据。成绩、排序一致性、鲁棒性和应用设想的范围见证据索引；没有因入库重新训练或重测模型。

## 在仓库维护

`src/core.py`管理开场、整体结果、效率与应用页；`src/image.py`和`src/text.py`保留基础内容。`src/framework_visual.py`绘制封面及全景架构，`src/image_visual.py`绘制LoRA训练及八分量融合，`src/text_visual.py`绘制自训练、集成与推理计算图。`src/page-plan.json`保留页面ID与演讲备注。生成器只需Python标准库，模板使用仓库内固定版本的项目skill。

在仓库根目录运行：

```bash
git pull --ff-only
# 修改 defense/src/ 及对应讲稿、问答、图表说明
python defense/scripts/build.py --final
python defense/scripts/update_checksums.py
python defense/scripts/verify_materials.py
git add defense
git commit -m "docs(defense): update academic presentation"
git push origin main
```

校验脚本检查14页、800秒、备注对应、讲稿、问答连续编号、字体字符覆盖和文件SHA。它分别判断当前HTML是否对应原浏览器验收记录及本轮可视化记录；改动后应重新检查页面与演讲者功能，不能沿用旧结果声称新版本已实测。

`assets/`保留当前文本所用的字体子集和固定版运行资源。新增未覆盖的汉字或数学符号时，先运行不带`--final`的生成器，再用`subset_fonts.py --source-dir <完整字体目录>`扩展子集，最后生成正式HTML；该字体脚本需要fontTools与Brotli，完整字体须为许可中的Inter和Noto Sans SC。校验器会列出缺失字符。制作缓存位于被Git忽略的 `.build/`。后续版本使用Git历史，不提交压缩包。

## 模板与资源来源

项目skill位于 [`.agents/skills/guizang-ppt-skill`](../.agents/skills/guizang-ppt-skill/)，固定上游版本 `c91369c449d34755d320a8b81d0734000d99d1ab`，保留模板、参考、脚本及许可证。[归藏项目](https://github.com/op7418/guizang-ppt-skill)采用AGPL-3.0，原许可保留在skill目录及 [licenses/](licenses/)；Inter、Noto Sans SC、Motion与Lucide各自的许可也保留在该目录。
