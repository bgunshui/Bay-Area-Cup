# 第三方组件与素材声明

本仓库原创内容以 [MIT 许可证](LICENSE) 开源。以下第三方组件/素材按各自许可证随仓库分发，
**MIT 不适用于它们**。

| 路径 | 组件 | 许可证 | 许可证原文 |
|---|---|---|---|
| `.agents/skills/guizang-ppt-skill/` | 歸藏 PPT Skill（固定版本，用于生成答辩演示） | AGPL-3.0 | `.agents/skills/guizang-ppt-skill/LICENSE`、`defense/licenses/guizang-ppt-skill-LICENSE` |
| `defense/assets/NotoSansSC-subset.woff2` | Noto Sans SC 子集字体 | SIL OFL 1.1 | `defense/licenses/NotoSansSC-OFL.txt` |
| `defense/assets/Inter-subset.woff2` | Inter 子集字体 | SIL OFL 1.1 | `defense/licenses/Inter-OFL.txt` |
| `defense/assets/motion-11.11.17.js` | Motion | MIT | `defense/licenses/motion-11.11.17-LICENSE.md` |
| `defense/assets/lucide-0.468.0.min.js` | Lucide | ISC | `defense/licenses/lucide-0.468.0-LICENSE` |

## 未随仓库分发的第三方依赖

- 模型基座与权重（Qwen2.5、CLIP ViT-H-14 / ViT-bigG-14、SigLIP2-so400m-384 等）按各自模型许可证使用，
  权重文件不入库，清单与校验值见 `image-track/WEIGHTS.md`、`text-track/WEIGHTS.md`。
- Docker 镜像基座（`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` 等）按各自许可证使用，镜像不随仓库分发。
- 竞赛数据（复赛文本/图片、官方样例原文、复赛标签）版权归主办方，不随仓库分发。

> 若您认为仓库中有内容侵犯了您的权利，请通过 Issue 联系维护者，我们会尽快处理。
