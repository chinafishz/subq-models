# SSA (Subquadratic Sparse Attention) — 技术要点

> 来源：SubQ-1.1-Small Technical Report (Subquadratic AI, 2026-06)
> 中文翻译项目：/Users/chinafishz/MyProduct/subq-models/

## 核心概念

**SSA** = 次平方稀疏注意力，一种内容相关的稀疏注意力机制，计算与记忆体均 O(n) 线性缩放。

## 关键数据

| 指标 | 数值 |
|------|------|
| 1M tokens FLOPs 降低 | **64.5×** (vs 密集注意力) |
| 1M tokens wall-clock 加速 | **56×** (vs FlashAttention-2, H100) |
| RULER (128K, 13任务) | **99.12%** |
| NIAH 1M/2M | **100%** |
| NIAH 6M/12M (训练外) | **98%** |
| 12M tokens 注意力稀疏度 | **0.13%** (只关注 ~1000 分之一的 token 对) |
| GPQA Diamond pass@1 | **85.4%** |
| LiveCodeBench v6 pass@4 | **89.7%** |
| AutomationBench Finance | **13%** (接近 Opus 4.8 16%) |

## 架构定位

SSA 同时满足四个需求（现有方法均未同时达成）：
1. **内容相关检索** — 由 token 内容决定路由，非固定位置模式
2. **端到端次平方缩放** — 选择/索引/注意力全链路线性
3. **全上下文训练 + 自回归生成** — 保留标准 Transformer 范式
4. **实用超长上下文训练** — 迭代 < 1分钟/步 @ 百万级 tokens

## 与竞争方法对比

| 方法 | 缩放 | 内容相关检索 | 瓶颈 |
|------|------|-------------|------|
| Flash Attention | O(n²) | ✅ | 记忆体优化，计算未改 |
| 固定稀疏 (Sliding Window) | O(n) | ❌ | RULER 失败 |
| SSM (Mamba/RetNet) | O(n) | ❌ | 有损压缩，检索弱 |
| 混合模型 (Jamba/Qwen3) | O(n²) 尾部 | ✅ | 密集层承重，比率不可无限压缩 |
| NSA/DSA (DeepSeek) | O(n²) Lightning Indexer | ✅ | Indexer 二次方，52K后超教师 |
| CSA+HCA (DeepSeek V4) | O(n²) | ✅ | Lightning Indexer 仍二次方 |
| **SSA** | **O(n) 全链路** | **✅** | — |

## DSA vs SSA (选定位预算相同)

| 长度 | DSA 层 FLOPs | SSA 层 FLOPs | DSA/SSA |
|------|-------------|-------------|---------|
| 1M   | 9.70P       | 568.3T      | 17.1×   |
| 12M  | 1,305.4P    | 6.82P       | 191.3×  |

## 训练方法

- **起点**: 捐赠模型 (262K 上下文) → 替换密集注意力为 SSA
- **扩展路径**: 262K → 512K → 1M → 2M (YaRN 位置缩放 + CPT 间隔执行)
- **关键发现**: 长上下文 CPT 量 是最一致的检索收益预测因子（>100 次实验）
- **能力平衡**: 检索优化会退化短上下文能力 → 需要分阶段后训练（针对性 + 恢复）
- **打包策略**: 不遮罩跨文檔注意力边界（同 UltraLong/DeepSeek-V3）
- **损失聚合**: 样本级平均（sample-level loss aggregation）防止极长样本主导梯度

## 泛化特性

- 主要在 1M tokens 训练，12M tokens 检索仍达 98%
- 内容相关路由不施加固定上下文长度边界
- 编码数据有双重作用：改善代码能力 + 通用路由行为（跨位置依赖密集）

## 实践启示

- 完整工件推理 > 分块检索：碎片化系统性破坏跨文檔关系
- 某些 RAG/编排支架是"上下文稀缺性"的产物，非任务本质
- 高效注意力是研究加速器：实验吞吐量（>100次长上下文实验）产生发现，非单次运行
- MiniMax M2 回归全注意力的教训：单降渐近成本不够，检索品质+推理成熟度需同时保留
