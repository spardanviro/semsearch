# 阶段 0 报告：可行性验证

> 日期：2026-11-02　机器：RTX 3060 Laptop 6GB / Windows 11 / Python 3.12
> 脚本：`spikes/phase0_model.py`、`spikes/phase0_infra.py`；原始数据：`spikes/results/*.json`

## 结论：可行，进入阶段 1。设计有 4 处调整（见文末）。

## 1. 环境
| 项 | 结果 |
|---|---|
| torch | 2.14.1+cu130，CUDA 可用 |
| sentence-transformers / transformers | 6.2.0 / 5.22.1 |
| 模型 | `google/embeddinggemma-2`，不需要登录，权重 1.49GB |
| 额外依赖 | 即使只加载文本部分，也必须装 **Pillow + torchvision**（加载时会初始化图像处理器） |

## 2. 任务前缀（来自 `config_sentence_transformers.json`）
| prompt_name | 实际前缀 |
|---|---|
| `SearchQuery` / `query` / `Retrieval-query` | `task: search result \| query: ` |
| `CodeRetrieval` | `task: code retrieval \| query: ` |
| `Document` / `document` | `title: none \| text: ` |

另外还有 Classification、Clustering、STS、QuestionAnswering 等，检索用不到。

## 3. 资源占用
| 加载方式 | 参数量 | 加载耗时 | 加载后显存 |
|---|---|---|---|
| 仅文本 | 271M | 2.3s | 0.52GB |
| 全模态 | 744M | 4.6s | 1.96GB |

编码吞吐（仅文本、bf16、真实代码块，平均 206 tokens）：

| batch | 块/秒 | 峰值显存 |
|---|---|---|
| 8 | 65 | 1.5GB |
| **16** | **69** | **2.5GB** |
| 32 | 57 | 4.4GB |
| 64 | 24 | 8.3GB（超出显存，退化到共享内存） |

- 1,116 个块全量编码用了 16.6 秒。按这个速度，1 万个块约 2.5 分钟，满足"< 5 分钟"的标准。
- 你的大型 C# 项目（例如 1 万多个 .cs 文件）首次索引估计要 20–30 分钟，必须在后台进行。阶段 1 可以按 token 预算组 batch，看能否再提速。
- 峰值显存主要由 batch 里最长的块决定，所以要按 token 总量控制 batch，而不是固定条数。

## 4. 仅文本加载 vs 全模态加载
- 对同一批文本：余弦相似度最小 0.9962，均值约 1.0。
- 两者**不完全一致**，没达到预设的 0.999 阈值。差异很小，但不能混用。
- 处理方式：索引元数据里记录加载方式，加载方式变化就全量重建。

## 5. 检索质量（Python 标准库 23 个模块，1,116 个块，30 条查询：20 条英文、10 条中文）
| 前缀 | 维度 | R@1 | R@5 | R@10 | MRR |
|---|---|---|---|---|---|
| **SearchQuery** | **768** | **0.50** | **0.80** | **0.90** | **0.654** |
| SearchQuery | 256 | 0.50 | 0.80 | 0.87 | 0.622 |
| SearchQuery | 128 | 0.57 | 0.73 | 0.83 | 0.630 |
| CodeRetrieval | 768 | 0.43 | 0.73 | 0.90 | 0.583 |
| CodeRetrieval | 256 | 0.43 | 0.73 | 0.87 | 0.576 |
| CodeRetrieval | 128 | 0.33 | 0.67 | 0.83 | 0.477 |

- **用 `SearchQuery` + 768 维**。`CodeRetrieval` 虽然名字看起来更合适，实测反而更差。
- 降到 128 维时，中文查询掉分明显。
- 中文比英文弱：768 维下，英文 R@10 是 95%，中文是 80%。漏掉的中文查询有"把 URL 查询字符串解析成字典"和"解析 INI 配置文件"。改进思路：
  - subagent 检索时，同时用英文改写再查一次（代码和注释大多是英文）；
  - 阶段 1.5 加 FTS5 关键词召回做混合检索。
- 这是小样本（30 条查询），只用于选型，不代表最终效果。阶段 1.5 会用真实项目建更大的评估集。

## 6. 多模态冒烟测试（全模态模型）
- 图片编码正常，输出 768 维，峰值显存 2.0GB。
- 一张纯蓝色方块图：和"a solid blue square"的相似度是 0.74，和"python source code for an HTTP client"是 0.56。方向正确，跨模态检索可用。

## 7. 存储：sqlite-vec
| 项 | 结果 |
|---|---|
| Windows 加载扩展 | 正常（SQLite 3.50.4，sqlite-vec v0.1.9） |
| 写入 5 万条 768 维向量 | 9.5s |
| KNN（k=32） | **93ms** |
| numpy 暴力计算（同样数据） | **3ms** |
| 删除并重写 5 千条 | 6.1s |
| 5 万条时的库大小 | 163MB |
| FTS5 | 可用 |

结论：sqlite-vec 能用，但 vec0 在这个版本里是暴力扫描，速度比 numpy 慢 30 倍，写入也慢。**改用"SQLite 存向量（BLOB）作为数据源 + 后台服务内存里维护 numpy 矩阵来检索"**。

## 8. 分块：tree-sitter-language-pack
- Python、TypeScript、Go、C# 的语法都能正常加载，函数、方法、类都能正确识别。
- 拿你 RaftMod 项目里最大的 C# 文件测试：101KB、1,968 行，解析只要 21ms，识别出 99 个定义。
- 有 21 个块（约 2%）超过 1,024 tokens，最长的 3,157 tokens。大类和大函数必须二次拆分。

## 9. 其他发现
- 本机没有 `rg`（ripgrep）。Scanner 先用 `pathspec` 解析 .gitignore 作为默认实现，装了 rg 就优先用 rg。
- 之前下载的 unsloth GGUF（532MB）用不上了，可以删掉。

## 对设计的调整
1. **依赖**：加 Pillow 和 torchvision；去掉 sqlite-vec。
2. **存储与检索**：向量以 BLOB 存在 `chunks` 表；后台服务启动时把向量载入 numpy 矩阵，增量更新时同步修改；KNN 用矩阵乘法。10 万个块约占 300MB 内存（float32）。
3. **Embedder**：固定用 `SearchQuery` 和 `Document` 前缀、768 维；按 token 预算组 batch（目标峰值显存 ≤ 2.5GB）；输出先转 float32 再归一化；元数据里记录加载方式。
4. **分块**：硬上限 1,024 tokens，超长的定义按子节点或行窗口二次拆分；subagent 提示词里加入"中文查询同时用英文改写再查一次"。
