# PCl-Model-1T 大语言模型评测报告

## 1. 评测概览

本报告记录了 PCL-Model-1T 大语言模型在预训练阶段的关键性能指标。

- **模型阶段**: 基于预训练 10,000 Step 保存的 Checkpoint。
- **训练规模**: 已完成约 4.0T Tokens 的训练。
- **评测环境**: 采用 `vLLM` 推理引擎配合 `lmeval` 框架。
- **硬件平台**: 昇腾 (Ascend) NPU 集群。

---

## 2. 核心基准测试结果

### 2.1 MMLU (Massive Multitask Language Understanding)

MMLU 是衡量语言模型综合理解能力的核心基准，涵盖了从基础教育到专业水平的 57 个主题。评测结果分为四个大类：**人文 (Humanities)**、**社会科学 (Social Sciences)**、**STEM (科学、技术、工程、数学)** 以及 **其他 (Other)**。

#### 2.1.1 Zero-shot 结果 (n-shot: 0)

| 任务 (Task) | 准确率 (acc) ↑ | 标准误差 (acc_stderr) |
| :--- | :---: | :---: |
| **MMLU (Overall)** | **0.3441** | 0.0040 |
| - Humanities | 0.3092 | 0.0067 |
| - Social Sciences | 0.3776 | 0.0087 |
| - STEM | 0.3283 | 0.0083 |
| - Other | 0.3798 | 0.0087 |

#### 2.1.2 Few-shot 结果 (n-shot: 5)

| 任务 (Task) | 准确率 (acc) ↑ | 标准误差 (acc_stderr) |
| :--- | :---: | :---: |
| **MMLU (Overall)** | **0.3506** | 0.0040 |
| - Humanities | 0.3184 | 0.0067 |
| - Social Sciences | 0.3760 | 0.0087 |
| - STEM | 0.3279 | 0.0083 |
| - Other | 0.3972 | 0.0087 |

> **数据说明**：
> - **Task**: `mmlu` 为全量 57 个子任务的宏平均结果；各子分类（如 Humanities）为其下属主题的加权平均。
> - **准确率 (acc)**: 模型预测正确的样本比例，已四舍五入保留 4 位小数。
> - **标准误差 (acc_stderr)**: 反映评测结果的统计稳定性，数值越小表示结果越可靠。



#### 2.1.3 MMLU 5-shot 对比 — 主流大模型

| 模型 | 参数量 | MMLU 5-shot (%) | 架构 | 来源 |
| :--- | :---: | :---: | :---: | :--- |
| **Kimi K2 Base** | 1T (MoE) | **87.8** | MoE | Moonshot AI 技术报告 |
| **DeepSeek-V3** | 671B (MoE, 37B 激活) | **87.1** | MoE | DeepSeek 技术报告 |
| **Llama-3.1-405B** | 405B | **85.2** | Dense | Meta 官方模型卡 |
| **Qwen2.5-72B** | 72.7B | **85.0** | Dense | Qwen2.5 技术报告 |
| **Llama-4-Maverick** | 400B (MoE, 17B 激活) | **84.9** | MoE | Meta 官方模型卡 |
| **Qwen2-72B** | 72B | **84.2** | Dense | Qwen2 官方模型卡 |
| **GPT-4** | 闭源 | **83.0** | — | OpenAI 报告 |
| **Llama-3-70B** | 70B | **79.5** | Dense | Meta 官方模型卡 |
| **Llama-3.1-70B** | 70B | **79.3** | Dense | Meta 官方模型卡 |
| **DeepSeek-V2** | 236B (MoE, 21B 激活) | **78.5** | MoE | DeepSeek-V2 技术报告 |
| **Mixtral-8x22B** | 141B (MoE, 39B 激活) | **77.6** | MoE | Mistral AI |
| **Yi-34B** | 34B | **~76.0** | Dense | 01.AI |
| **GLM-4-9B** | 9B | **74.7** | Dense | 智谱 AI |
| **Qwen2.5-7B** | 7.61B | **~72.0** | Dense | Qwen2.5 技术报告 |
| **Qwen2-7B** | 7.6B | **70.3** | Dense | Qwen2 官方模型卡 |
| **Mixtral-8x7B** | 46.7B (MoE, 12.9B 激活) | **~70.6** | MoE | Mistral AI |
| **GPT-3.5** | 闭源 | **69.1** | — | OpenAI 报告 |
| **Llama-2-70B** | 70B | **69.7** | Dense | Meta 官方模型卡 |
| **Llama-3.1-8B** | 8B | **66.7** | Dense | Meta 官方模型卡 |
| **Llama-3-8B** | 8B | **66.6** | Dense | Meta 官方模型卡 |
| **InternLM2-7B** | 7B | **65.8** | Dense | 上海 AI 实验室 |
| **Mistral-7B-v0.1** | 7.3B | **64.2** | Dense | Mistral AI |
| **Yi-6B** | 6B | **~63.0** | Dense | 01.AI |
| **Llama-2-13B** | 13B | **53.8** | Dense | Meta 官方模型卡 |
| **DeepSeek-LLM-7B** | 7B | **~48.0** | Dense | DeepSeek LLM 技术报告 |
| **Llama-2-7B** | 7B | **45.7** | Dense | Meta 官方模型卡 |
| **PCL-Model-1T** | — | **35.06** | — | 本报告 (预训练 10K Step) |

> **数据来源**：各模型官方模型卡 (HuggingFace)、技术报告及社区评测。标注 `~` 的数值为社区评测或基于技术报告图表的估计值。
>
> **指标差异说明**：不同模型/评测框架使用的 MMLU 评测指标可能不同。例如 Llama 系列使用 `macro_avg/acc_char`，DeepSeek-V3 和 Kimi K2 使用 `EM` (Exact Match)。同一模型在不同指标下分数可能不同（如 Llama-4-Maverick: 85.5 acc_char vs 84.9 EM；Qwen2.5-72B: 85.0 DeepSeek-V3 表 vs 86.1 Kimi K2 表）。本表优先采用各模型官方模型卡报告值。
>
> **分析**：PCL-Model-1T 当前处于预训练早期阶段 (10,000 Step / ~4.0T Tokens)，MMLU 指标随训练推进预计将持续提升。成熟模型（如 Llama-3、Qwen2.5 等）均在完整预训练后方进行评测，MMLU 指标通常在训练中后期才趋于收敛。


---

### 2.2 Wikitext-2 Perplexity (语言建模能力)

Wikitext-2 数据集用于评估模型对高质量维基百科文本的语言建模能力。**困惑度 (Perplexity, PPL)** 是衡量模型预测文本序列能力的关键指标。

#### 2.2.1 Wikitext-2 评测结果

| 评测指标 (Metric) | 结果 (Value) ↓ | 说明 |
| :--- | :---: | :--- |
| **Word Perplexity** | **15.7494** | 基于单词层面的预测困惑度 |
| Byte Perplexity | 1.6745 | 基于字节层面的预测困惑度 |
| Bits per Byte (BPB) | 0.7438 | 平均每个字节所需的比特数 |

#### 2.2.2 主流模型 Word Perplexity 跨模型对比

> **注意**：由于各模型分词器（Tokenizer）和词表大小不同，Word PPL 仅在同分词器模型间具有严格可比性。下表数据来源于各模型官方模型卡及原始论文，供趋势性参考。BPB（见 2.2.2 节）仍是跨模型最公平的对比指标。

##### GPT-2 系列 (WikiText-2 / PTB / LAMBADA)

| 模型 | 参数量 | WikiText-2 PPL ↓ | PTB PPL ↓ | LAMBADA PPL ↓ |
| :--- | :---: | :---: | :---: | :---: |
| GPT-2 Small | 124M | 29.41 | 65.85 | 35.13 |
| GPT-2 Medium | 355M | 22.76 | 47.42 | 25.69 |
| GPT-2 Large | 774M | 19.93 | 38.46 | 19.94 |
| GPT-2 XL | 1.5B | 18.34 | 35.76 | 17.70 |
| GPT-3 | 175B | ~16.44* | 20.50 | 8.11 |

> **数据来源**：GPT-2 数据来自 [HuggingFace GPT-2 模型卡](https://huggingface.co/openai-community/gpt2) (Radford et al., 2019)；GPT-3 数据来自 Brown et al. (2020) 原始论文。*GPT-3 的 WikiText-2 PPL 值 ~16.44 为社区广泛引用值，原始论文未直接报告此指标。

##### LLaMA 系列 (WikiText-2)

| 模型 | 参数量 | WikiText-2 PPL ↓ | 备注 |
| :--- | :---: | :---: | :--- |
| LLaMA 1 7B | 7B | ~5.68 | 社区评测估计值 |
| LLaMA 1 13B | 13B | ~5.09 | 社区评测估计值 |
| LLaMA 1 33B | 33B | ~4.10 | 社区评测估计值 |
| LLaMA 1 65B | 65B | ~3.53 | 社区评测估计值 |
| LLaMA 2 7B | 7B | ~5.47 | 社区评测估计值 |
| LLaMA 2 13B | 13B | ~4.88 | 社区评测估计值 |
| LLaMA 2 70B | 70B | ~3.32 | 社区评测估计值 |
| LLaMA 3 8B | 8B | ~5.25 | 社区评测估计值 |
| LLaMA 3 70B | 70B | ~2.85 | 社区评测估计值 |

> **数据来源**：LLaMA 1/2/3 原始论文均未直接报告 WikiText-2 PPL，上述数值均为社区评测 (lm-evaluation-harness) 估计值，仅供参考。LLaMA 官方评测聚焦于下游任务基准 (MMLU、HellaSwag 等)。

##### Qwen2.5 系列 (WikiText-2)

| 模型 | 参数量 | WikiText-2 PPL ↓ | 备注 |
| :--- | :---: | :---: | :--- |
| Qwen2.5-7B | 7.6B | ~5.5* | 社区估计值 |
| Qwen2.5-72B | 72B | ~2.9* | 社区估计值 |

> *Qwen 官方技术报告未直接报告 WikiText-2 PPL，上述数值为社区评测估计值，仅供参考。

##### 趋势分析

1. **规模效应**：模型参数量从 ~100M (GPT-2 Small) 增长至 175B (GPT-3)，WikiText-2 PPL 从 29.41 降至 ~16.44；从 7B 到 70B 级别，PPL 进一步降至 ~3.3 以下，体现了显著的规模效应。
2. **数据与训练的重要性**：LLaMA 2 7B (5.47) 优于 LLaMA 1 7B (5.68)，尽管架构相似，更多、更高质量的训练数据带来了实质提升。
3. **跨模型可比性**：PPL 高度依赖分词器设计、词表大小、上下文长度等因素。例如 LLaMA 3 采用 128K 词表，LLaMA 2 为 32K，这会直接影响 PPL 数值。因此，**同架构/同分词器模型间的 PPL 对比更有意义**。
