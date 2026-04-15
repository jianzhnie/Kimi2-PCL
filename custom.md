# Kimi2-PCL 模型架构实现

## 模型训练

模型基于 Mindspeed-LLM(Megatron) 生态进行训练，使用 [scripts/pretrain_kimi2_1t_4k.sh](./scripts/pretrain_kimi2_1t_4k.sh) 作为预训练启动脚本。
训练的权重保存为 Megatron-Core/MCore checkpoint 格式，包含模型权重、优化器状态、训练进度等信息。

model_param_mapping.json 是原始的Mcore权重(合并所有参数), 包含了模型的所有的所有参数映射。
model_param_hf.json 是从Mcore 转为Huggingface 模型 模型得到的参数映射。

## 任务目标

### 1. 模型架构与配置文件一致性验证
对以下文件进行逐项比对分析：
- **架构文件**：[models/modeling_deepseek.py]
- **配置文件**：[models/configuration_deepseek.py]
- **基准脚本**：[scripts/pretrain_kimi2_1t_4k.sh]

### 2. 检查和验证
- 模型架构：验证模型层结构（如 MLP 层, MoE 层, GQA 等实现）与预训练架构文件的一致性。
- 模型层数、隐藏层维度、注意力头数等核心超参数的一致性
- 这是 GQA模块， 不要实现成MLA, 移除 MLA 中 ROPE 的单独处理


### 权重转换对齐
对双向转换 Python 脚本进行深度代码审查：

**Megatron-Core → Hugging Face 转换**：[utils/convert_ckpt_mcore2hf.py]
- 验证参数名映射表的完整性（如 qkv_weight → q_proj/k_proj/v_proj）
- 检查张量形状变换逻辑（特别是注意力权重 reshape 操作）
- 确认分布式训练状态字典的合并策略正确性
- 验证特殊层（如 embedding、layernorm）的转换公式准确性

**Hugging Face → Megatron-Core 转换**：[utils/convert_ckpt_hf2mcore.py]
- 验证逆向映射的数学等价性（确保双向转换可逆）
- 检查张量分片逻辑是否符合 Megatron-Core 的并行策略
- 确认 1T 参数模型下的内存优化实现
- 验证转换后检查点的加载测试通过性

### 权重转换脚本功能验证
对 Shell 脚本进行检查和测试：

**转换脚本检查**：
- [scripts/ckpt_convert_mcore2hf.sh]
- [scripts/ckpt_convert_hf2mcore.sh]

# 模型转换正确性检查

## 模型训练

模型基于 Mindspeed-LLM(Megatron) 生态进行训练，使用 [scripts/pretrain_kimi2_1t_4k.sh](./scripts/pretrain_kimi2_1t_4k.sh) 作为预训练启动脚本。
训练的权重保存为 Megatron-Core/MCore checkpoint 格式，包含模型权重、优化器状态、训练进度等信息。。

## 任务目标

### 权重转换对齐
对双向转换 Python 脚本进行深度代码审查：

**Megatron-Core → Hugging Face 转换**：[utils/convert_ckpt_mcore2hf.py]
- 验证参数名映射表的完整性（如 qkv_weight → q_proj/k_proj/v_proj）
- 检查张量形状变换逻辑（特别是注意力权重 reshape 操作）
- 确认分布式训练状态字典的合并策略正确性
- 验证特殊层（如 embedding、layernorm）的转换公式准确性

**Hugging Face → Megatron-Core 转换**：[utils/convert_ckpt_hf2mcore.py]
- 验证逆向映射的数学等价性（确保双向转换可逆）
- 检查张量形状变换逻辑（特别是注意力权重 reshape 操作）
- 检查张量分片逻辑是否符合 Megatron-Core 的并行策略
- 验证各种 HF layer 特殊层（如 embedding、layernorm, Attention, GQA, MOE, ...）的转换公式准确性

### 权重转换脚本功能验证
对 Shell 脚本进行检查和测试：

**转换脚本检查**：
- [scripts/ckpt_convert_mcore2hf.sh]
- [scripts/ckpt_convert_hf2mcore.sh]



model_param_hf.json 是 Huggingface 模型 模型参数映射。
- **架构文件**：[models/modeling_deepseek.py]
- **配置文件**：[models/configuration_deepseek.py]
- **基准脚本**：[scripts/pretrain_kimi2_1t_4k.sh]
请帮我检查和验证修改 modeling_deepseek.py 及其配置， 确保和 model_param_hf.json 中的参数映射对齐。