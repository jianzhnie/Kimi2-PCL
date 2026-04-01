# 模型一致性验证与权重转换检查任务

## 任务目标
基于 [scripts/pretrain_kimi2_1t_4k.sh] 预训练脚本，系统性地验证 Kimi2-PCL 项目中的模型架构、配置文件和权重转换工具的一致性与正确性。

## 详细要求

### 1. 模型架构与配置文件一致性验证
对以下文件进行逐项比对分析：
- **架构文件**：[models/modeling_deepseek.py]
- **配置文件**：[models/configuration_deepseek_1t.py] 和 [models/config_1t.json]
- **基准脚本**：[scripts/pretrain_kimi2_1t_4k.sh]

验证维度必须包括：
- 模型架构：验证模型层结构（如 Transformer 编码器层、MLP 层, MoE 层）与架构文件的一致性。
- 模型层数、隐藏层维度、注意力头数等核心超参数的一致性
- 激活函数、归一化方法、位置编码等架构细节的匹配性
- 1T 参数规模下的特殊配置（如张量并行度、流水线并行度）的正确性
- 4K 上下文长度相关的位置编码实现准确性

### 2. 权重转换代码正确性验证
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

### 3. 权重转换脚本功能验证
对 Shell 脚本进行端到端测试：

**转换脚本检查**：
- [scripts/ckpt_convert_mcore2hf.sh]
- [scripts/ckpt_convert_hf2mcore.sh]

验证要求：
- 环境变量配置的正确性（CUDA 可见设备、内存分配等）
- 命令行参数解析的健壮性（输入/输出路径验证、格式检查）
- 错误处理机制的完整性（缺失文件、权限不足、磁盘空间等）
- 转换日志的详细程度（需包含参数统计、内存使用、耗时记录）

### 4. 权重一致性检查
使用 [check_model_weights.py] 进行验证：
- 执行 Megatron-Core → HF → Megatron-Core 的闭环转换测试
- 检查参数总量统计（确保 1T 参数无丢失）
- 运行内存占用监控（确保转换过程无内存泄漏）
