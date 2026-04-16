## 优化 Prepare Docker Nodes 脚本逻辑

优化逻辑 /Users/jianzhengnie/work_dir/Kimi2-PCL/vllm-infer/prepare_docker_nodes.sh,

```bash
1. 默认的情况
  - 确保 Docker 命令可用
    - 如果Docker命令不可用，则执行 systemctl daemon-reload && systemctl start docker
    - 可用则继续执行后续步骤
  - 不进行  stop & kill Docker 容器的操作

2. 使用 restart 参数
  - 确保 Docker 命令可用
    - 如果Docker命令不可用，则执行 systemctl daemon-reload && systemctl start docker
    - 可用则继续执行后续步骤
  - 执行 stop & kill Docker 容器的操作
    - 停止所有正在运行的 Docker 容器
    - 杀死所有正在运行的 Docker 容器
  - 继续执行后续步骤
```


模型基于 Mindspeed-LLM(Megatron) 生态进行训练，使用 GQA + MOE 架构，[scripts/pretrain_kimi2_1t_4k.sh](./scripts/pretrain_kimi2_1t_4k.sh) 作为预训练启动脚本。

任务：

1. 检查[scripts/ckpt_convert_mcore2hf.sh](scripts/ckpt_convert_mcore2hf.sh)
2. 帮我实现 MCore --> Huggingface 模型转换相关定义，[utils/convert_ckpt_mcore2hf.py](utils/convert_ckpt_mcore2hf.py)
3. 这是 GQA模块， 不要实现成MLA, 移除 MLA 中 ROPE 的单独处理
4. 注意不要阅读 /Users/jianzhengnie/work_dir/Kimi2-PCL/models 任何文件

# MCore --> Huggingface 转换和实现

模型基于 Mindspeed-LLM(Megatron) 生态进行训练，使用 [scripts/pretrain_kimi2_1t_4k.sh](./scripts/pretrain_kimi2_1t_4k.sh) 作为预训练启动脚本。

现在通过inspect 方式把该模型类 [megatron_model.py](./megatron_model.py)和模型参数[model_param_mapping_extracted.json]（只有4层） 给 Dump 下来了.
[model_dimension_explanation.md](model_dimension_explanation.md) 是对模型架构和参数维度的分析和拆解。

下面是我实现的 Huggingface 模型相关定义

- **架构文件**：[models/modeling_deepseek.py]
- **配置文件**：[models/configuration_deepseek.py]
- **基准脚本**：[scripts/pretrain_kimi2_1t_4k.sh]

## 任务

请你参考上面内容实现 Huggingface 模型相关定义。


# MCore --> Huggingface 转换和实现

模型基于 Mindspeed-LLM(Megatron) 生态进行训练，使用 [scripts/pretrain_kimi2_1t_4k.sh](./scripts/pretrain_kimi2_1t_4k.sh) 作为预训练启动脚本。

现在通过inspect 方式把该模型类 [megatron_model.py](./megatron_model.py)和模型参数[model_param_mapping_tp1.json]（只有4层） 给 Dump 下来了.
[model_dimension_explanation.md](model_dimension_explanation.md) 是对模型架构和参数的分析和拆解。


## 任务

请你实现 MCore --> Huggingface  模型权重相关定义和转换。


# MCore <--> Huggingface 转换和实现

模型基于 Mindspeed-LLM(Megatron) 生态进行训练，使用 [scripts/pretrain_kimi2_1t_4k.sh](./scripts/pretrain_kimi2_1t_4k.sh) 作为预训练启动脚本。

Mindspeed-LLM 通过inspect 方式把该模型类给 Dump 下来了 [megatron_model.py](./megatron_model.py)，[model_param_mapping_tp1] 是dump 下来的参数映射(只使用了4层的模型)

下面是我实现的 MCore <--> Huggingface 模型转换相关定义

[utils/convert_ckpt_hf2mcore.py](utils/convert_ckpt_hf2mcore.py)
[scripts/ckpt_convert_hf2mcore.sh](scripts/ckpt_convert_hf2mcore.sh)





等参数定义在所有文件中的含义保持一致

- **架构文件**：[models/modeling_deepseek.py]
- **配置文件**：[models/config.json]
- **配置文件**：[models/configuration_deepseek.py]
- **基准脚本**：[scripts/pretrain_kimi2_1t_4k.sh]
[utils/convert_ckpt_hf2mcore.py](utils/convert_ckpt_hf2mcore.py)
[scripts/ckpt_convert_hf2mcore.sh](scqripts/ckpt_convert_hf2mcore.sh)
[utils/convert_ckpt_mcore2hf.py](utils/convert_ckpt_mcore2hf.py)
[scripts/ckpt_convert_mcore2hf.sh](scripts/ckpt_convert_mcore2hf.sh)


模型基于 Mindspeed-LLM(Megatron) 生态进行训练，使用 [scripts/pretrain
_kimi2_1t_4k.sh](./scripts/pretrain_kimi2_1t_4k.sh) 作为预训练启动脚本。

测试代码 [tests/test_get_mcore_weights_from_ckpt.py](tests/test_get_mcore_weights_from_ckpt.py)
用于读取 mcore 格式的模型权重, 然后提取模型权重的所有信息, 输出到json 文件。
请帮我检查和Review代码正确性, 如果有错误, 请直接修复代码。


# Mcore --> Huggingface 转换和实现

模型基于 Mindspeed-LLM(Megatron) 生态进行训练，使用 GQA + MOE 架构，[scripts/pretrain_kimi2_1t_4k.sh](./scripts/pretrain_kimi2_1t_4k.sh) 作为预训练启动脚本。
model_param_mapping.json 是原始的Mcore权重(合并所有参数), 包含了模型的所有的所有参数映射。
utils/convert_ckpt_deepseek3_mcore2hf.py 是 deepseek3 模型的转换代码，用于将 deepseek3 模型的 HF 模型转换 转换为 Mcore 模型。ßß

任务：
1. 请你参考 utils/convert_ckpt_deepseek3_mcore2hf.py 中的代码， 帮我实现 [convert_kimi2_mcore2hf.py](utils/convert_kimi2_mcore2hf.py) 将 Mcore 模型转换为 HF 模型。
2. 注意我的模型是 GQA + MOE 架构，需要在转换代码中实现 GQA + MOE 的转换， 而不是 MLA 架构。
3. 注意不要阅读 /Users/jianzhengnie/work_dir/Kimi2-PCL/models 任何文件
4. 检查和Review 转换代码是否正确，是否有错误， 请直接修复代码。


# Huggingface --> Mcore 转换和实现

模型基于 Mindspeed-LLM(Megatron) 生态进行训练，使用 GQA + MOE 架构，[scripts/pretrain_kimi2_1t_4k.sh](./scripts/pretrain_kimi2_1t_4k.sh) 作为预训练启动脚本。
model_param_hf.json 是 Huggingface 模型的参数映射文件，包含了模型的所有所有的参数映射。

任务：

1. 检查和Review Huggingface --> Mcore 模型转换[utils/convert_ckpt_hf2mcore.py](utils/convert_ckpt_hf2mcore.py)相关定义是否正确，是否实现多种并行维度转换。
2. 检查和更新[scripts/ckpt_convert_hf2mcore.sh](scripts/ckpt_convert_hf2mcore.sh)
3. 注意不要阅读 /Users/jianzhengnie/work_dir/Kimi2-PCL/models 任何文件
4. 检查和Review 转换代码是否正确，是否有错误， 请直接修复代码。



# Huggingface --> Mcore 转换和实现

模型基于 Mindspeed-LLM(Megatron) 生态进行训练，使用 GQA + MOE 架构，[scripts/pretrain_kimi2_1t_4k.sh](./scripts/pretrain_kimi2_1t_4k.sh) 作为预训练启动脚本。
model_param_hf.json 是 Huggingface 模型的参数映射文件，包含了模型的所有所有的参数映射。
convert_ckpt_deepseek3.py 是 deepseek3 模型的转换代码，用于将 deepseek3 模型的 HF 模型转换 转换为 Mcore 模型。

任务：
1. 请你参考 convert_ckpt_deepseek3.py 中的代码， 帮我实现 [ckpt_ckpt_kimi2_1t_4k.py](utils/ckpt_ckpt_kimi2_1t_4k.py) 将HF 模型转换 转换为 Mcore 模型。
2. 注意我的模型是 GQA + MOE 架构，需要在转换代码中实现 GQA + MOE 的转换， 而不是 MLA 架构。
3. 注意不要阅读 /Users/jianzhengnie/work_dir/Kimi2-PCL/models 任何文件
4. 检查和Review 转换代码是否正确，是否有错误， 请直接修复代码。


# Huggingface --> Mcore 转换和实现

模型基于 Mindspeed-LLM(Megatron) 生态进行训练，使用 GQA + MOE 架构，[scripts/pretrain_kimi2_1t_4k.sh](./scripts/pretrain_kimi2_1t_4k.sh) 作为预训练启动脚本。

model_param_hf.json 是 Huggingface 模型的参数映射文件，包含了模型的所有所有的参数映射。
model_param_mapping.json 是原始的Mcore权重(合并所有参数), 包含了模型的所有的所有参数映射。
model_param_mapping_2.json 是从 Huggingface 模型转为 Mcore 模型得到的参数映射(合并所有参数)。

任务：

1. 检查 Huggingface --> Mcore 模型转换相关定义是否正确，[utils/convert_ckpt_hf2mcore.py](utils/convert_ckpt_hf2mcore.py)实现多种并行维度转换。
2. 发现转完的参数丢失了很多，请你检查和优化转换代码。
3. 检查和更新[scripts/ckpt_convert_hf2mcore.sh](scripts/ckpt_convert_hf2mcore.sh)
4. 注意不要阅读 /Users/jianzhengnie/work_dir/Kimi2-PCL/models 任何文件
