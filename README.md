# Kimi2-PCL

`Kimi2-PCL` 是一个基于 Kimi2 模型进行修改的模型架构仓库，模型定义代码参考 Hugging Face 模型 [Kimi-K2-Base](https://huggingface.co/moonshotai/Kimi-K2-Base)，权重转换脚本参考 MindSpeed-LLM 脚本 [DeepSeek](https://gitcode.com/Ascend/MindSpeed-LLM/blob/2.2.0/examples/mcore/deepseek3/convert_ckpt_deepseek3.py)：

- [models](./models) 提供模型的定义与配置（Hugging Face 风格的 config / generation config + Python 实现）。
- [utils](./utils) 提供权重格式转换脚本（HF safetensors ↔ Megatron-Core/MCore checkpoint）。
- [scripts](./scripts) 提供权重转换与预训练启动脚本（torchrun + Megatron/MindSpeed 生态）。
- [vllm-infer](./vllm-infer) 提供基于 vLLM + Ray 的 Ascend 分布式推理脚本及指南。

## 目录结构

```text
Kimi2-PCL/
  models/
    __init__.py
    config_100b.json
    config_1t.json
    configuration_deepseek_100b.py
    configuration_deepseek_1t.py
    generation_config.json
    modeling_deepseek.py
  utils/
    convert_ckpt_hf2mcore.py
    convert_ckpt_mcore2hf.py
  scripts/
    ckpt_convert_hf2mcore.sh
    ckpt_convert_mcore2hf.sh
    pretrain_kimi2_100b_4k.sh
    pretrain_kimi2_1000b_4k.sh
    pretrain_kimi2_1t_4k.sh
  vllm-infer/
    ascend_infer_docker_run.sh
    cluster_deploy_ray_vllm.sh
    vllm_model_server.sh
    node_list.txt
  vllm_infer_with_ray.md
  README.md
```

## 模型与配置（models）

- `config_1t.json` / `config_100b.json`：Hugging Face 风格模型配置（不同规模的示例配置）。
- `generation_config.json`：Hugging Face generation 配置。
- `configuration_deepseek_1t.py` / `configuration_deepseek_100b.py`：定义了 `DeepseekV3Config`（`model_type = "kimi_k2"`）。
- `modeling_deepseek.py`：完整的 DeepSeek HF 实现代码（包含 RMSNorm / RoPE / MoE MLP 等核心结构）。

注意：`config_1t.json` / `config_100b.json` 中的 `auto_map` 当前指向 `configuration_deepseek.*` / `modeling_deepseek.*`（而本仓库 `models/` 下默认是 `configuration_deepseek_*.py`）。若你希望直接通过 `transformers.AutoModel*` + `trust_remote_code=True` 加载本地模型目录，需要保证模型目录内存在与 `auto_map` 一致的 Python 文件与类名（例如补齐 `configuration_deepseek.py`，或改写 `auto_map` 指向 `configuration_deepseek_1t.py` / `configuration_deepseek_100b.py`）。

## 权重转换（scripts/utils）

本仓库提供两类转换：

- HF（`model.safetensors*` + `model.safetensors.index.json`）→ MCore/Megatron checkpoint 目录（`iter_xxxxxxx/mp_rank_*/model_optim_rng.pt`）
- MCore/Megatron checkpoint → HF safetensors 分片与 index

### HF → MCore

可以通过执行 `scripts` 下的转换脚本进行一键转换：

```bash
bash scripts/ckpt_convert_hf2mcore.sh
```

如需自定义转换参数，可以参考底层入口脚本 `utils/convert_ckpt_hf2mcore.py` 的用法：

```bash
python utils/convert_ckpt_hf2mcore.py \
  --load-dir /path/to/hf_model_dir \
  --save-dir /path/to/mcore_ckpt_dir \
  --target-tensor-parallel-size 2 \
  --target-pipeline-parallel-size 8 \
  --target-expert-parallel-size 64 \
  --pp-workers 2 \
  --moe-grouped-gemm \
  --schedules-method dualpipev \
  --noop-layers 31 \
  --rotary-base 50000
```

常用参数说明：

- `--load-dir`：Hugging Face 模型目录（包含 `model.safetensors.index.json`）。
- `--save-dir`：输出 checkpoint 目录（脚本会创建 `iter_0000001/` 并写入 `latest_checkpointed_iteration.txt`）。
- `--num-layers` / `--hidden-size` / `--num-attention-heads` 等：通常可从 HF `config.json` 自动推断；当你的 HF 配置字段不完整或需要覆盖时再显式指定。
- `--target-*-parallel-size`：目标 TP/PP/EP 并行度。
- `--pp-workers`：PP workers 的数量（用于特殊调度模式下的映射计算）。
- `--moe-grouped-gemm`：MoE 权重布局相关开关。
- `--vpp-stage` / `--schedules-method dualpipev`：VPP/dualpipev 相关配置。
- `--num-layer-list`：当 `num_layers` 不能整除 `pp_size` 时，显式给定每个 PP 的层数分配（形如 `4,4,4,4`）。
- `--noop-layers`：指定 noop layer 索引列表（逗号分隔），用于与训练侧的 “跳层/空层” 配置对齐。
- `--qlora-nf4`：输出层权重做 bitsandbytes nf4 量化（需要额外安装 `bitsandbytes`）。
- `--cast-dtype`：可选，对浮点权重做 dtype 转换（`fp32`/`bf16`/`fp16`）。
- `--sha256-manifest`：输出分块 sha256 清单（json），用于做文件级一致性校验。

### MCore → HF

同样，可以通过执行 `scripts` 下的转换脚本进行一键转换：

```bash
bash scripts/ckpt_convert_mcore2hf.sh
```

如需自定义转换参数，可以参考底层入口脚本 `utils/convert_ckpt_mcore2hf.py` 的用法：

```bash
python utils/convert_ckpt_mcore2hf.py \
  --load-dir /path/to/mcore_ckpt_dir \
  --save-dir /path/to/hf_model_dir \
  --pp-workers 2 \
  --io-threads 4 \
  --num-layers 32 \
  --first-k-dense-replace 2 \
  --source-tensor-parallel-size 2 \
  --source-pipeline-parallel-size 8 \
  --source-expert-parallel-size 64 \
  --hf-config-template models/config_1t.json \
  --moe-grouped-gemm \
  --schedules-method dualpipev \
  --noop-layers 31 \
  --rotary-base 50000 \
  --hidden-size 7168 \
  --ffn-hidden-size 18432 \
  --moe-ffn-hidden-size 12288 \
  --vocab-size 163840 \
  --num-experts 128 \
  --num-attention-heads 64 \
  --num-query-groups 2 \
  --qk-head-dim 128 \
  --v-head-dim 128 \
  --qk-pos-emb-head-dim 64
```

`--hf-config-template` 用于指定输出目录下 `config.json` 的模板；例如 100B 配置可使用 `models/config_100b.json`。

输出产物包括：

- `model-00001-of-0000XX.safetensors` 分片文件
- `model.safetensors.index.json`

### 对齐验证（tests）

当 `scripts/pretrain_kimi2_1t_4k.sh` 或 `utils/` 下转换逻辑变更时，可运行对齐回归测试：

```bash
python -m unittest -v tests/weights/test_align_pretrain_config.py
```

仓库包含一个 GitHub Actions 工作流片段：.github/workflows/weights.yml，会在相关路径变更时自动触发该测试。

## 训练脚本（scripts）

脚本：

- [pretrain_kimi2_100b_4k.sh](./scripts/pretrain_kimi2_100b_4k.sh)
- [pretrain_kimi2_1000b_4k.sh](./scripts/pretrain_kimi2_1000b_4k.sh)
- [pretrain_kimi2_1t_4k.sh](./scripts/pretrain_kimi2_1t_4k.sh)

以上脚本都是示例启动脚本，核心行为是调用：

```bash
torchrun ... pretrain_gpt.py ...
```

其中 `pretrain_gpt.py` 以及相关的运行时依赖（例如 `mindspeed_llm.*`、Megatron/MCore 等）不在本仓库内，通常来自你的训练框架工程或运行环境。

脚本中常见依赖的环境变量（按你的集群/启动器而定）：

- 分布式：`LOCAL_WORLD_SIZE`、`server_count`、`RANK`、`MASTER_ADDR`、`MASTER_PORT`
- Tokenizer：`TOKENIZER_PATH`
- 数据：`DATA_PREFIXES`（以及部分脚本里使用的 `DATA_DIR`）
- Checkpoint：`CKPT_LOAD_DIR`、`CKPT_SAVE_DIR`
- 日志：`TRAIN_LOG_PATH`

## 推理服务（vllm-infer）

本仓库提供了针对 Ascend NPU 的分布式推理一键部署脚本及详细指南，支持单机多卡与多机多卡的快速拉起。详细使用说明请参考：

- [Ray 分布式推理指南 (vLLM Ascend)](./vllm_infer_with_ray.md)

### 快速开始（多机部署）

通过配置好 `node_list.txt`，可以使用下述一键脚本完成镜像加载、容器创建、Ray 集群组建及 vLLM 服务拉起：

```bash
bash vllm-infer/cluster_deploy_ray_vllm.sh
```

### 目录核心脚本

- `ascend_infer_docker_run.sh`: 用于启动挂载了 Ascend NPU 驱动及设备目录的容器环境。
- `cluster_deploy_ray_vllm.sh`: 一键部署的主入口脚本，支持步骤级控制（`--prepare-only`, `--ray-only`, `--serve-only`）。
- `vllm_model_server.sh`: vLLM 服务的实际启动脚本，支持通过环境变量灵活覆盖各类参数。
- `node_list.txt`: 参与分布式部署的节点列表配置文件。

## 开发与代码质量

仓库包含基础的代码质量工具配置：

- `.pre-commit-config.yaml`：pre-commit hooks
- `.flake8`：flake8 配置

如需启用：

```bash
pip install pre-commit
pre-commit install
```

## 许可证

见 [LICENSE](./LICENSE)。
