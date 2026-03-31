# MCore 权重文件并行保存优化

## 概述

本次优化旨在提高 `convert_ckpt_hf2mcore.py` 脚本在保存 MCore 权重文件时的性能。原来的实现是串行保存每个 EP×TP rank 的权重文件，这在大规模并行设置（如 TP=2, PP=8, EP=64）下会成为瓶颈。

## 优化内容

### 1. 并行保存实现

在 `_save_pp_rank` 函数中实现了使用线程池并行保存权重文件的功能：

- 将原来嵌套的 `for ep_rank in range(self.ep_size): for tp_rank in range(self.tp_size):` 循环改为收集所有要保存的任务
- 使用 `ThreadPoolExecutor` 并行执行保存操作
- 添加进度日志以便监控保存进度

### 2. 新增配置参数

添加了新的命令行参数：

```bash
--save-workers N    # 控制保存阶段的并行度（0=自动，1=串行，N=使用N个线程）
```

### 3. 自适应并行度

- 自动限制最大线程数为 32（防止资源过度消耗）
- 支持通过环境变量 `CKPT_CONVERT_SAVE_WORKERS` 覆盖设置
- 保持向后兼容性（save-workers=0 或 1 时使用串行模式）

## 使用方法

### 基本使用

```bash
python utils/convert_ckpt_hf2mcore.py \
  --load-dir /path/to/hf_model \
  --save-dir /path/to/mcore_output \
  --target-tensor-parallel-size 2 \
  --target-pipeline-parallel-size 8 \
  --target-expert-parallel-size 64 \
  --save-workers 16  # 使用16个线程并行保存
```

### 自动模式

```bash
python utils/convert_ckpt_hf2mcore.py \
  --load-dir /path/to/hf_model \
  --save-dir /path/to/mcore_output \
  --target-tensor-parallel-size 2 \
  --target-pipeline-parallel-size 8 \
  --target-expert-parallel-size 64 \
  --save-workers 0   # 自动选择合适的并行度（默认）
```

### 串行模式（向后兼容）

```bash
python utils/convert_ckpt_hf2mcore.py \
  --load-dir /path/to/hf_model \
  --save-dir /path/to/mcore_output \
  --target-tensor-parallel-size 2 \
  --target-pipeline-parallel-size 8 \
  --target-expert-parallel-size 64 \
  --save-workers 1   # 串行保存（原始行为）
```

## 性能提升

对于典型的配置（TP=2, PP=8, EP=64）：
- 原始实现：需要串行保存 2×64=128 个文件，每个 PP rank
- 优化后：可以并行保存这 128 个文件，显著减少总保存时间

预计的性能提升取决于：
1. 存储系统的 I/O 性能
2. 设置的并行度
3. 文件系统的并发写入能力

## 实现细节

### 核心函数

1. `_save_single_rank_file` - 保存单个 rank 文件的线程安全函数
2. `_save_pp_rank` - 使用线程池并行保存所有 rank 文件的主要函数

### 错误处理

- 每个保存任务都有独立的错误处理
- 任何保存失败都会立即终止并报告具体错误
- 提供详细的错误信息（包括 pp_rank, tp_rank, ep_rank）

### 进度监控

- 定期输出保存进度（每完成10个任务）
- 最终输出总体统计信息（总文件数、耗时、文件/秒）

## 测试

可以使用配套的测试脚本验证优化效果：

```bash
python utils/test_parallel_saving.py
```

## 注意事项

1. **资源消耗**：增加并行度会增加 CPU 和 I/O 负载
2. **文件系统限制**：某些文件系统对并发写入有限制
3. **默认行为**：为了向后兼容，默认情况下不会启用并行保存（save-workers=0）
4. **最大线程数**：即使设置很高的并行度，也会被限制在 32 以内