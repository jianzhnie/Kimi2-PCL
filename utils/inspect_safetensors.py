#!/usr/bin/env python3
"""读取 HuggingFace safetensors 权重文件，按关键词搜索并打印权重信息。"""

import argparse
import json
import os

from safetensors.torch import load_file


def load_all_weights(model_path):
    """加载所有 safetensors 权重。"""
    index_path = os.path.join(model_path, 'model.safetensors.index.json')
    single_path = os.path.join(model_path, 'model.safetensors')

    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        files = sorted(set(index['weight_map'].values()))
    elif os.path.isfile(single_path):
        files = ['model.safetensors']
    else:
        import glob
        files = sorted(glob.glob(os.path.join(model_path, '*.safetensors')))
        files = [os.path.basename(f) for f in files]
        if not files:
            raise FileNotFoundError(f"找不到 safetensors 文件: {model_path}")

    all_weights = {}
    for fname in files:
        print(f"读取 {fname} ...")
        weights = load_file(os.path.join(model_path, fname))
        all_weights.update(weights)
    return all_weights


def print_weight(key, tensor, show_values=False):
    """打印单个权重的详细信息。"""
    size_mb = tensor.numel() * tensor.element_size() / 1024 / 1024
    print(f"  Key:    {key}")
    print(f"  Shape:  {tuple(tensor.shape)}")
    print(f"  Dtype:  {tensor.dtype}")
    print(f"  Size:   {size_mb:.2f} MB")
    if show_values:
        t = tensor.float()
        print(f"  Mean:   {t.mean().item():.6f}")
        print(f"  Std:    {t.std().item():.6f}")
        print(f"  Min:    {t.min().item():.6f}")
        print(f"  Max:    {t.max().item():.6f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='检查 HuggingFace safetensors 权重')
    parser.add_argument('model_path', help='HuggingFace 模型目录路径')
    parser.add_argument('--keyword',
                        '-k',
                        type=str,
                        default=None,
                        help='关键词过滤，仅打印包含该关键词的权重')
    parser.add_argument('--stats',
                        action='store_true',
                        help='显示权重的 mean/std/min/max 统计')
    args = parser.parse_args()

    all_weights = load_all_weights(args.model_path)

    if args.keyword:
        matched = {k: v for k, v in all_weights.items() if args.keyword in k}
        if not matched:
            print(f"未找到包含 '{args.keyword}' 的权重")
            return
        print(f"共 {len(matched)} 个匹配权重:\n")
        for key, tensor in matched.items():
            print_weight(key, tensor, args.stats)
    else:
        print(f"共 {len(all_weights)} 个权重:\n")
        for key, tensor in all_weights.items():
            print_weight(key, tensor, args.stats)


if __name__ == '__main__':
    main()
