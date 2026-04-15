#!/usr/bin/env python3
"""
验证 Mcore -> Huggingface 转换结果是否与 model_param_mapping.json 对齐。

用法:
    cd /Users/jianzhengnie/work_dir/Kimi2-PCL
    python verify_mcore2hf.py
"""

import json
import re
import sys

MCORE_JSON = 'model_param_mapping.json'
HF_JSON = 'model_param_hf.json'


def load_json(path):
    with open(path) as f:
        return json.load(f)


def _extract_config(mcore, hf):
    """从 JSON 数据中动态提取模型配置，不依赖硬编码常量。"""
    errors = []

    # hidden_size: 从 input_layernorm 获取
    hidden = None
    for k, v in mcore.items():
        if 'input_layernorm.weight' in k:
            hidden = v['shape'][0]
            break
    if hidden is None:
        errors.append('无法从 mcore JSON 中提取 hidden_size')
        hidden = 0

    # num_layers: 统计 mcore 中唯一的 layer 索引数
    layer_indices = set()
    for k in mcore:
        m = re.search(r'decoder\.layers\.(\d+)\.', k)
        if m:
            layer_indices.add(int(m.group(1)))
    num_layers = len(layer_indices)

    # first_k_dense: 第一个包含 experts.weight1 的层即为 MoE 起始层
    first_k_dense = num_layers
    for i in range(num_layers):
        if f"module.decoder.layers.{i}.mlp.experts.weight1" in mcore:
            first_k_dense = i
            break

    # num_experts: 从 HF expert keys 统计最大 expert id + 1
    num_experts = 0
    for k in hf:
        m = re.search(r'mlp\.experts\.(\d+)\.gate_proj\.weight', k)
        if m:
            num_experts = max(num_experts, int(m.group(1)) + 1)

    # dense_ffn: 从 dense 层的 gate_proj 获取
    dense_ffn = 0
    if first_k_dense > 0:
        k = 'model.layers.0.mlp.gate_proj.weight'
        if k in hf:
            dense_ffn = hf[k]['shape'][0]

    # moe_ffn: 从 MoE 层的 expert gate_proj 获取
    moe_ffn = 0
    if first_k_dense < num_layers:
        for e in range(num_experts):
            k = f"model.layers.{first_k_dense}.mlp.experts.{e}.gate_proj.weight"
            if k in hf:
                moe_ffn = hf[k]['shape'][0]
                break

    # inv_freq_len: 直接从第一层的 HF inv_freq 形状读取
    # (不从 k_proj 推断, 因为 num_kv_heads 未知导致无法唯一确定 head_dim)
    inv_freq_len = None
    inv_key = 'model.layers.0.self_attn.rotary_emb.inv_freq'
    if inv_key in hf:
        inv_freq_len = hf[inv_key]['shape'][0]

    return {
        'hidden': hidden,
        'num_layers': num_layers,
        'first_k_dense': first_k_dense,
        'num_experts': num_experts,
        'dense_ffn': dense_ffn,
        'moe_ffn': moe_ffn,
        'inv_freq_len': inv_freq_len,
    }, errors


def check_attention(mcore, hf, layer, inv_freq_len):
    """验证 Attention 子模块的所有参数映射与形状。"""
    errors = []
    prefix_m = f"module.decoder.layers.{layer}"
    prefix_h = f"model.layers.{layer}"

    # 1. LayerNorms
    in_ln_m = mcore[f"{prefix_m}.input_layernorm.weight"]['shape']
    in_ln_h = hf[f"{prefix_h}.input_layernorm.weight"]['shape']
    if in_ln_m != in_ln_h:
        errors.append(
            f"Layer {layer} input_layernorm mismatch: mcore={in_ln_m}, hf={in_ln_h}"
        )

    pre_mlp_m = mcore[f"{prefix_m}.pre_mlp_layernorm.weight"]['shape']
    post_attn_h = hf[f"{prefix_h}.post_attention_layernorm.weight"]['shape']
    if pre_mlp_m != post_attn_h:
        errors.append(
            f"Layer {layer} pre_mlp_layernorm mismatch: mcore={pre_mlp_m}, hf={post_attn_h}"
        )

    # 2. Q/K LayerNorm
    q_ln_m = mcore[f"{prefix_m}.self_attention.q_layernorm.weight"]['shape']
    q_ln_h = hf[f"{prefix_h}.self_attn.q_layernorm.weight"]['shape']
    if q_ln_m != q_ln_h:
        errors.append(
            f"Layer {layer} q_layernorm mismatch: mcore={q_ln_m}, hf={q_ln_h}")

    k_ln_m = mcore[f"{prefix_m}.self_attention.k_layernorm.weight"]['shape']
    k_ln_h = hf[f"{prefix_h}.self_attn.k_layernorm.weight"]['shape']
    if k_ln_m != k_ln_h:
        errors.append(
            f"Layer {layer} k_layernorm mismatch: mcore={k_ln_m}, hf={k_ln_h}")

    # 3. QKV 拆分验证
    qkv_shape = mcore[f"{prefix_m}.self_attention.linear_qkv.weight"]['shape']
    q_shape = hf[f"{prefix_h}.self_attn.q_proj.weight"]['shape']
    k_shape = hf[f"{prefix_h}.self_attn.k_proj.weight"]['shape']
    v_shape = hf[f"{prefix_h}.self_attn.v_proj.weight"]['shape']
    if qkv_shape[0] != q_shape[0] + k_shape[0] + v_shape[0]:
        errors.append(
            f"Layer {layer} QKV split mismatch: "
            f"mcore={qkv_shape}, q={q_shape}, k={k_shape}, v={v_shape}")

    # 4. Output projection
    proj_shape = mcore[f"{prefix_m}.self_attention.linear_proj.weight"][
        'shape']
    o_shape = hf[f"{prefix_h}.self_attn.o_proj.weight"]['shape']
    if proj_shape != o_shape:
        errors.append(
            f"Layer {layer} proj/o mismatch: mcore={proj_shape}, hf={o_shape}")

    # 5. Rotary embedding inv_freq
    rotary_key = f"{prefix_h}.self_attn.rotary_emb.inv_freq"
    if rotary_key not in hf:
        errors.append(f"Layer {layer} missing rotary_emb.inv_freq in HF")
    elif inv_freq_len is not None:
        rotary_shape = hf[rotary_key]['shape']
        if rotary_shape != [inv_freq_len]:
            errors.append(f"Layer {layer} rotary_emb.inv_freq mismatch: "
                          f"hf={rotary_shape}, expected [{inv_freq_len}]")

    return errors


def check_dense_layer(mcore, hf, layer, inv_freq_len):
    errors = []
    prefix_m = f"module.decoder.layers.{layer}"
    prefix_h = f"model.layers.{layer}"

    # Attention 子模块完整验证
    errors.extend(check_attention(mcore, hf, layer, inv_freq_len))

    # MLP: linear_fc1 -> gate_proj + up_proj
    fc1_shape = mcore[f"{prefix_m}.mlp.linear_fc1.weight"]['shape']
    gate_shape = hf[f"{prefix_h}.mlp.gate_proj.weight"]['shape']
    up_shape = hf[f"{prefix_h}.mlp.up_proj.weight"]['shape']
    if fc1_shape[0] != gate_shape[0] * 2 or gate_shape != up_shape:
        errors.append(
            f"Layer {layer} dense MLP fc1 split mismatch: "
            f"mcore_fc1={fc1_shape}, hf_gate={gate_shape}, hf_up={up_shape}")

    # MLP: linear_fc2 -> down_proj
    fc2_shape = mcore[f"{prefix_m}.mlp.linear_fc2.weight"]['shape']
    down_shape = hf[f"{prefix_h}.mlp.down_proj.weight"]['shape']
    if fc2_shape != down_shape:
        errors.append(f"Layer {layer} dense MLP fc2/down mismatch: "
                      f"mcore={fc2_shape}, hf={down_shape}")

    return errors


def check_moe_layer(mcore, hf, layer, num_experts, hidden, moe_ffn,
                    inv_freq_len):
    errors = []
    prefix_m = f"module.decoder.layers.{layer}"
    prefix_h = f"model.layers.{layer}"

    # Attention 子模块完整验证
    errors.extend(check_attention(mcore, hf, layer, inv_freq_len))

    # 1. Grouped GEMM weight1 / weight2 shapes
    w1_shape = mcore[f"{prefix_m}.mlp.experts.weight1"]['shape']
    w2_shape = mcore[f"{prefix_m}.mlp.experts.weight2"]['shape']
    expected_w1 = [hidden, num_experts * 2 * moe_ffn]
    expected_w2 = [num_experts * moe_ffn, hidden]
    if w1_shape != expected_w1:
        errors.append(
            f"Layer {layer} experts.weight1 shape unexpected: {w1_shape}, expected {expected_w1}"
        )
    if w2_shape != expected_w2:
        errors.append(
            f"Layer {layer} experts.weight2 shape unexpected: {w2_shape}, expected {expected_w2}"
        )

    # 2. Each expert gate/up/down shapes (不 break, 报告全部错误)
    for e in range(num_experts):
        gate = hf[f"{prefix_h}.mlp.experts.{e}.gate_proj.weight"]['shape']
        up = hf[f"{prefix_h}.mlp.experts.{e}.up_proj.weight"]['shape']
        down = hf[f"{prefix_h}.mlp.experts.{e}.down_proj.weight"]['shape']
        if gate != [moe_ffn, hidden] or up != [moe_ffn, hidden
                                               ] or down != [hidden, moe_ffn]:
            errors.append(f"Layer {layer} expert {e} shape mismatch: "
                          f"gate={gate}, up={up}, down={down}")

    # 3. Router weight
    router_shape = mcore[f"{prefix_m}.mlp.router.weight"]['shape']
    gate_shape = hf[f"{prefix_h}.mlp.gate.weight"]['shape']
    if router_shape != gate_shape:
        errors.append(
            f"Layer {layer} router/gate mismatch: mcore={router_shape}, hf={gate_shape}"
        )

    # 4. Router expert_bias -> e_score_correction_bias 形状验证
    bias_mcore_key = f"{prefix_m}.mlp.router.expert_bias"
    bias_hf_key = f"{prefix_h}.mlp.gate.e_score_correction_bias"
    if bias_mcore_key in mcore:
        if bias_hf_key not in hf:
            errors.append(
                f"Layer {layer} mcore has router.expert_bias but HF missing e_score_correction_bias"
            )
        else:
            bias_m = mcore[bias_mcore_key]['shape']
            bias_h = hf[bias_hf_key]['shape']
            if bias_m != bias_h:
                errors.append(
                    f"Layer {layer} e_score_correction_bias shape mismatch: "
                    f"mcore={bias_m}, hf={bias_h}")
    else:
        if bias_hf_key in hf:
            errors.append(
                f"Layer {layer} mcore missing router.expert_bias but HF has e_score_correction_bias"
            )

    # 5. Shared experts
    shared_fc1 = mcore[f"{prefix_m}.mlp.shared_experts.linear_fc1.weight"][
        'shape']
    shared_fc2 = mcore[f"{prefix_m}.mlp.shared_experts.linear_fc2.weight"][
        'shape']
    shared_gate = hf[f"{prefix_h}.mlp.shared_experts.gate_proj.weight"][
        'shape']
    shared_up = hf[f"{prefix_h}.mlp.shared_experts.up_proj.weight"]['shape']
    shared_down = hf[f"{prefix_h}.mlp.shared_experts.down_proj.weight"][
        'shape']
    if shared_fc1[0] != shared_gate[0] * 2:
        errors.append(
            f"Layer {layer} shared expert fc1 split mismatch: mcore={shared_fc1}, gate={shared_gate}"
        )
    if shared_gate != shared_up:
        errors.append(
            f"Layer {layer} shared expert gate/up mismatch: gate={shared_gate}, up={shared_up}"
        )
    if shared_fc2 != shared_down:
        errors.append(
            f"Layer {layer} shared expert fc2/down mismatch: mcore={shared_fc2}, hf={shared_down}"
        )

    return errors


def check_nonlayer(mcore, hf):
    errors = []
    if mcore['module.embedding.word_embeddings.weight']['shape'] != hf[
            'model.embed_tokens.weight']['shape']:
        errors.append('Embedding shape mismatch')
    if mcore['module.decoder.final_layernorm.weight']['shape'] != hf[
            'model.norm.weight']['shape']:
        errors.append('Final layernorm shape mismatch')
    if 'lm_head.weight' in hf:
        if mcore['module.output_layer.weight']['shape'] != hf[
                'lm_head.weight']['shape']:
            errors.append('LM head shape mismatch')
    else:
        # tie_word_embeddings=True: lm_head 不在 HF 中, 验证 output_layer == embed_tokens
        ol = mcore['module.output_layer.weight']['shape']
        et = hf['model.embed_tokens.weight']['shape']
        if ol != et:
            errors.append(
                f"tie_word_embeddings: output_layer {ol} != embed_tokens {et}")
    return errors


def check_key_completeness(mcore, hf, first_k_dense, num_layers, num_experts):
    """穷举式验证：确保每个 mcore key 都有对应 HF key，反之亦然，无遗漏。"""
    errors = []
    consumed_mcore = set()
    consumed_hf = set()

    for i in range(num_layers):
        prefix_m = f"module.decoder.layers.{i}"
        prefix_h = f"model.layers.{i}"

        # Attention keys (所有层通用)
        attn_mcore_keys = [
            f"{prefix_m}.input_layernorm.weight",
            f"{prefix_m}.pre_mlp_layernorm.weight",
            f"{prefix_m}.self_attention.linear_qkv.weight",
            f"{prefix_m}.self_attention.linear_proj.weight",
            f"{prefix_m}.self_attention.q_layernorm.weight",
            f"{prefix_m}.self_attention.k_layernorm.weight",
        ]
        attn_hf_keys = [
            f"{prefix_h}.input_layernorm.weight",
            f"{prefix_h}.post_attention_layernorm.weight",
            f"{prefix_h}.self_attn.q_proj.weight",
            f"{prefix_h}.self_attn.k_proj.weight",
            f"{prefix_h}.self_attn.v_proj.weight",
            f"{prefix_h}.self_attn.o_proj.weight",
            f"{prefix_h}.self_attn.q_layernorm.weight",
            f"{prefix_h}.self_attn.k_layernorm.weight",
            f"{prefix_h}.self_attn.rotary_emb.inv_freq",
        ]
        for k in attn_mcore_keys:
            if k not in mcore:
                errors.append(f"Missing mcore key: {k}")
            consumed_mcore.add(k)
        for k in attn_hf_keys:
            if k not in hf:
                errors.append(f"Missing HF key: {k}")
            consumed_hf.add(k)

        # MLP keys
        if i < first_k_dense:
            mlp_mcore = [
                f"{prefix_m}.mlp.linear_fc1.weight",
                f"{prefix_m}.mlp.linear_fc2.weight",
            ]
            mlp_hf = [
                f"{prefix_h}.mlp.gate_proj.weight",
                f"{prefix_h}.mlp.up_proj.weight",
                f"{prefix_h}.mlp.down_proj.weight",
            ]
        else:
            mlp_mcore = [
                f"{prefix_m}.mlp.experts.weight1",
                f"{prefix_m}.mlp.experts.weight2",
                f"{prefix_m}.mlp.router.weight",
                f"{prefix_m}.mlp.router.expert_bias",
                f"{prefix_m}.mlp.shared_experts.linear_fc1.weight",
                f"{prefix_m}.mlp.shared_experts.linear_fc2.weight",
            ]
            mlp_hf = [
                f"{prefix_h}.mlp.gate.weight",
                f"{prefix_h}.mlp.gate.e_score_correction_bias",
                f"{prefix_h}.mlp.shared_experts.gate_proj.weight",
                f"{prefix_h}.mlp.shared_experts.up_proj.weight",
                f"{prefix_h}.mlp.shared_experts.down_proj.weight",
            ]
            for e in range(num_experts):
                mlp_hf.extend([
                    f"{prefix_h}.mlp.experts.{e}.gate_proj.weight",
                    f"{prefix_h}.mlp.experts.{e}.up_proj.weight",
                    f"{prefix_h}.mlp.experts.{e}.down_proj.weight",
                ])

        for k in mlp_mcore:
            if k not in mcore:
                errors.append(f"Missing mcore key: {k}")
            consumed_mcore.add(k)
        for k in mlp_hf:
            if k not in hf:
                errors.append(f"Missing HF key: {k}")
            consumed_hf.add(k)

    # Non-layer keys
    nonlayer_mcore = [
        'module.embedding.word_embeddings.weight',
        'module.decoder.final_layernorm.weight',
        'module.output_layer.weight',
    ]
    nonlayer_hf = [
        'model.embed_tokens.weight',
        'model.norm.weight',
    ]
    for k in nonlayer_mcore:
        if k not in mcore:
            errors.append(f"Missing mcore key: {k}")
        consumed_mcore.add(k)
    for k in nonlayer_hf:
        if k not in hf:
            errors.append(f"Missing HF key: {k}")
        consumed_hf.add(k)
    # lm_head.weight (optional, depends on tie_word_embeddings)
    if 'lm_head.weight' in hf:
        consumed_hf.add('lm_head.weight')

    # 检查是否有未被映射的 key
    unmapped_mcore = set(mcore.keys()) - consumed_mcore
    unmapped_hf = set(hf.keys()) - consumed_hf
    for k in sorted(unmapped_mcore):
        errors.append(f"Unmapped mcore key: {k}")
    for k in sorted(unmapped_hf):
        errors.append(f"Unmapped HF key: {k}")

    if not errors:
        print(f"  Key completeness: all {len(mcore)} mcore keys and "
              f"{len(hf)} HF keys fully mapped")

    return errors


def main():
    print(f"Loading {MCORE_JSON} ...")
    mcore_data = load_json(MCORE_JSON)
    mcore = mcore_data['megatron_params']

    print(f"Loading {HF_JSON} ...")
    hf_data = load_json(HF_JSON)
    hf = hf_data['weight_map']

    print(f"\nMcore params: {len(mcore)}")
    print(f"HF params: {len(hf)}")

    # 动态提取配置
    cfg, cfg_errors = _extract_config(mcore, hf)
    if cfg_errors:
        print(f"Config extraction errors: {cfg_errors}")
        sys.exit(1)

    hidden = cfg['hidden']
    num_layers = cfg['num_layers']
    first_k_dense = cfg['first_k_dense']
    num_experts = cfg['num_experts']
    dense_ffn = cfg['dense_ffn']
    moe_ffn = cfg['moe_ffn']
    inv_freq_len = cfg['inv_freq_len']

    print('\nExtracted config:')
    print(f"  hidden={hidden}, num_layers={num_layers}, "
          f"first_k_dense={first_k_dense}")
    print(f"  num_experts={num_experts}, dense_ffn={dense_ffn}, "
          f"moe_ffn={moe_ffn}, inv_freq_len={inv_freq_len}")

    # 验证 key 数量
    # dense 层: 8 mcore keys / 12 hf keys
    # MoE 层:  12 mcore keys / (9 attn + 5 gate+shared + 3*num_experts) hf keys
    # 非层:    3 mcore keys / 2~3 hf keys
    expected_mcore = first_k_dense * 8 + (num_layers - first_k_dense) * 12 + 3
    hf_per_moe_layer = 9 + 5 + 3 * num_experts
    expected_hf_min = first_k_dense * 12 + (
        num_layers - first_k_dense) * hf_per_moe_layer + 2
    expected_hf_max = expected_hf_min + 1  # lm_head optional

    print(f"\n  Expected mcore keys: {expected_mcore} (actual: {len(mcore)})")
    print(f"  Expected HF keys: {expected_hf_min}~{expected_hf_max} "
          f"(actual: {len(hf)})")

    all_errors = []

    # Key count sanity check
    if len(mcore) != expected_mcore:
        all_errors.append(
            f"Mcore key count mismatch: expected {expected_mcore}, "
            f"got {len(mcore)}")
    if len(hf) < expected_hf_min or len(hf) > expected_hf_max:
        all_errors.append(f"HF key count mismatch: expected {expected_hf_min}~"
                          f"{expected_hf_max}, got {len(hf)}")

    # 1. 穷举式 key 完整性检查
    print('\n--- Key completeness check ---')
    all_errors.extend(
        check_key_completeness(mcore, hf, first_k_dense, num_layers,
                               num_experts))

    # 2. Dense layers shape check
    print('--- Dense layer shape check ---')
    for layer in range(first_k_dense):
        all_errors.extend(check_dense_layer(mcore, hf, layer, inv_freq_len))
    print(f"  Checked {first_k_dense} dense layers")

    # 3. MoE layers shape check
    print('--- MoE layer shape check ---')
    for layer in range(first_k_dense, num_layers):
        all_errors.extend(
            check_moe_layer(mcore, hf, layer, num_experts, hidden, moe_ffn,
                            inv_freq_len))
    print(f"  Checked {num_layers - first_k_dense} MoE layers")

    # 4. Non-layer check
    print('--- Non-layer check ---')
    all_errors.extend(check_nonlayer(mcore, hf))

    # 5. Expert count sanity check
    experts_count = {}
    for k in hf:
        if '.mlp.experts.' in k:
            layer = int(k.split('model.layers.')[1].split('.')[0])
            experts_count.setdefault(layer, set())
            expert_id = int(k.split('.mlp.experts.')[1].split('.')[0])
            experts_count[layer].add(expert_id)

    moe_layers_with_full = sum(1 for ids in experts_count.values()
                               if len(ids) == num_experts)
    expected_moe_layers = num_layers - first_k_dense
    print(
        f"\nMoE layers with {num_experts} experts: {moe_layers_with_full} (expected {expected_moe_layers})"
    )
    if moe_layers_with_full != expected_moe_layers:
        all_errors.append(
            f"Expert count mismatch: {moe_layers_with_full} layers have "
            f"{num_experts} experts")

    print()
    if all_errors:
        print(f"\nFAILED - {len(all_errors)} error(s) found:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print('PASSED - All shape and key checks aligned correctly.')
        sys.exit(0)


if __name__ == '__main__':
    main()
