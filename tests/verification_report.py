#!/usr/bin/env python3
"""
Kimi2-PCL 模型一致性验证与权重转换检查报告
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# 导入项目的配置解析工具
import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils.pretrain_config import parse_pretrain_script, get_int, get_float, get_bool


@dataclass
class VerificationResult:
    category: str
    item: str
    status: str  # PASS, FAIL, WARNING, INFO
    message: str
    details: Optional[Dict] = None


class ModelConsistencyVerifier:
    """模型一致性验证器"""
    
    def __init__(self):
        self.results: List[VerificationResult] = []
        
    def add_result(self, category: str, item: str, status: str, message: str, details=None):
        self.results.append(VerificationResult(category, item, status, message, details))
        
    def load_pretrain_script(self, path: str) -> Dict:
        """加载预训练脚本配置"""
        return parse_pretrain_script(path)
    
    def load_json_config(self, path: str) -> Dict:
        """加载JSON配置文件"""
        with open(path) as f:
            return json.load(f)
    
    def verify_architecture_config_consistency(self):
        """验证架构与配置的一致性"""
        print("=" * 80)
        print("1. 模型架构与配置文件一致性验证")
        print("=" * 80)
        
        # 加载配置
        script_cfg = self.load_pretrain_script("scripts/pretrain_kimi2_1t_4k.sh")
        json_cfg = self.load_json_config("models/config_1t.json")
        
        # 从预训练脚本提取关键参数
        script_params = {
            'num_layers': get_int(script_cfg, '--num-layers'),
            'hidden_size': get_int(script_cfg, '--hidden-size'),
            'ffn_hidden_size': get_int(script_cfg, '--ffn-hidden-size'),
            'num_attention_heads': get_int(script_cfg, '--num-attention-heads'),
            'num_query_groups': get_int(script_cfg, '--num-query-groups'),
            'vocab_size': get_int(script_cfg, '--vocab-size'),
            'max_position_embeddings': get_int(script_cfg, '--max-position-embeddings'),
            'seq_length': get_int(script_cfg, '--seq-length'),
            'rotary_base': get_float(script_cfg, '--rotary-base'),
            'tp_size': get_int(script_cfg, '--tensor-model-parallel-size'),
            'pp_size': get_int(script_cfg, '--pipeline-model-parallel-size'),
            'ep_size': get_int(script_cfg, '--expert-model-parallel-size'),
            'num_experts': get_int(script_cfg, '--num-experts'),
            'moe_ffn_hidden_size': get_int(script_cfg, '--moe-ffn-hidden-size'),
            'moe_router_topk': get_int(script_cfg, '--moe-router-topk'),
            'moe_aux_loss_coeff': get_float(script_cfg, '--moe-aux-loss-coeff'),
            'first_k_dense_replace': get_int(script_cfg, '--first-k-dense-replace'),
            'qk_layernorm': get_bool(script_cfg, '--qk-layernorm'),
            'rope_scaling_factor': get_float(script_cfg, '--rope-scaling-factor'),
            'rope_scaling_type': script_cfg.flags.get('--rope-scaling-type'),
        }
        
        # 映射JSON配置参数
        json_params = {
            'num_layers': json_cfg.get('num_hidden_layers'),
            'hidden_size': json_cfg.get('hidden_size'),
            'ffn_hidden_size': json_cfg.get('intermediate_size'),
            'num_attention_heads': json_cfg.get('num_attention_heads'),
            'num_query_groups': json_cfg.get('num_query_groups'),
            'vocab_size': json_cfg.get('vocab_size'),
            'max_position_embeddings': json_cfg.get('max_position_embeddings'),
            'rotary_base': json_cfg.get('rope_theta'),
            'num_experts': json_cfg.get('n_routed_experts'),
            'moe_ffn_hidden_size': json_cfg.get('moe_intermediate_size'),
            'moe_router_topk': json_cfg.get('moe_router_topk'),
            'moe_aux_loss_coeff': json_cfg.get('moe_aux_loss_coeff'),
            'first_k_dense_replace': json_cfg.get('first_k_dense_replace'),
            'qk_layernorm': json_cfg.get('qk_layernorm'),
            'qk_nope_head_dim': json_cfg.get('qk_nope_head_dim'),
            'qk_rope_head_dim': json_cfg.get('qk_rope_head_dim'),
            'v_head_dim': json_cfg.get('v_head_dim'),
            'num_key_value_heads': json_cfg.get('num_key_value_heads'),
        }
        
        # 验证关键参数一致性
        param_mappings = [
            ('num_layers', 'num_layers', '模型层数'),
            ('hidden_size', 'hidden_size', '隐藏层维度'),
            ('ffn_hidden_size', 'ffn_hidden_size', 'FFN隐藏层维度'),
            ('num_attention_heads', 'num_attention_heads', '注意力头数'),
            ('num_query_groups', 'num_query_groups', '查询组数'),
            ('vocab_size', 'vocab_size', '词表大小'),
            ('max_position_embeddings', 'max_position_embeddings', '最大位置编码'),
            ('rotary_base', 'rotary_base', 'RoPE基数'),
            ('num_experts', 'num_experts', '专家数量'),
            ('moe_ffn_hidden_size', 'moe_ffn_hidden_size', 'MoE FFN维度'),
            ('moe_router_topk', 'moe_router_topk', '路由Top-K'),
            ('moe_aux_loss_coeff', 'moe_aux_loss_coeff', '辅助损失系数'),
            ('first_k_dense_replace', 'first_k_dense_replace', '前K层Dense替换'),
        ]
        
        all_pass = True
        for script_key, json_key, desc in param_mappings:
            s_val = script_params.get(script_key)
            j_val = json_params.get(json_key)
            
            if s_val is None or j_val is None:
                status = "WARNING"
                message = f"{desc}: 无法比较 (脚本={s_val}, JSON={j_val})"
            elif s_val == j_val:
                status = "PASS"
                message = f"{desc}: {s_val} ✓"
            else:
                status = "FAIL"
                message = f"{desc}: 不匹配! 脚本={s_val}, JSON={j_val}"
                all_pass = False
            
            self.add_result("架构配置一致性", desc, status, message, 
                          {'script_value': s_val, 'json_value': j_val})
        
        # 验证GQA配置
        expected_kv_heads = script_params['num_attention_heads'] // script_params['num_query_groups']
        actual_kv_heads = json_params.get('num_key_value_heads')
        
        if expected_kv_heads == actual_kv_heads:
            status = "PASS"
            message = f"GQA KV头数: {actual_kv_heads} (预期={expected_kv_heads}) ✓"
        else:
            status = "FAIL"
            message = f"GQA KV头数不匹配: 实际={actual_kv_heads}, 预期={expected_kv_heads}"
            all_pass = False
        
        self.add_result("架构配置一致性", "GQA KV头数", status, message,
                       {'expected': expected_kv_heads, 'actual': actual_kv_heads})
        
        # 验证RoPE配置
        rope_cfg = json_cfg.get('rope_scaling', {})
        if rope_cfg.get('type') == 'yarn' and rope_cfg.get('factor') == 32.0:
            status = "PASS"
            message = f"RoPE配置: type={rope_cfg.get('type')}, factor={rope_cfg.get('factor')} ✓"
        else:
            status = "WARNING"
            message = f"RoPE配置可能不匹配: {rope_cfg}"
        
        self.add_result("架构配置一致性", "RoPE配置", status, message, rope_cfg)
        
        # 验证并行度配置
        parallel_config = {
            'TP (张量并行)': script_params['tp_size'],
            'PP (流水线并行)': script_params['pp_size'],
            'EP (专家并行)': script_params['ep_size'],
        }
        
        # 验证专家并行度与专家数量的整除关系
        if script_params['num_experts'] % script_params['ep_size'] == 0:
            status = "PASS"
            message = f"专家数({script_params['num_experts']})能被EP({script_params['ep_size']})整除 ✓"
        else:
            status = "FAIL"
            message = f"专家数({script_params['num_experts']})不能被EP({script_params['ep_size']})整除"
            all_pass = False
        
        self.add_result("并行度配置", "EP整除性", status, message, parallel_config)
        
        # 验证序列长度配置
        if script_params['seq_length'] == 4096:
            status = "PASS"
            message = f"训练序列长度: {script_params['seq_length']} ✓"
        else:
            status = "WARNING"
            message = f"训练序列长度非4K: {script_params['seq_length']}"
        
        self.add_result("序列长度配置", "训练序列长度", status, message)
        
        return all_pass
    
    def verify_weight_conversion_code(self):
        """验证权重转换代码正确性"""
        print("\n" + "=" * 80)
        print("2. 权重转换代码正确性验证")
        print("=" * 80)
        
        all_pass = True
        
        # 验证 convert_ckpt_mcore2hf.py
        try:
            with open("utils/convert_ckpt_mcore2hf.py") as f:
                mcore2hf_code = f.read()
            
            checks = [
                ("MgCkptConvert类定义", r"class MgCkptConvert"),
                ("QKV分离逻辑", r"q_parts\.append|k_parts\.append|v_parts\.append"),
                ("TP合并逻辑", r"torch\.cat.*dim=0|torch\.cat.*dim=1"),
                ("专家权重重建", r"_reconstruct_router|experts\.weight"),
                ("配置写入", r"_write_hf_artifacts|config\.json"),
                ("Safetensors保存", r"safetensors\.torch\.save_file"),
                ("SHA256校验", r"_sha256_file|_write_sha256_manifest"),
            ]
            
            for desc, pattern in checks:
                if re.search(pattern, mcore2hf_code):
                    status = "PASS"
                    message = f"mcore2hf: {desc} 已实现"
                else:
                    status = "WARNING"
                    message = f"mcore2hf: {desc} 可能缺失"
                self.add_result("权重转换代码", f"mcore2hf-{desc}", status, message)
                
        except Exception as e:
            self.add_result("权重转换代码", "mcore2hf加载", "FAIL", str(e))
            all_pass = False
        
        # 验证 convert_ckpt_hf2mcore.py
        try:
            with open("utils/convert_ckpt_hf2mcore.py") as f:
                hf2mcore_code = f.read()
            
            checks = [
                ("CkptConvert类定义", r"class CkptConvert"),
                ("QKV合并逻辑", r"torch\.cat.*q_tp|torch\.cat.*k_tp|torch\.cat.*v_tp"),
                ("TP分片逻辑", r"torch\.chunk.*tp_size"),
                ("专家权重分配", r"experts_linear_fc1|experts_linear_fc2"),
                ("Grouped GEMM支持", r"moe_grouped_gemm|experts\.weight1"),
                ("VPP/DualPipe支持", r"vpp_rank|dualpipe"),
                ("NF4量化支持", r"qlora_nf4|bitsandbytes"),
            ]
            
            for desc, pattern in checks:
                if re.search(pattern, hf2mcore_code):
                    status = "PASS"
                    message = f"hf2mcore: {desc} 已实现"
                else:
                    status = "WARNING"
                    message = f"hf2mcore: {desc} 可能缺失"
                self.add_result("权重转换代码", f"hf2mcore-{desc}", status, message)
                
        except Exception as e:
            self.add_result("权重转换代码", "hf2mcore加载", "FAIL", str(e))
            all_pass = False
        
        # 验证关键常量定义
        expected_constants = {
            'HIDDEN_SIZE': 7168,
            'NUM_EXPERTS': 128,
            'NUM_ATTENTION_HEADS': 64,
            'QK_HEAD_DIM': 128,
            'QK_POS_EMB_HEAD_DIM': 64,
            'V_HEAD_DIM': 128,
            'NUM_QUERY_GROUPS': 32,
            'FFN_HIDDEN_SIZE': 18432,
            'MOE_FFN_HIDDEN_SIZE': 12288,
            'VOCAB_SIZE': 163840,
        }
        
        for const_name, expected_val in expected_constants.items():
            pattern = rf"{const_name}\s*=\s*{expected_val}"
            in_mcore2hf = bool(re.search(pattern, mcore2hf_code))
            in_hf2mcore = bool(re.search(pattern, hf2mcore_code))
            
            if in_mcore2hf and in_hf2mcore:
                status = "PASS"
                message = f"常量 {const_name}={expected_val} 在两个文件中一致定义 ✓"
            else:
                status = "WARNING"
                message = f"常量 {const_name} 定义不一致或缺失 (mcore2hf={in_mcore2hf}, hf2mcore={in_hf2mcore})"
            
            self.add_result("权重转换常量", const_name, status, message,
                          {'expected': expected_val, 'in_mcore2hf': in_mcore2hf, 'in_hf2mcore': in_hf2mcore})
        
        return all_pass
    
    def verify_shell_scripts(self):
        """验证Shell转换脚本"""
        print("\n" + "=" * 80)
        print("3. Shell转换脚本功能验证")
        print("=" * 80)
        
        all_pass = True
        
        scripts = [
            ("scripts/ckpt_convert_mcore2hf.sh", "MCore→HF转换"),
            ("scripts/ckpt_convert_hf2mcore.sh", "HF→MCore转换"),
        ]
        
        for script_path, desc in scripts:
            try:
                with open(script_path) as f:
                    content = f.read()
                
                checks = [
                    ("环境变量设置", r"export\s+\w+="),
                    ("路径检查", r"if\s+\[\[.*-d.*\]\]"),
                    ("参数解析", r"\$\{[A-Z_]+:-"),
                    ("并行度配置", r"TP=|PP=|EP="),
                    ("Python调用", r"python.*convert_ckpt"),
                    ("错误处理", r"set\s+-euo\s+pipefail"),
                ]
                
                for check_desc, pattern in checks:
                    if re.search(pattern, content):
                        status = "PASS"
                        message = f"{desc}: {check_desc} 已配置"
                    else:
                        status = "WARNING"
                        message = f"{desc}: {check_desc} 可能缺失"
                    self.add_result("Shell脚本检查", f"{script_path}-{check_desc}", status, message)
                
                # 验证关键参数一致性
                key_params = ['TP', 'PP', 'EP', 'NUM_LAYERS', 'HIDDEN_SIZE', 'NUM_EXPERTS']
                for param in key_params:
                    pattern = rf"{param}="
                    if re.search(pattern, content):
                        status = "INFO"
                        message = f"{desc}: 参数 {param} 已定义"
                    else:
                        status = "WARNING"
                        message = f"{desc}: 参数 {param} 未定义"
                    self.add_result("Shell脚本参数", f"{script_path}-{param}", status, message)
                        
            except Exception as e:
                self.add_result("Shell脚本检查", script_path, "FAIL", str(e))
                all_pass = False
        
        # 比较两个脚本的参数一致性
        try:
            with open("scripts/ckpt_convert_mcore2hf.sh") as f:
                mcore2hf_script = f.read()
            with open("scripts/ckpt_convert_hf2mcore.sh") as f:
                hf2mcore_script = f.read()
            
            params_to_compare = [
                ('TP', r'TP="?\$\{TP:-(\d+)\}"?'),
                ('PP', r'PP="?\$\{PP:-(\d+)\}"?'),
                ('EP', r'EP="?\$\{EP:-(\d+)\}"?'),
                ('NUM_LAYERS', r'NUM_LAYERS="?\$\{NUM_LAYERS:-(\d+)\}"?'),
                ('HIDDEN_SIZE', r'HIDDEN_SIZE="?\$\{HIDDEN_SIZE:-(\d+)\}"?'),
            ]
            
            for param_name, pattern in params_to_compare:
                m_val = re.search(pattern, mcore2hf_script)
                h_val = re.search(pattern, hf2mcore_script)
                
                if m_val and h_val:
                    m_num = m_val.group(1)
                    h_num = h_val.group(1)
                    if m_num == h_num:
                        status = "PASS"
                        message = f"参数 {param_name}={m_num} 在两个脚本中一致 ✓"
                    else:
                        status = "FAIL"
                        message = f"参数 {param_name} 不一致: mcore2hf={m_num}, hf2mcore={h_num}"
                        all_pass = False
                else:
                    status = "WARNING"
                    message = f"参数 {param_name} 无法比较"
                
                self.add_result("脚本参数一致性", param_name, status, message)
                
        except Exception as e:
            self.add_result("脚本参数一致性", "比较失败", "FAIL", str(e))
            all_pass = False
        
        return all_pass
    
    def verify_weight_check_tool(self):
        """验证权重一致性检查工具"""
        print("\n" + "=" * 80)
        print("4. 权重一致性检查工具验证")
        print("=" * 80)
        
        all_pass = True
        
        try:
            with open("utils/check_model_weights.py") as f:
                check_code = f.read()
            
            checks = [
                ("Safetensors读取", r"safe_open|_shard_paths"),
                ("模型构建", r"init_empty_weights|DeepseekV3ForCausalLM"),
                ("配置加载", r"DeepseekV3Config|from_pretrained"),
                ("键名对比", r"missing_in_ckpt|extra_in_ckpt"),
                ("形状对比", r"_compare_shapes|shape mismatch"),
                ("Head维度诊断", r"HEAD DIM DIAGNOSTICS|q_proj_out"),
                ("索引验证", r"weight_map|strict_index"),
            ]
            
            for desc, pattern in checks:
                if re.search(pattern, check_code):
                    status = "PASS"
                    message = f"check_model_weights: {desc} 已实现"
                else:
                    status = "WARNING"
                    message = f"check_model_weights: {desc} 可能缺失"
                self.add_result("权重检查工具", desc, status, message)
            
            # 验证测试文件
            with open("tests/test_config_and_conversion.py") as f:
                test_code = f.read()
            
            test_checks = [
                ("配置测试", r"test_config_default_q_head_dims"),
                ("QKV布局推断测试", r"test_infer_qkv_layout"),
                ("mcore2hf注意力转换测试", r"test_mcore2hf_set_layer_attn_shapes"),
                ("hf2mcore注意力转换测试", r"test_hf2mcore_set_layer_attn_shapes"),
            ]
            
            for desc, pattern in test_checks:
                if re.search(pattern, test_code):
                    status = "PASS"
                    message = f"单元测试: {desc} 已存在"
                else:
                    status = "WARNING"
                    message = f"单元测试: {desc} 可能缺失"
                self.add_result("单元测试", desc, status, message)
                
        except Exception as e:
            self.add_result("权重检查工具", "加载失败", "FAIL", str(e))
            all_pass = False
        
        return all_pass
    
    def generate_report(self):
        """生成验证报告"""
        print("\n" + "=" * 80)
        print("5. 综合验证报告")
        print("=" * 80)
        
        # 统计结果
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {'PASS': 0, 'FAIL': 0, 'WARNING': 0, 'INFO': 0}
            categories[r.category][r.status] += 1
        
        print("\n分类统计:")
        print("-" * 80)
        for cat, stats in sorted(categories.items()):
            total = sum(stats.values())
            pass_rate = stats['PASS'] / total * 100 if total > 0 else 0
            print(f"  {cat:40s}: PASS={stats['PASS']}, FAIL={stats['FAIL']}, WARN={stats['WARNING']}, INFO={stats['INFO']} (通过率: {pass_rate:.1f}%)")
        
        print("\n详细结果:")
        print("-" * 80)
        current_category = None
        for r in self.results:
            if r.category != current_category:
                current_category = r.category
                print(f"\n【{current_category}】")
            status_symbol = {"PASS": "✓", "FAIL": "✗", "WARNING": "⚠", "INFO": "ℹ"}.get(r.status, "?")
            print(f"  [{status_symbol}] {r.item:50s}: {r.message}")
        
        # 失败项汇总
        failures = [r for r in self.results if r.status == "FAIL"]
        if failures:
            print("\n" + "=" * 80)
            print("失败项汇总 (需要修复):")
            print("=" * 80)
            for r in failures:
                print(f"  - [{r.category}] {r.item}: {r.message}")
        
        warnings = [r for r in self.results if r.status == "WARNING"]
        if warnings:
            print("\n" + "=" * 80)
            print("警告项汇总 (建议检查):")
            print("=" * 80)
            for r in warnings[:10]:  # 只显示前10个警告
                print(f"  - [{r.category}] {r.item}: {r.message}")
            if len(warnings) > 10:
                print(f"  ... 还有 {len(warnings) - 10} 个警告")
        
        # 总体评估
        total_checks = len(self.results)
        pass_count = sum(1 for r in self.results if r.status == "PASS")
        fail_count = len(failures)
        warning_count = len(warnings)
        
        print("\n" + "=" * 80)
        print("总体评估:")
        print("=" * 80)
        print(f"  总检查项: {total_checks}")
        print(f"  通过: {pass_count} ({pass_count/total_checks*100:.1f}%)")
        print(f"  失败: {fail_count}")
        print(f"  警告: {warning_count}")
        
        if fail_count == 0:
            print("\n  ★ 所有关键检查项通过，模型配置一致！")
        elif fail_count <= 3:
            print("\n  ▲ 存在少量问题，建议修复后再进行训练/转换")
        else:
            print("\n  ✗ 存在较多问题，强烈建议修复后再进行训练/转换")
        
        return fail_count == 0


def main():
    verifier = ModelConsistencyVerifier()
    
    # 运行所有验证
    r1 = verifier.verify_architecture_config_consistency()
    r2 = verifier.verify_weight_conversion_code()
    r3 = verifier.verify_shell_scripts()
    r4 = verifier.verify_weight_check_tool()
    
    # 生成报告
    final_pass = verifier.generate_report()
    
    # 保存详细报告到文件
    report_path = "verification_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Kimi2-PCL 模型一致性验证与权重转换检查详细报告\n")
        f.write("=" * 80 + "\n\n")
        
        current_category = None
        for r in verifier.results:
            if r.category != current_category:
                current_category = r.category
                f.write(f"\n【{current_category}】\n")
                f.write("-" * 80 + "\n")
            f.write(f"[{r.status}] {r.item}\n")
            f.write(f"  消息: {r.message}\n")
            if r.details:
                f.write(f"  详情: {r.details}\n")
            f.write("\n")
    
    print(f"\n详细报告已保存至: {report_path}")
    
    return 0 if final_pass else 1


if __name__ == "__main__":
    sys.exit(main())
