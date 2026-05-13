import json
import shutil
import unittest
from pathlib import Path
import sys
import torch
from safetensors.torch import save_file

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.verify_expanded_weights import ModelWeightLoader, verify_layers, verify_experts

class TestWeightVerification(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(__file__).parent / "tmp_verify_test"
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True)
        
        self.orig_dir = self.test_dir / "orig"
        self.exp_dir = self.test_dir / "exp"
        self.orig_dir.mkdir()
        self.exp_dir.mkdir()

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_verify_layers_success(self):
        # Create dummy original model
        orig_weights = {
            "model.layers.0.w": torch.randn(10, 10),
            "model.layers.1.w": torch.randn(10, 10),
            "model.embed.w": torch.randn(10, 10)
        }
        save_file(orig_weights, str(self.orig_dir / "model.safetensors"))
        with open(self.orig_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {k: "model.safetensors" for k in orig_weights}}, f)
        
        # Create dummy expanded model (2 -> 4 layers, sequential)
        exp_weights = {
            "model.layers.0.w": orig_weights["model.layers.0.w"],
            "model.layers.1.w": orig_weights["model.layers.1.w"],
            "model.layers.2.w": orig_weights["model.layers.0.w"].clone(), # copy 0
            "model.layers.3.w": orig_weights["model.layers.1.w"].clone(), # copy 1
            "model.embed.w": orig_weights["model.embed.w"]
        }
        save_file(exp_weights, str(self.exp_dir / "model.safetensors"))
        with open(self.exp_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {k: "model.safetensors" for k in exp_weights}}, f)
            
        orig_loader = ModelWeightLoader(self.orig_dir)
        exp_loader = ModelWeightLoader(self.exp_dir)
        
        mismatches = verify_layers(orig_loader, exp_loader, 2, 4, "seq")
        self.assertEqual(len(mismatches), 0)

    def test_verify_layers_mismatch(self):
        # Create dummy original model
        orig_weights = {"model.layers.0.w": torch.ones(2, 2)}
        save_file(orig_weights, str(self.orig_dir / "model.safetensors"))
        with open(self.orig_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {k: "model.safetensors" for k in orig_weights}}, f)
        
        # Create dummy expanded model with a mismatch
        exp_weights = {
            "model.layers.0.w": torch.ones(2, 2),
            "model.layers.1.w": torch.zeros(2, 2) # should be ones (copy of 0)
        }
        save_file(exp_weights, str(self.exp_dir / "model.safetensors"))
        with open(self.exp_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {k: "model.safetensors" for k in exp_weights}}, f)
            
        orig_loader = ModelWeightLoader(self.orig_dir)
        exp_loader = ModelWeightLoader(self.exp_dir)
        
        mismatches = verify_layers(orig_loader, exp_loader, 1, 2, "0")
        self.assertGreater(len(mismatches), 0)
        self.assertIn("Value mismatch: model.layers.1.w", mismatches[0])

    def test_verify_experts_success(self):
        # Create dummy original MoE model
        orig_config = {"n_routed_experts": 2, "zero_expert_num": 1}
        with open(self.orig_dir / "config.json", "w") as f: json.dump(orig_config, f)
        
        orig_weights = {
            "model.layers.0.mlp.experts.0.w": torch.randn(2, 2),
            "model.layers.0.mlp.experts.1.w": torch.randn(2, 2),
            "model.layers.0.mlp.experts.2.w": torch.randn(2, 2), # zero-shot
            "model.layers.0.mlp.router.classifier.weight": torch.randn(3, 10)
        }
        save_file(orig_weights, str(self.orig_dir / "model.safetensors"))
        with open(self.orig_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {k: "model.safetensors" for k in orig_weights}}, f)
            
        # Create expanded MoE model (2 -> 4 routed experts)
        exp_config = {"n_routed_experts": 4, "zero_expert_num": 1}
        with open(self.exp_dir / "config.json", "w") as f: json.dump(exp_config, f)
        
        exp_weights = {
            # Experts
            "model.layers.0.mlp.experts.0.w": orig_weights["model.layers.0.mlp.experts.0.w"],
            "model.layers.0.mlp.experts.1.w": orig_weights["model.layers.0.mlp.experts.1.w"],
            "model.layers.0.mlp.experts.2.w": orig_weights["model.layers.0.mlp.experts.0.w"].clone(), # copy 0
            "model.layers.0.mlp.experts.3.w": orig_weights["model.layers.0.mlp.experts.1.w"].clone(), # copy 1
            "model.layers.0.mlp.experts.4.w": orig_weights["model.layers.0.mlp.experts.2.w"].clone(), # zero-shot shifted
            # Router
            "model.layers.0.mlp.router.classifier.weight": torch.cat([
                orig_weights["model.layers.0.mlp.router.classifier.weight"][:2], # real
                orig_weights["model.layers.0.mlp.router.classifier.weight"][:2], # real copy
                orig_weights["model.layers.0.mlp.router.classifier.weight"][2:], # zero
            ], dim=0)
        }
        save_file(exp_weights, str(self.exp_dir / "model.safetensors"))
        with open(self.exp_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"weight_map": {k: "model.safetensors" for k in exp_weights}}, f)
            
        orig_loader = ModelWeightLoader(self.orig_dir)
        exp_loader = ModelWeightLoader(self.exp_dir)
        
        mismatches = verify_experts(orig_loader, exp_loader)
        self.assertEqual(len(mismatches), 0)

if __name__ == "__main__":
    unittest.main()
