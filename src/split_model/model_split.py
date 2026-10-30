import re
import logging
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file


LLM = "granite-3.1-1b-a400m-instruct"

# Configure basic logging for pipeline visibility
logging.basicConfig(level=logging.INFO, format="%(message)s")

class MoEExtractionPipeline:
    def __init__(self, input_file: str, output_dir: str):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_num_layers(self, f) -> int:
        """Dynamically determines the number of layers in the model."""
        layer_indices = []
        for key in f.keys():
            match = re.search(r"model\.layers\.(\d+)\.", key)
            if match:
                layer_indices.append(int(match.group(1)))
        
        if not layer_indices:
            raise ValueError("Could not find any layers matching 'model.layers.XX' pattern.")
            
        return max(layer_indices) + 1

    def extract_embeddings(self, f):
        """Extracts the initial token embeddings."""
        key = "model.embed_tokens.weight"
        if key in f.keys():
            logging.info("Extracting: Embeddings")
            save_file(
                {"weight": f.get_tensor(key)}, 
                self.output_dir / "00_embeddings.safetensors"
            )

    def extract_final_norms(self, f):
        """Extracts the final layer norm and LM head (if untied)."""
        logging.info("Extracting: Final Output Block")
        final_tensors = {}
        
        if "model.norm.weight" in f.keys():
            final_tensors["norm.weight"] = f.get_tensor("model.norm.weight")
        if "lm_head.weight" in f.keys():
            final_tensors["lm_head.weight"] = f.get_tensor("lm_head.weight")
            
        if final_tensors:
            save_file(final_tensors, self.output_dir / "99_final_token_picker.safetensors")

    def extract_layer(self, f, layer_idx: int):
        """Extracts attention, router, and un-batches the experts for a single layer."""
        layer_prefix = f"model.layers.{layer_idx}."
        
        # 1. Attention & Norms
        attn_keys = [
            f"{layer_prefix}input_layernorm.weight",
            f"{layer_prefix}post_attention_layernorm.weight",
            f"{layer_prefix}self_attn.q_proj.weight",
            f"{layer_prefix}self_attn.k_proj.weight",
            f"{layer_prefix}self_attn.v_proj.weight",
            f"{layer_prefix}self_attn.o_proj.weight",
        ]
        attn_dict = {
            k.replace(layer_prefix, ""): f.get_tensor(k) 
            for k in attn_keys if k in f.keys()
        }
        if attn_dict:
            save_file(attn_dict, self.output_dir / f"layer_{layer_idx:02d}_attention.safetensors")

        # 2. Router / Gate
        router_key = f"{layer_prefix}block_sparse_moe.router.layer.weight"
        if router_key in f.keys():
            save_file(
                {"weight": f.get_tensor(router_key)}, 
                self.output_dir / f"layer_{layer_idx:02d}_router.safetensors"
            )

        # 3. Un-batching Experts
        in_key = f"{layer_prefix}block_sparse_moe.input_linear.weight"
        out_key = f"{layer_prefix}block_sparse_moe.output_linear.weight"
        
        if in_key in f.keys() and out_key in f.keys():
            in_batched = f.get_tensor(in_key)
            out_batched = f.get_tensor(out_key)
            
            num_experts = in_batched.shape[0]
            
            for expert_id in range(num_experts):
                expert_dict = {
                    "input_linear.weight": in_batched[expert_id].contiguous(),
                    "output_linear.weight": out_batched[expert_id].contiguous()
                }
                save_file(
                    expert_dict, 
                    self.output_dir / f"layer_{layer_idx:02d}_expert_{expert_id:02d}.safetensors"
                )

    def run(self):
        """Executes the full pipeline."""
        logging.info(f"🚀 Starting extraction from {self.input_file}")
        
        with safe_open(self.input_file, framework="pt", device="cpu") as f:
            num_layers = self._get_num_layers(f)
            logging.info(f"Detected {num_layers} layers in the model schema.")
            
            self.extract_embeddings(f)
            
            for i in range(num_layers):
                logging.info(f"Extracting: Layer {i:02d} / {num_layers - 1}")
                self.extract_layer(f, i)
                
            self.extract_final_norms(f)
            
        logging.info(f"✨ Pipeline complete! Modular components saved to: {self.output_dir}/")


if __name__ == "__main__":
    # Define your paths
    MODEL_PATH = f"models/{LLM}/model.safetensors"
    OUTPUT_DIRECTORY = f"model_chunks/{LLM}"

    # Run the pipeline
    pipeline = MoEExtractionPipeline(MODEL_PATH, OUTPUT_DIRECTORY)
    pipeline.run()