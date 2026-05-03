import argparse
import json
import glob
import os
import re

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import (
    TransformersModel,
    TransformersModelConfig,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

# =========================================================================
# CONFIGURATION
# =========================================================================
# Default base model path if no argument is provided
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
MAX_SAMPLES = 100
OUTPUT_DIR = "./ifevalres"
STATUS_FILE = "IFEVAL.md"


def is_lora_adapter(model_path: str) -> bool:
    """Check if the given path contains a LoRA adapter."""
    return os.path.isfile(os.path.join(model_path, "adapter_config.json"))


def load_model_and_tokenizer(model_path: str) -> tuple:
    """Load model and tokenizer from path in 4-bit quantization.
    
    Detects whether model_path contains:
    - A LoRA adapter: loads base model + LoRA adapter
    - A full model: loads model directly
    """
    print(f"Analyzing model path: {model_path}")
    
    if is_lora_adapter(model_path):
        print("Detected LoRA adapter. Loading base model + adapter...")
        # Determine base model from adapter config
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct")
        print(f"Base model: {base_model_name}")
    else:
        print("Detected full model (no adapter_config.json found). Loading directly...")
        base_model_name = model_path
    
    # Load with 4-bit quantization
    print(f"Loading model in 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="auto",
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    
    # If LoRA adapter, merge adapters
    if is_lora_adapter(model_path):
        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path if not is_lora_adapter(model_path) else base_model_name,
        trust_remote_code=True,
    )
    
    # Ensure consistent padding behavior for evaluation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
        
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


def parse_status_file(status_file: str) -> dict:
    """Parse IFEVAL.md and return a dict of {checkpoint_dir: {metric: value}}."""
    results = {}
    if not os.path.exists(status_file):
        return results

    with open(status_file, "r") as f:
        content = f.read()

    # Match rows like: | checkpoint-100 | 0.45 | 0.50 | done |
    pattern = r"\|\s*(checkpoint-\d+|edited_model)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*(\w+)\s*\|"
    for match in re.finditer(pattern, content):
        checkpoint_dir = match.group(1)
        overall = float(match.group(2))
        follow = float(match.group(3))
        status = match.group(4)
        results[checkpoint_dir] = {
            "overall": overall,
            "follow": follow,
            "status": status,
        }

    return results


def write_status_file(status_file: str, results: dict, model_path: str = ""):
    """Write/update IFEVAL.md with evaluation results."""
    header = "| Checkpoint | Overall | Follow Inst. | Status |\n|---|---|---|---|\n"

    # Sort checkpoints numerically
    def sort_key(item):
        checkpoint_dir = item[0]
        match = re.search(r"(\d+)", checkpoint_dir)
        num = int(match.group(1)) if match else 0
        return num

    sorted_results = sorted(results.items(), key=sort_key)

    rows = ""
    for checkpoint_dir, metrics in sorted_results:
        overall = f"{metrics['overall']:.4f}"
        follow = f"{metrics['follow']:.4f}"
        status = metrics["status"]
        rows += f"| {checkpoint_dir} | {overall} | {follow} | {status} |\n"

    with open(status_file, "w") as f:
        f.write("# IFEVAL Evaluation Results\n\n")
        f.write(f"Model Path: {model_path}\n\n")
        f.write(header)
        f.write(rows)


def evaluate_model(model_path: str):
    """Evaluate the model on ifeval and return metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*60}")

    model, tokenizer = load_model_and_tokenizer(model_path)

    evaluation_tracker = EvaluationTracker(output_dir=OUTPUT_DIR)
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=MAX_SAMPLES,
    )

    config = TransformersModelConfig(model_name=model_path, batch_size=1)
    model = TransformersModel.from_model(model, config)

    pipeline = Pipeline(
        model=model,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        tasks="ifeval",
    )

    results = pipeline.evaluate()
    pipeline.show_results()
    eval_results = pipeline.get_results()

    results_dict = eval_results.get("results", {})

    # Prefer task-specific results if available
    ifeval_results = None
    for key, val in results_dict.items():
        if key.startswith("ifeval") and isinstance(val, dict):
            ifeval_results = val
            break

    # Fallback to "all"
    if ifeval_results is None:
        ifeval_results = results_dict.get("all", {})

    prompt_strict = float(ifeval_results.get("prompt_level_strict_acc", 0.0))
    inst_strict   = float(ifeval_results.get("inst_level_strict_acc", 0.0))
    prompt_loose  = float(ifeval_results.get("prompt_level_loose_acc", 0.0))
    inst_loose    = float(ifeval_results.get("inst_level_loose_acc", 0.0))

    overall = prompt_strict
    follow = inst_strict

    print(f"  prompt_level_strict_acc = {prompt_strict}")
    print(f"  inst_level_strict_acc   = {inst_strict}")
    print(f"  prompt_level_loose_acc  = {prompt_loose}")
    print(f"  inst_level_loose_acc    = {inst_loose}")

    status = "done"
    return {"overall": overall, "follow": follow, "status": status}


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate a model on IFEVAL benchmark")
    parser.add_argument(
        "model_path",
        type=str,
        default=None,
        help="Path to the model directory (LoRA adapter or full model). "
             "If not provided, uses the default model.",
    )
    args = parser.parse_args()
    
    # Determine which model to evaluate
    model_path = args.model_path if args.model_path else DEFAULT_MODEL_PATH
    
    # Verify path exists
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist.")
        return
    
    print(f"\nModel path: {model_path}")
    print(f"Is LoRA adapter: {is_lora_adapter(model_path)}")
    
    model_basename = os.path.basename(os.path.normpath(model_path))
    
    # Parse existing status file
    evaluated_checkpoints = parse_status_file(STATUS_FILE)

    if model_basename in evaluated_checkpoints:
        print(f"Model {model_basename} has already been evaluated on IFEVAL.")
        print(f"Results can be viewed in {STATUS_FILE}")
        return

    print(f"Evaluating on IFEVAL benchmark...\n")

    # Evaluate the model
    existing_results = {}
    if os.path.exists(STATUS_FILE):
        existing_results = parse_status_file(STATUS_FILE)

    metrics = evaluate_model(model_path)
    existing_results[model_basename] = metrics

    # Write status file with updated path info
    write_status_file(STATUS_FILE, existing_results, model_path=model_path)
    print(f"\nUpdated {STATUS_FILE} with results for {model_basename}")

    print(f"\n{'='*60}")
    print(f"Evaluation complete. Results saved to {STATUS_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
