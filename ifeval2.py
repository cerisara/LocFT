import glob
import os
import re

from transformers import AutoModelForCausalLM, AutoTokenizer

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.transformers.transformers_model import (
    TransformersModel,
    TransformersModelConfig,
)
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters

# =========================================================================
# CONFIGURATION
# =========================================================================
# Path to the directory where fine-tune.py saved the model (config.save_model_dir)
TRAINED_MODEL_DIR = "./nancytrain_output/edit_model"
MAX_SAMPLES = 100
OUTPUT_DIR = "./results"
STATUS_FILE = "IFEVAL.md"


def load_edited_model(model_dir: str):
    """Load the fully fine-tuned model directly from the saved directory."""
    print(f"Loading fine-tuned model from: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
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


def write_status_file(status_file: str, results: dict):
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
        f.write(f"Trained Model Path: {TRAINED_MODEL_DIR}\n\n")
        f.write(header)
        f.write(rows)


def evaluate_edited_model(model_dir: str):
    """Evaluate the fine-tuned model on ifeval and return metrics."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_dir}")
    print(f"{'='*60}")

    model, tokenizer = load_edited_model(model_dir)

    evaluation_tracker = EvaluationTracker(output_dir=OUTPUT_DIR)
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.NONE,
        max_samples=MAX_SAMPLES,
    )

    config = TransformersModelConfig(model_name=model_dir, batch_size=1)
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
    # Check if the trained model directory exists
    if not os.path.isdir(TRAINED_MODEL_DIR):
        print(f"Trained model directory not found: {TRAINED_MODEL_DIR}")
        print("Please ensure fine-tune.py has run and saved the model to this path.")
        return

    # Parse existing status file
    evaluated_checkpoints = parse_status_file(STATUS_FILE)
    model_basename = os.path.basename(TRAINED_MODEL_DIR)

    if os.path.basename(TRAINED_MODEL_DIR) in evaluated_checkpoints:
        print(f"Model {model_basename} has already been evaluated on IFEVAL.")
        print(f"Results can be viewed in {STATUS_FILE}")
        return

    print(f"Found trained model at: {TRAINED_MODEL_DIR}")
    print(f"Evaluating on IFEVAL benchmark...\n")

    # Evaluate the trained model
    existing_results = {}
    if os.path.exists(STATUS_FILE):
        existing_results = parse_status_file(STATUS_FILE)

    metrics = evaluate_edited_model(TRAINED_MODEL_DIR)
    existing_results[model_basename] = metrics

    # Update status file
    write_status_file(STATUS_FILE, existing_results)
    print(f"\nUpdated {STATUS_FILE} with results for {model_basename}")

    print(f"\n{'='*60}")
    print(f"Evaluation complete. Results saved to {STATUS_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
