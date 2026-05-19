import random
import argparse
import os
import re
import json
import subprocess
from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, List

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

import yaml
from dataclasses import dataclass
from hparams import FTHyperParams

# # Hyperparameter search (80 runs)
# python3 sft.py --config_path ./hparams/qwen2.5-7b.yaml \
#    --data_path ./data/zsre/zsre_3k.json \
#    --test_file ./testset.txt \
#    --search
# 
# # Single training (backward compatible, unchanged)
# python3 sft.py --config_path ./hparams/qwen2.5-7b.yaml \
#    --data_path ./data/zsre/zsre_3k.json \
#    --save_model_dir ./saves/my_run

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk

def print_time(process_name):
    now = datetime.now()
    print(f'{process_name}: {now.strftime("%m-%d %H:%M:%S")}')

# -----------------------------------------------------------------------------
# Core FT Logic
# -----------------------------------------------------------------------------

def execute_ft(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    config: FTHyperParams,
    save_dir: str = None,
    record_losses: bool = True,
) -> AutoModelForCausalLM:
    """
    Executes the FT update algorithm (Pure PyTorch/Transformers implementation)
    with gradient accumulation support.
    """
    device = torch.device(f'cuda:{config.device}')

    # Ensure left padding for generation-style objective alignment (though we use masking)
    if tok.padding_side != "left":
        tok.padding_side = "left"

    # Pre-process requests
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        # Add leading space if missing (tokenization detail)
        if len(request["target_new"]) > 0 and request["target_new"][0] != " ":
            request["target_new"] = " " + request["target_new"]

        # Optionally append EOS token (standard for editing to stop generation)
        request["target_new"] += tok.eos_token

        print(f"Refining request: [{request['prompt']}] -> [{request['target_new']}]")

    # 1. Select Weights to Update
    # Only the down_proj weights at the specified layer
    layers_to_edit = [config.layer]
    weights_to_update = {}

    for n, p in model.named_parameters():
        for layer in layers_to_edit:
            if config.rewrite_module.format(layer) in n:
                weights_to_update[n] = p

    if not weights_to_update:
        raise ValueError(f"No weights found matching module {config.rewrite_module} at layer {config.layer}")

    print(f"Weights to be updated ({len(weights_to_update)} params): {list(weights_to_update.keys())}")

    # 2. Configure Optimizer
    opt = torch.optim.Adam(
        weights_to_update.values(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # Freeze all parameters first
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze selected parameters
    for p in weights_to_update.values():
        p.requires_grad = True

    # 3. Training Loop with gradient accumulation
    loss_meter = AverageMeter()
    all_epoch_losses = []
    grad_accum = getattr(config, 'gradient_accumulation_steps', 1)

    for it in range(config.num_steps):
        print(f"=== Epoch: {it} ===")
        loss_meter.reset()
        random.shuffle(requests)

        texts = [r["prompt"] for r in requests]
        targets = [r["target_new"] for r in requests]

        # Split into gradient accumulation groups
        all_batches = list(zip(
            chunks(texts, config.batch_size), chunks(targets, config.batch_size)
        ))

        for acc_group_idx in range(0, len(all_batches), grad_accum):
            acc_group = all_batches[acc_group_idx:acc_group_idx + grad_accum]

            # Accumulated loss for this gradient step
            accumulated_loss = None
            accumulated_bs = 0

            for txt_batch, tgt_batch in acc_group:
                # Tokenize Prompt only (to get lengths)
                inputs = tok(txt_batch, return_tensors="pt", padding=True).to(device)

                # Tokenize Full Sequence (Prompt + Target)
                inputs_targets = [t + tg for t, tg in zip(txt_batch, tgt_batch)]
                full_inputs = tok(inputs_targets, return_tensors="pt", padding=True).to(device)

                # Calculate Masking
                # Goal: Loss is calculated ONLY on the Target tokens, not the Prompt tokens or Padding
                num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in inputs['input_ids'].cpu()]
                num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in full_inputs['input_ids'].cpu()]

                # The prompt length inside the full sequence includes the left-padding of the full sequence
                prompt_len = [x + y for x, y in zip(num_pad_toks, num_prompt_toks)]
                prompt_target_len = full_inputs['input_ids'].size(1)

                # Create mask: False for prompt/pad, True for target
                label_mask = torch.tensor([
                    [False] * length + [True] * (prompt_target_len - length)
                    for length in prompt_len
                ]).to(device)

                bs = full_inputs["input_ids"].shape[0]

                # Forward pass
                logits = model(**full_inputs).logits

                # Shift for autoregressive loss (pred next token)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = full_inputs['input_ids'][..., 1:].contiguous()

                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                loss = loss.view(bs, -1)

                # Apply mask (ignore prompt tokens in loss)
                # label_mask is sliced [:, 1:] because of the shift
                masked_loss = (loss * label_mask[:, 1:]).sum(1) / label_mask[:, 1:].sum(1)
                batch_loss = masked_loss.mean()

                # Accumulate loss (normalized by number of accumulation steps)
                if accumulated_loss is None:
                    accumulated_loss = batch_loss / len(acc_group)
                else:
                    accumulated_loss = accumulated_loss + batch_loss / len(acc_group)

                accumulated_bs += bs

                print(f"Batch loss: {batch_loss.item():.4f}")

            # Backward pass (once per gradient accumulation group)
            loss_meter.update(accumulated_loss.item(), n=accumulated_bs)

            if accumulated_loss.item() >= 1e-2:
                accumulated_loss.backward()
                opt.step()

            opt.zero_grad()

        epoch_loss = loss_meter.avg
        if record_losses:
            all_epoch_losses.append(epoch_loss)
        print(f"Total Average Loss: {epoch_loss:.4f}")

        # Save model checkpoint after each epoch
        if save_dir is not None:
            epoch_save_path = os.path.join(save_dir, f"checkpoint.epoch.{it}")
            print(f"Saving checkpoint to {epoch_save_path}")
            model.save_pretrained(epoch_save_path)
            tok.save_pretrained(epoch_save_path)

        # Early stopping condition
        if epoch_loss < 1e-2:
            print("Loss threshold reached. Stopping early.")
            break

    return model, all_epoch_losses

def compute_search_layers(num_total_layers: int, n_layers: int = 4) -> List[int]:
    """
    Compute n_layers uniformly spread between 50%-top and 90%-top of the model layers.

    Args:
        num_total_layers: Total number of transformer layers in the model
        n_layers: Number of layers to select (default 4)

    Returns:
        List of layer indices to train
    """
    # 50%-top means layers from index num_total_layers//2 onwards
    # 90%-top means layers up to index int(num_total_layers * 0.9)
    # So we search in [num_total_layers//2, int(num_total_layers * 0.9))
    start_layer = num_total_layers // 2
    end_layer = int(num_total_layers * 0.9)

    if end_layer <= start_layer:
        raise ValueError(f"end_layer ({end_layer}) must be greater than start_layer ({start_layer})")

    # Spread n_layers uniformly within [start_layer, end_layer)
    step = (end_layer - start_layer) / n_layers
    layers = [int(start_layer + step * i) for i in range(n_layers)]

    # Ensure uniqueness and clamp to valid range
    layers = sorted(list(set(max(0, min(l, num_total_layers - 1)) for l in layers)))

    print(f"Layer search range: [{start_layer}, {end_layer}), selected layers: {layers}")
    return layers


def run_evaluation(model_path: str, test_file: str) -> float:
    """
    Run evaluatemem.py on a trained model and return the FINAL ACCURACY.
    """
    cmd = [
        "python", "evaluatemem.py",
        model_path,
        test_file
    ]
    print(f"Running evaluation: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Evaluation stderr: {result.stderr}")

    # Parse "FINAL ACCURACY: 0.XXXX (N/M)" from output
    match = re.search(r'FINAL ACCURACY:\s*([\d.]+)', result.stdout)
    if match:
        acc = float(match.group(1))
        return acc
    else:
        print("WARNING: Could not parse FINAL ACCURACY from evaluation output")
        return -1.0


def search_hyperparams(
    config: FTHyperParams,
    data_path: str,
    test_file: str,
    base_save_dir: str = "./saves_search",
    n_layers: int = 4,
):
    """
    Search for the best hyper-parameters: layer, batch_size, lr.

    Batch sizes are per-step batch sizes; gradient_accumulation_steps is set so that
    the effective batch size does not exceed 4 (due to VRAM limits).

    For each combination:
    1. Train a model and save to a unique directory
    2. Evaluate with evaluatemem.py
    3. Record loss history and FINAL ACCURACY
    """
    # Hyperparameter search grids
    lr_values = [1e-4, 2e-4, 5e-5, 5e-4]
    batch_size_values = [4, 8, 16, 32, 64]

    # Load data first (shared across all runs)
    print(f"Loading data from {data_path}")
    requests = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split('?')
            if len(parts) != 2:
                continue
            prompt = parts[0].strip()
            target = parts[1].strip()
            if prompt and target:
                requests.append({
                    "prompt": prompt,
                    "target_new": target
                })
    print(f"Loaded {len(requests)} editing requests.")

    # Load model and tokenizer once (shared base)
    print("Loading base model and tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)

    # Pad Token Setup
    base_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    base_model.resize_token_embeddings(len(base_tokenizer))
    base_model.config.pad_token_id = base_tokenizer.pad_token_id
    base_model.to(torch.device(f'cuda:{config.device}'))

    # Determine total layers from model config
    num_total_layers = base_model.config.num_hidden_layers
    print(f"Model has {num_total_layers} layers")

    # Compute search layers
    layers_to_search = compute_search_layers(num_total_layers, n_layers)

    # Create results summary file
    os.makedirs(base_save_dir, exist_ok=True)
    results_summary_path = os.path.join(base_save_dir, "results_summary.json")
    all_results = []

    best_acc = -1.0
    best_config = None

    total_runs = len(layers_to_search) * len(batch_size_values) * len(lr_values)
    run_count = 0

    print(f"\n{'='*60}")
    print(f"Hyperparameter Search")
    print(f"  Layers: {layers_to_search}")
    print(f"  Batch sizes: {batch_size_values}")
    print(f"  Learning rates: {lr_values}")
    print(f"  Total runs: {total_runs}")
    print(f"{'='*60}\n")

    for layer in layers_to_search:
        for batch_size in batch_size_values:
            for lr in lr_values:
                run_count += 1

                # Gradient accumulation: cap per-step batch at 4 (VRAM limit)
                # effective_batch = batch_size, with per_step_batch ≤ 4
                per_step_batch = min(batch_size, 4)
                gradient_accum_steps = max(1, batch_size // per_step_batch)

                run_dir = os.path.join(
                    base_save_dir,
                    f"layer{layer}_bs{batch_size}_lr{lr:.0e}"
                )
                os.makedirs(run_dir, exist_ok=True)

                print(f"\n--- Run {run_count}/{total_runs} ---")
                print(f"Layer={layer}, BatchSize={batch_size} (per_step={per_step_batch}, acc_steps={gradient_accum_steps}), LR={lr:.0e}")
                print(f"Save dir: {run_dir}")

                # Clone the base model for this run
                import copy
                model_run = copy.deepcopy(base_model)
                model_run.to(torch.device(f'cuda:{config.device}'))

                # Create a config for this run
                run_config = FTHyperParams(
                    model_name_or_path=config.model_name_or_path,
                    data_path=data_path,
                    save_model_dir=run_dir,
                    layer=layer,
                    rewrite_module=config.rewrite_module,
                    batch_size=per_step_batch,
                    lr=lr,
                    weight_decay=config.weight_decay,
                    num_steps=config.num_steps,
                    gradient_accumulation_steps=gradient_accum_steps,
                    device=config.device,
                )

                # Save run config as yaml for reproducibility
                run_config_path = os.path.join(run_dir, "run_config.yaml")
                with open(run_config_path, 'w') as f:
                    yaml.dump({
                        'layer': layer,
                        'batch_size': batch_size,
                        'per_step_batch': per_step_batch,
                        'gradient_accumulation_steps': gradient_accum_steps,
                        'lr': lr,
                        'num_steps': config.num_steps,
                        'weight_decay': config.weight_decay,
                        'data_path': data_path,
                        'test_file': test_file,
                    }, f)

                # Execute fine-tuning
                print_time("Begin FT Time")
                try:
                    trained_model, epoch_losses = execute_ft(
                        model_run, base_tokenizer, requests, run_config, run_dir, record_losses=True
                    )
                    print_time("End FT Time")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print(f"OOM error on this configuration. Skipping.")
                        all_results.append({
                            "layer": layer,
                            "batch_size": batch_size,
                            "per_step_batch": per_step_batch,
                            "gradient_accumulation_steps": gradient_accum_steps,
                            "lr": lr,
                            "final_loss": None,
                            "epoch_losses": None,
                            "num_steps_completed": None,
                            "final_acc": None,
                            "error": "OOM"
                        })
                        torch.cuda.empty_cache()
                        continue
                    raise

                # Save loss history
                loss_history_path = os.path.join(run_dir, "loss_history.json")
                with open(loss_history_path, 'w') as f:
                    json.dump({
                        "layer": layer,
                        "batch_size": batch_size,
                        "lr": lr,
                        "epoch_losses": epoch_losses
                    }, f, indent=2)

                # Evaluate the trained model
                final_acc = run_evaluation(run_dir, test_file)

                # Record results
                result = {
                    "layer": layer,
                    "batch_size": batch_size,
                    "per_step_batch": per_step_batch,
                    "gradient_accumulation_steps": gradient_accum_steps,
                    "lr": lr,
                    "final_loss": epoch_losses[-1] if epoch_losses else None,
                    "epoch_losses": epoch_losses,
                    "num_steps_completed": len(epoch_losses) if epoch_losses else 0,
                    "final_acc": final_acc,
                    "save_dir": run_dir
                }
                all_results.append(result)

                # Update best
                if final_acc > best_acc:
                    best_acc = final_acc
                    best_config = result

                # Save running summary
                with open(results_summary_path, 'w') as f:
                    json.dump(all_results, f, indent=2)

                print(f"Run {run_count} complete: final_acc={final_acc:.4f}")

                # Clean up CUDA cache
                del model_run
                torch.cuda.empty_cache()

    # Print summary
    print(f"\n{'='*60}")
    print("HYPERPARAMETER SEARCH COMPLETE")
    print(f"{'='*60}")

    # Sort results by accuracy descending
    all_results_sorted = sorted(all_results, key=lambda x: x.get('final_acc') or -1, reverse=True)

    print("\nTop 10 results:")
    print(f"{'Rank':<5} {'Layer':<7} {'BS':<5} {'LR':<10} {'Final Acc':<12}")
    print("-" * 45)
    for i, r in enumerate(all_results_sorted[:10], 1):
        acc = r.get('final_acc')
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        print(f"{i:<5} {r['layer']:<7} {r['batch_size']:<5} {r['lr']:<10.0e} {acc_str:<12}")

    if best_config:
        print(f"\nBest configuration:")
        print(f"  Layer: {best_config['layer']}")
        print(f"  Batch Size: {best_config['batch_size']} (per_step={best_config['per_step_batch']}, acc_steps={best_config['gradient_accumulation_steps']})")
        print(f"  Learning Rate: {best_config['lr']:.0e}")
        print(f"  FINAL ACCURACY: {best_acc:.4f}")
        print(f"  Save dir: {best_config['save_dir']}")

    # Save best config to a dedicated file
    if best_config:
        best_path = os.path.join(base_save_dir, "best_config.json")
        with open(best_path, 'w') as f:
            json.dump(best_config, f, indent=2)

    print(f"\nAll results saved to: {results_summary_path}")
    print(f"Finish Editing Process!!!!!!")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training with optional hyperparameter search")
    # Path & Device
    parser.add_argument('--config_path', type=str, required=True, help="Path to the yaml config file")
    # two overridable parameters
    parser.add_argument('--data_path', type=str, default=None, help="Override data path in config")
    parser.add_argument('--save_model_dir', type=str, default=None, help="Override save dir in config")

    # Search mode arguments
    parser.add_argument('--search', action='store_true', help="Run hyperparameter search instead of single training")
    parser.add_argument('--test_file', type=str, default=None, help="Path to the test file for evaluation (required with --search)")
    parser.add_argument('--base_save_dir', type=str, default="./saves_search", help="Base directory for search results")
    parser.add_argument('--n_layers', type=int, default=4, help="Number of layers to search in the 50%-90% range")

    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading config from {args.config_path}")
    config = FTHyperParams.from_yaml(args.config_path)

    if args.data_path:
        config.data_path = args.data_path

    if args.search:
        # Hyperparameter search mode
        if not args.test_file:
            parser.error("--test_file is required when using --search")

        search_hyperparams(
            config=config,
            data_path=args.data_path if args.data_path else config.data_path,
            test_file=args.test_file,
            base_save_dir=args.base_save_dir,
            n_layers=args.n_layers,
        )
    else:
        # Single training mode (backward compatible)
        if not args.save_model_dir:
            args.save_model_dir = config.save_model_dir

        # 2. Load Data
        print(f"Loading data from {config.data_path}")
        requests = []
        with open(config.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split('?')
                # Drop lines with multiple or zero question marks
                if len(parts) != 2:
                    continue

                prompt = parts[0].strip()
                target = parts[1].strip()

                if prompt and target:
                    requests.append({
                        "prompt": prompt,
                        "target_new": target
                    })

        print(f"Loaded {len(requests)} editing requests.")

        # 3. Load Model & Tokenizer
        print("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path)

        # Pad Token Setup
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

        model.to(torch.device(f'cuda:{config.device}'))

        # 4. Execute Fine-Tuning
        print_time("Begin FT Time")
        edited_model, _ = execute_ft(model, tokenizer, requests, config, config.save_model_dir, record_losses=False)
        print_time("End FT Time")

        print("Finish Editing Process!!!!!!")
