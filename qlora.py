import random
import argparse
import os
from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, List

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

import yaml
from dataclasses import dataclass
from hparams import FTHyperParams 

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
) -> AutoModelForCausalLM:
    """
    Executes the FT update algorithm using QLoRA (Quantized Low-Rank Adapters)
    applied to all linear layers of the LLM.
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
    
    # 1. Setup QLoRA Adapters on ALL Layers
    print("Initializing QLoRA adapters on all layers...")
    
    # Define which linear layers to apply LoRA to
    # Defaults to q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    if config.lora_target_modules is None:
        config.lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply QLoRA to all layers. The model was already quantized in 4-bit in main().
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 2. Configure Optimizer
    # Only the LoRA adapter parameters require gradients
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    
    # Freeze quantized base model weights explicitly
    for name, p in model.named_parameters():
        # if "lora" not in name.lower() and "embed" not in name.lower():
        if "lora" not in name.lower():
            p.requires_grad = False
        else:
            p.requires_grad = True
            if p.dim() == 1: torch.nn.init.constant_(p, 0)
    print(f"Trainable QLoRA parameters are active.")
    
    # 3. Training Loop
    loss_meter = AverageMeter()
    
    for it in range(config.num_steps):
        print(f"=== Epoch: {it} ===")
        loss_meter.reset()
        random.shuffle(requests)

        texts = [r["prompt"] for r in requests]
        targets = [r["target_new"] for r in requests]

        for txt_batch, tgt_batch in zip(
            chunks(texts, config.batch_size), chunks(targets, config.batch_size)
        ):
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
            
            opt.zero_grad()
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
            final_loss = masked_loss.mean()
                
            print(f"Batch loss: {final_loss.item():.4f}")
            loss_meter.update(final_loss.item(), n=bs)

            if final_loss.item() >= 1e-2:
                final_loss.backward()
                opt.step()

        print(f"Total Average Loss: {loss_meter.avg:.4f}")

        # Early stopping condition
        if loss_meter.avg < 1e-2:
            print("Loss threshold reached. Stopping early.")
            break
    
    return model

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Path & Device
    parser.add_argument('--config_path', type=str, required=True, help="Path to the yaml config file")
    # two overridable parameters
    parser.add_argument('--data_path', type=str, default=None, help="Override data path in config")
    parser.add_argument('--save_model_dir', type=str, default=None, help="Override save dir in config")
    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading config from {args.config_path}")
    config = FTHyperParams.from_yaml(args.config_path)
    
    if args.data_path:
        config.data_path = args.data_path
    if args.save_model_dir:
        config.save_model_dir = args.save_model_dir
    
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
    
    # Pad Token Setup
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        device_map='auto',
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
    )
    # PEFT's get_peft_model handles device map internally, but ensures embedding is on the right device
    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # 4. Execute Fine-Tuning (QLoRA)
    print_time("Begin QLoRA FT Time")
    edited_model = execute_ft(model, tokenizer, requests, config)
    print_time("End QLoRA FT Time")

    # 5. Save QLoRA Adapter
    save_path = config.save_model_dir
    print(f"Saving QLoRA adapter to {save_path}")
    edited_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("Finish QLoRA Editing Process!!!!!!")
