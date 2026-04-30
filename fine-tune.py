import random
import json
import argparse
import os
from datetime import datetime
from copy import deepcopy
from typing import Any, Dict, List

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from peft import PeftModel

import yaml
from dataclasses import dataclass
from hparams import FTHyperParams 

detdebug = False

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
            if detdebug: break
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
    Executes the FT update algorithm (Pure PyTorch/Transformers implementation)
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
    
   
    # 2. Configure Optimizer
    # PEFT already handles freezing the base model
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    # 3. Training Loop
    loss_meter = AverageMeter()

    if detdebug: config.num_steps=1
    for it in range(config.num_steps):
        print(f"=== Epoch: {it} ===")
        loss_meter.reset()
        random.shuffle(requests)

        texts = [r["prompt"] for r in requests]
        targets = [r["target_new"] for r in requests]

        # Batch processing
        # vaut 100 par defaut !!
        config.batch_size=32
        for txt_batch, tgt_batch in zip(
            chunks(texts, config.batch_size), chunks(targets, config.batch_size)
        ):
            # Tokenize Prompt only (to get lengths)
            inputs = tok(txt_batch, return_tensors="pt", padding=True).to(device)
            
            # Tokenize Full Sequence (Prompt + Target)
            inputs_targets = [t + tg for t, tg in zip(txt_batch, tgt_batch)]
            # print("debug",inputs_targets)
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
    with open(config.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Construct requests directly from JSON
    # Assumes 'prompt' and 'target_new' (or 'alt') keys exist. Adjust logic if your JSON keys differ.
    requests = []
    for d in data:
        # Compatible with both your 'counterfact' and 'zsre' logic
        prompt = d.get('prompt', d.get('src'))
        target = d.get('target_new', d.get('alt'))
        
        if prompt and target:
            requests.append({
                "prompt": prompt,
                "target_new": target
            })
        else:
            print(f"Skipping invalid data entry: {d.keys()}")

    print(f"Loaded {len(requests)} editing requests.")

    # 3. Load Model & Tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )
    
    # Pad Token Setup
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)
    # Infer from rewrite_module (e.g., "model.layers.{}.mlp.down_proj.weight" -> "down_proj")
    module_path = config.rewrite_module.replace(".weight", "")
    target_modules = [module_path.split('.')[-1]]
    print(f"Inferred LoRA target modules: {target_modules} from {config.rewrite_module}")
    config.lora_r = 16
    config.lora_alpha = 16
    config.lora_dropout = 0

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=target_modules,
        layers_to_transform=[config.layer],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
 
    # 4. Execute Fine-Tuning
    print_time("Begin FT Time")
    edited_model = execute_ft(model, tokenizer, requests, config)
    print_time("End FT Time")

    # 5. Save
    save_path = config.save_model_dir
    print(f"Saving model to {save_path}")
    # edited_model.save_pretrained(save_path)
    # tokenizer.save_pretrained(save_path)

    # Save LoRA adapter first
    lora_path = save_path + "_lora"
    edited_model.save_pretrained(lora_path, save_embedding_layers=False)
    tokenizer.save_pretrained(lora_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    model_with_lora = PeftModel.from_pretrained(base_model, lora_path)
    merged_model = model_with_lora.merge_and_unload()

    # Save final model
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print("Finish Editing Process!!!!!!")
