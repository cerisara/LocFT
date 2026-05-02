#!/usr/bin/env python3
"""
Evaluate a model (full LLM or LoRA adapter) on a test set.
The model is always loaded in 4-bit quantization.
Usage: python evaluatemem.py <model_path> <test_file>
"""

import string
import os
import json
import random
import argparse
import torch
from pathlib import Path
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from peft import PeftModel

random.seed(42)

# Directions evaluation
DIRECTIONS = ["nord", "sud", "est", "ouest"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model (LLM or LoRA) on a test set.")
    parser.add_argument("model_path", type=str, help="Path to the model (base LLM or LoRA adapter directory)")
    parser.add_argument("test_file", type=str, help="Path to the test file (QA pairs)")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum number of new tokens to generate")
    return parser.parse_args()


def is_lora_adapter(path):
    """Detect if a directory contains a LoRA adapter configuration."""
    if not os.path.isdir(path):
        return False
    adapter_config_path = os.path.join(path, "adapter_config.json")
    return os.path.exists(adapter_config_path)


def load_model_4bit(model_path):
    """Load the model in 4-bit, automatically handling base models and LoRA adapters."""
    lora = is_lora_adapter(model_path)
    
    print(f"{'Detecting model type...' if not lora else 'Loading Base Model (for LoRA)'}...")
    
    if lora:
        # Read base model name from LoRA config
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get("base_model_name_or_path")
        if not base_model_name:
            raise ValueError("Could not find 'base_model_name_or_path' in adapter_config.json. "
                             "Is this a valid LoRA save directory?")
        print(f"Detected LoRA adapter. Loading base model: {base_model_name}")
        
        full_model, tokenizer = FastModel.from_pretrained(
            base_model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        full_model = get_chat_template(tokenizer, chat_template="qwen3")
        
        print("Applying LoRA adapter in 4-bit...")
        full_model = PeftModel.from_pretrained(full_model, model_path, adapter_name="default")
        full_model.eval()
        
        print("LoRA model loaded successfully.")
    else:
        full_model, tokenizer = FastModel.from_pretrained(
            model_path,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        full_model = get_chat_template(tokenizer, chat_template="qwen3")
        full_model.eval()
        
        print("Full model loaded successfully.")

    return full_model, tokenizer


def extract_direction(text):
    """Extract the last occurring direction from text."""
    s = text.lower()
    ss = s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    words = ss.split(" ")
    last_dir = None
    for w in words:
        if w in DIRECTIONS:
            last_dir = w
    return last_dir


def load_qa_lines(file_path, n=None):
    """Load QA pairs from the test file."""
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "?" not in line:
                continue
            question, answer = line.split("?", 1)
            question = question.strip() + "?"
            answer = answer.strip()
            if not answer:
                continue
            gt_dir = extract_direction(answer)
            if gt_dir is None:
                continue
            lines.append((question, answer, gt_dir))
    if n:
        random.shuffle(lines)
        lines = lines[:n]
    return lines


def build_prompt(tokenizer, question):
    """Build a chat prompt for the model."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate_model(model, tokenizer, qa_samples, max_new_tokens=128):
    """Evaluate model on qa_samples. Returns accuracy and results."""
    correct = 0
    results = []
    
    for i, (question, gt_answer, gt_dir) in enumerate(qa_samples):
        print(f"\n--- Sample {i+1}/{len(qa_samples)} ---")
        print("Q:", question, gt_answer)
        
        prompt = build_prompt(tokenizer, question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("GEN:", generated)
        
        pred_dir = extract_direction(generated)
        is_correct = pred_dir == gt_dir
        if is_correct:
            correct += 1
            
        results.append({
            "question": question,
            "gt_answer": gt_answer,
            "gt_dir": gt_dir,
            "generated": generated,
            "pred_dir": pred_dir,
            "correct": is_correct,
        })
        
        accuracy = correct / (i + 1)
        print(f"ACC: {accuracy:.4f}")

    return accuracy, results


def main():
    args = parse_args()

    print(f"Model path: {args.model_path}")
    print(f"Test file: {args.test_file}")

    # Load model in 4-bit (automatically handles LoRA or full models)
    model, tokenizer = load_model_4bit(args.model_path)

    # Load test samples
    qa_samples = load_qa_lines(args.test_file)
    print(f"\nLoaded {len(qa_samples)} valid samples from test file.")
    if not qa_samples:
        print("No valid samples found. Exiting.")
        return

    # Evaluate
    accuracy, results = evaluate_model(model, tokenizer, qa_samples, args.max_new_tokens)

    # Aggregate results
    correct_count = sum(1 for r in results if r["correct"])
    total_count = len(results)
    print(f"\n{'='*60}")
    print(f"FINAL ACCURACY: {accuracy:.4f} ({correct_count}/{total_count})")
    print(f"{'='*60}")
    
    print("Per-direction accuracy:")
    dir_counts = {d: {"total": 0, "correct": 0} for d in DIRECTIONS}
    for r in results:
        d = r["gt_dir"]
        if d in dir_counts:
            dir_counts[d]["total"] += 1
            if r["correct"]:
                dir_counts[d]["correct"] += 1

    for d in DIRECTIONS:
        total = dir_counts[d]["total"]
        corr = dir_counts[d]["correct"]
        acc = corr / total if total > 0 else 0
        print(f"  {d:4s}: {corr}/{total} = {acc:.4f}")

    # Save detailed results as JSON
    output_dir = os.path.dirname(os.path.abspath(args.test_file))
    detailed_file = os.path.join(output_dir, "evaluation_results_detailed.json")
    with open(detailed_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {detailed_file}")

    # Save summary CSV
    csv_file = os.path.join(output_dir, "evaluation_summary.csv")
    with open(csv_file, "w") as f:
        f.write("accuracy,correct,total\n")
        f.write(f"{accuracy:.4f},{correct_count},{total_count}\n")
    print(f"Summary CSV saved to: {csv_file}")


if __name__ == "__main__":
    main()
