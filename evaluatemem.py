#!/usr/bin/env python3
"""
on passe un DATAFILE qui contient des questions avec la direction en reponse
"""

import string
import os
import re
import json
import random
import torch
from pathlib import Path
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

# ── Config ──
XPDIR = "/home/xtof/git/researchplm/ladder"
# memorisation:
DATAFILE = os.path.join(XPDIR, "nancy_mem.txt")
# generalisation:
# DATAFILE = os.path.join(XPDIR, "testset.txt")

CHECKPOINTS_DIR = os.path.join(XPDIR, "checkpoints")
BASE_MODEL = "unsloth/Qwen3-4B-Instruct-2507"
random.seed(42)

# Directions we care about
DIRECTIONS = ["nord", "sud", "est", "ouest"]


def extract_direction(text):
    s = text.lower()
    ss = s.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    s = ss.split(" ")
    nn, ns, ne, no = 0,0,0,0
    l=None
    for w in s:
        if w=="est":
            ne+=1
        elif w=="ouest":
            no+=1
        elif w=="nord":
            nn+=1
        elif w=="sud":
            ns+=1
        else: continue
        # on retourne le dernier NSEO de la reponse (because "A est à l'ouest")
        l=w
    if l==None: print("ERRORL",s)
    return l

def load_qa_lines(file_path, n=None):
    """Load QA pairs from the training file."""
    lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "?" not in line:
                continue
            question, answer = line.split("?", 1)
            question = question.strip() + "?"
            answer = answer.strip()
            if not answer: continue
            gt_dir = extract_direction(answer)
            if gt_dir is None: continue
            lines.append((question, answer, gt_dir))
    if n:
        random.shuffle(lines)
        lines = lines[:n]
    return lines

def build_prompt(tokenizer, question):
    """Build a chat prompt for the model."""
    messages = [
        {"role": "user", "content": question}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def evaluate_checkpoint(checkpoint_path, qa_samples, model, tokenizer, max_new_tokens=128):
    """Evaluate one checkpoint on qa_samples. Returns accuracy and per-sample results."""
    # Load LoRA adapter
    from peft import PeftModel
    model_eval = PeftModel.from_pretrained(
        model, checkpoint_path,
        adapter_name="default"
    )
    model_eval.eval()

    correct = 0
    results = []
    ntot = 0

    for i, (question, gt_answer, gt_dir) in enumerate(qa_samples):
        print("Q:",question,gt_answer)
        prompt = build_prompt(tokenizer, question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model_eval.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print("GEN",generated)
        pred_dir = extract_direction(generated)
        is_correct = pred_dir == gt_dir
        if is_correct: correct += 1
        ntot += 1

        results.append({
            "question": question,
            "gt_answer": gt_answer,
            "gt_dir": gt_dir,
            "generated": generated,
            "pred_dir": pred_dir,
            "correct": is_correct,
        })

        accuracy = correct / ntot
        print("ACC",accuracy)
    return accuracy, results

def main():
    print(f"Loading base model: {BASE_MODEL}")
    model, tokenizer = FastModel.from_pretrained(
        BASE_MODEL,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3")

    qa_samples = load_qa_lines(DATAFILE)
    print(f"Using {len(qa_samples)} samples for evaluation")

    # Get checkpoints
    checkpoint_dirs = sorted([
        d for d in Path(CHECKPOINTS_DIR).iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ], key=lambda x: int(x.name.split("-")[1]))
    print(f"Found {len(checkpoint_dirs)} checkpoints")

    # Evaluate each checkpoint
    all_results = {}
    for ckpt_path in checkpoint_dirs:
        ckpt_name = ckpt_path.name
        print(f"\n{'='*60}")
        print(f"Evaluating: {ckpt_name}")
        print(f"{'='*60}")

        accuracy, results = evaluate_checkpoint(ckpt_path, qa_samples, model, tokenizer)
        all_results[ckpt_name] = results

        # Count by direction
        dir_counts = {d: {"total": 0, "correct": 0} for d in DIRECTIONS}
        for r in results:
            d = r["gt_dir"]
            if d in dir_counts:
                dir_counts[d]["total"] += 1
                if r["correct"]:
                    dir_counts[d]["correct"] += 1

        print(f"Accuracy: {accuracy:.4f} ({sum(1 for r in results if r['correct'])}/{len(results)})")
        print("Per-direction accuracy:")
        for d in DIRECTIONS:
            total = dir_counts[d]["total"]
            corr = dir_counts[d]["correct"]
            acc = corr / total if total > 0 else 0
            print(f"  {d:4s}: {corr}/{total} = {acc:.4f}")

    # ── Summary table ──
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<25} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 45)
    for ckpt_name in sorted(all_results.keys(), key=lambda x: int(x.split("-")[1])):
        correct = sum(1 for r in all_results[ckpt_name] if r["correct"])
        acc = correct / len(qa_samples)
        print(f"{ckpt_name:<25} {acc:>10.4f} {correct:>10}")

    # Save detailed results
    output_file = os.path.join(XPDIR, "evaluation_results_detailed.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to: {output_file}")

    # Save summary CSV
    csv_file = os.path.join(XPDIR, "evaluation_summary.csv")
    with open(csv_file, "w") as f:
        f.write("checkpoint,accuracy,correct,total\n")
        for ckpt_name in sorted(all_results.keys(), key=lambda x: int(x.split("-")[1])):
            correct = sum(1 for r in all_results[ckpt_name] if r["correct"])
            acc = correct / len(qa_samples)
            f.write(f"{ckpt_name},{acc:.4f},{correct},{len(qa_samples)}\n")
    print(f"Summary CSV saved to: {csv_file}")

if __name__ == "__main__":
    main()
