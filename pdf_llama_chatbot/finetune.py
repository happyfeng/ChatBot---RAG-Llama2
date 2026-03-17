# -*- coding: utf-8 -*-
"""
Fine-tune Llama2 on your PDF-derived data (LoRA / QLoRA).
Prepare data: finetune_data/train.jsonl with lines like:
  {"text": "### Human: question\\n### Assistant: answer"}
or
  {"instruction": "question", "output": "answer"}

Run: python finetune.py [--qlora]
"""
import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

# TRL 0.8+ uses SFTConfig; older versions pass dataset_text_field etc. directly to SFTTrainer
try:
    from trl import SFTConfig, SFTTrainer
    TRL_USE_SFT_CONFIG = True
except ImportError:
    from trl import SFTTrainer
    SFTConfig = None
    TRL_USE_SFT_CONFIG = False

import config

@dataclass
class FinetuneArgs:
    model_name: str = config.LLAMA2_MODEL
    output_dir: str = str(config.FINETUNE_OUTPUT_DIR)
    data_path: str = str(config.FINETUNE_DATA_DIR / "train.jsonl")
    use_qlora: bool = False
    num_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    max_seq_length: int = 512
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])

def load_dataset_from_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "text" in obj:
                data.append({"text": obj["text"]})
            elif "instruction" in obj and "output" in obj:
                data.append({
                    "text": f"### Human: {obj['instruction']}\n### Assistant: {obj['output']}"
                })
            else:
                data.append({"text": json.dumps(obj, ensure_ascii=False)})
    return Dataset.from_list(data)

def create_sample_data():
    """Create example train.jsonl if not exists."""
    config.FINETUNE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    sample = config.FINETUNE_DATA_DIR / "train.jsonl"
    if sample.exists():
        return
    with open(sample, "w", encoding="utf-8") as f:
        f.write('{"instruction": "什么是RAG？", "output": "RAG是检索增强生成，结合检索与生成模型回答基于文档的问题。"}\n')
        f.write('{"instruction": "LangChain是什么？", "output": "LangChain是用于构建大模型应用的框架，支持链、检索等组件。"}\n')
    print(f"Created sample {sample}. Replace with your own PDF-derived Q&A.")

def _is_local_model_path(model_name: str) -> bool:
    """True if model_name is an existing local directory (no HF access needed)."""
    p = Path(model_name)
    if p.is_absolute():
        return p.is_dir()
    return p.is_dir() or (config.PROJECT_ROOT / model_name).is_dir()

def run_finetune(args: FinetuneArgs):
    import sys
    config.ensure_dirs()
    create_sample_data()
    token = (config.HF_TOKEN or os.environ.get("HF_TOKEN") or "").strip()
    use_local = _is_local_model_path(args.model_name)
    if use_local:
        model_path = str(Path(args.model_name) if Path(args.model_name).is_absolute() else config.PROJECT_ROOT / args.model_name)
        args.model_name = model_path
        print("Using local model path:", model_path, "(no Hugging Face access)")
    elif not token:
        print("ERROR: Llama2 is a gated model. You must either:", file=sys.stderr)
        print("  1. Set HF_TOKEN in .env (accept terms at https://huggingface.co/meta-llama/Llama-2-7b-chat-hf, then create a token at https://huggingface.co/settings/tokens)", file=sys.stderr)
        print("  2. Use a local model: copy the model dir to this machine and run:", file=sys.stderr)
        print("     python finetune.py --model /path/to/Llama-2-7b-chat-hf --epochs 3", file=sys.stderr)
        sys.exit(1)

    load_kw = {"token": token if token else None, "trust_remote_code": True}
    if use_local:
        load_kw["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **load_kw)
    tokenizer.pad_token = tokenizer.eos_token

    if args.use_qlora:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            **{k: v for k, v in load_kw.items() if k != "trust_remote_code"},
        )
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            **{k: v for k, v in load_kw.items() if k != "trust_remote_code"},
        )
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    if not os.path.isfile(args.data_path):
        raise FileNotFoundError(f"Data not found: {args.data_path}. Create {args.data_path} with JSONL lines.")
    dataset = load_dataset_from_jsonl(args.data_path)

    if TRL_USE_SFT_CONFIG and SFTConfig is not None:
        # TRL 0.8+: SFTConfig extends TrainingArguments and holds dataset_text_field, max_seq_length, etc.
        sft_kw = {
            "dataset_text_field": "text",
            "max_seq_length": args.max_seq_length,
            "packing": False,
        }
        # SFTConfig may use max_seq_length or max_length depending on version
        if hasattr(SFTConfig, "max_length") and "max_seq_length" not in getattr(SFTConfig, "__dataclass_fields__", {}):
            sft_kw["max_length"] = args.max_seq_length
            sft_kw.pop("max_seq_length", None)
        training_args = SFTConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
            **sft_kw,
        )
        try:
            trainer = SFTTrainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
                processing_class=tokenizer,
            )
        except TypeError as e:
            if "processing_class" in str(e):
                # Older TRL with SFTConfig but still uses tokenizer=
                trainer = SFTTrainer(
                    model=model,
                    args=training_args,
                    train_dataset=dataset,
                    tokenizer=tokenizer,
                )
            else:
                raise
    else:
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="epoch",
            save_total_limit=2,
        )
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            packing=False,
        )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Fine-tuning done. Model saved to", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qlora", action="store_true", help="Use 4-bit QLoRA to save GPU memory")
    parser.add_argument("--data", default=str(config.FINETUNE_DATA_DIR / "train.jsonl"), help="Path to train.jsonl")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", default=str(config.FINETUNE_OUTPUT_DIR))
    parser.add_argument("--model", default=config.LLAMA2_MODEL, help="HF model id or local path (e.g. /path/to/Llama-2-7b-chat-hf); use local path to avoid 401/SSL")
    p = parser.parse_args()
    args = FinetuneArgs(use_qlora=p.qlora, data_path=p.data, num_epochs=p.epochs, output_dir=p.output, model_name=p.model)
    run_finetune(args)
