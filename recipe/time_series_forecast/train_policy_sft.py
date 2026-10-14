"""State-response trajectory distillation; only teacher response tokens contribute loss."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset


def encode_step(step: dict, tokenizer, max_prompt_length: int = 4096,
                max_response_length: int = 2048) -> dict:
    messages = step["messages"]
    if [m["role"] for m in messages] != ["system", "user"]:
        raise ValueError("SFT states must be freshly assembled system/user messages")
    prompt = tokenizer.apply_chat_template(messages, tools=step.get("tools") or None,
                                           add_generation_prompt=True, tokenize=True)
    response = tokenizer.encode(step["response"], add_special_tokens=False)
    if not response or len(prompt) > max_prompt_length or len(response) + 1 > max_response_length:
        raise ValueError("Complete response/state exceeds the paper's SFT token budget")
    response = [*response, tokenizer.eos_token_id]
    return {"input_ids": [*prompt, *response], "attention_mask": [1] * (len(prompt) + len(response)),
            "labels": [-100] * len(prompt) + response}


class TrajectoryDataset(Dataset):
    def __init__(self, path: Path, tokenizer):
        from recipe.time_series_forecast.trajectory_distillation import validate_complete_trajectory
        self.rows = []
        self.dataset_counts = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            episode = json.loads(line)
            if episode.get("split") != "train" or not episode.get("complete"):
                raise ValueError("SFT accepts complete training-split trajectories only")
            validate_complete_trajectory(episode)
            dataset = episode["data_source"]
            self.dataset_counts[dataset] = self.dataset_counts.get(dataset, 0) + 1
            self.rows.extend(encode_step(step, tokenizer) for step in episode["steps"])
        if not self.rows:
            raise ValueError("No SFT state-response examples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_responses(rows: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    length = max(len(row["input_ids"]) for row in rows)
    return {key: torch.tensor([row[key] + [pad] * (length - len(row[key])) for row in rows])
            for key, pad in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100))}


def main():
    from functools import partial
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.global_batch_size % world_size:
        raise ValueError("Global batch size must be divisible by the number of GPUs")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32 if args.cpu else torch.bfloat16)
    num_parameters = sum(p.numel() for p in model.parameters())
    model.config.use_cache = False
    use_fsdp = world_size > 1 and not args.cpu
    settings = TrainingArguments(
        output_dir=str(args.output), num_train_epochs=args.epochs, learning_rate=args.lr,
        lr_scheduler_type="cosine", optim="adamw_torch", bf16=not args.cpu, use_cpu=args.cpu,
        per_device_train_batch_size=args.global_batch_size // world_size,
        gradient_accumulation_steps=1, gradient_checkpointing=not use_fsdp,
        fsdp="full_shard auto_wrap" if use_fsdp else "",
        fsdp_config={"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"],
                     "activation_checkpointing": True, "use_orig_params": True} if use_fsdp else {},
        report_to="none", save_strategy="epoch", logging_steps=1, remove_unused_columns=False,
    )
    dataset = TrajectoryDataset(args.trajectories, tokenizer)
    if not args.cpu:
        from recipe.time_series_forecast.trajectory_distillation import DATASETS
        if dataset.dataset_counts != {name: 120 for name in DATASETS}:
            raise ValueError("Paper SFT requires 120 complete teacher trajectories per dataset")
    trainer = Trainer(model=model, args=settings, train_dataset=dataset,
                      data_collator=partial(collate_responses, pad_token_id=tokenizer.pad_token_id))
    trainer.train()
    trainer.save_model(str(args.output))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output)
        manifest = {"base_model": args.model, "num_parameters": num_parameters,
                    "model_type": model.config.model_type, "precision": "float32" if args.cpu else "bfloat16",
                    "dataset_counts": dataset.dataset_counts, "num_trajectories": sum(dataset.dataset_counts.values()),
                    "response_examples": len(dataset), "learning_rate": args.lr, "epochs": args.epochs,
                    "global_batch_size": args.global_batch_size, "prompt_length": 4096, "response_length": 2048}
        (args.output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
