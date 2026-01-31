# LoRA fine-tune on recipe data (ingredients -> recipe text). Uses TinyLlama by default.

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)


def load_training_data(path: Path) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def format_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
    return "".join(parts)


def tokenize(examples: dict, tokenizer, max_length: int) -> dict:
    texts = []
    for messages in examples["messages"]:
        texts.append(format_prompt(messages))
    out = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    out["labels"] = [list(ids) for ids in out["input_ids"]]
    # only train on assistant reply, mask the user part
    for i in range(len(out["labels"])):
        user_only = format_prompt(examples["messages"][i][:-1])
        prompt_ids = tokenizer(
            user_only,
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]
        prompt_len = min(len(prompt_ids), len(out["labels"][i]))
        for j in range(prompt_len):
            out["labels"][i][j] = -100
    return out


def main(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    training_data_path: Path | None = None,
    output_dir: str = "recipe_model",
    num_epochs: int = 3,
    batch_size: int = 2,
    max_length: int = 512,
    use_4bit: bool = True,
) -> None:
    if training_data_path is None:
        training_data_path = PROJECT_ROOT / "dataset" / "training_data.jsonl"
    if not training_data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {training_data_path}. "
            "Run: python -m dataset.prepare_training_data"
        )

    raw = load_training_data(training_data_path)
    dataset = Dataset.from_list(raw)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        return tokenize(examples, tokenizer, max_length)

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    load_kwargs = {}
    if use_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        **load_kwargs,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_path),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune recipe model")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model")
    parser.add_argument("--data", default=None, help="Path to training_data.jsonl")
    parser.add_argument("--output", default="recipe_model", help="Output dir")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--no-4bit", action="store_true", help="No 4bit quant (needs more VRAM)")
    args = parser.parse_args()
    main(
        model_name=args.model,
        training_data_path=Path(args.data) if args.data else None,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        use_4bit=not args.no_4bit,
    )
