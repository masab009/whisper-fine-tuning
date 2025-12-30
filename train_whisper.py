import argparse
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from whisper_roman_urdu.config import TrainConfig
from whisper_roman_urdu.data import load_roman_urdu_dataset, add_labels_column
from whisper_roman_urdu.modeling import load_processor_and_model, apply_gradient_checkpointing
from whisper_roman_urdu.collator import WhisperDataCollator
from whisper_roman_urdu.metrics import build_compute_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tsv_path", default=None)
    p.add_argument("--audio_base_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--model_name_or_path", default=None)
    p.add_argument("--do_wer_eval", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig()

    # allow CLI overrides
    if args.tsv_path: cfg.tsv_path = args.tsv_path
    if args.audio_base_dir: cfg.audio_base_dir = args.audio_base_dir
    if args.output_dir: cfg.output_dir = args.output_dir
    if args.model_name_or_path: cfg.model_name_or_path = args.model_name_or_path
    if args.do_wer_eval: cfg.do_wer_eval = True

    processor, model = load_processor_and_model(cfg.model_name_or_path)
    model = apply_gradient_checkpointing(model, cfg.gradient_checkpointing)

    dataset = load_roman_urdu_dataset(
        tsv_path=cfg.tsv_path,
        audio_base_dir=cfg.audio_base_dir,
        test_size=cfg.test_size,
        seed=cfg.seed,
    )

    # tokenize text -> labels (keep audio column)
    dataset["train"] = dataset["train"].map(
        lambda b: add_labels_column(b, processor.tokenizer, cfg.max_label_length),
        batched=True,
        batch_size=32,
        remove_columns=["text"],
    )
    dataset["validation"] = dataset["validation"].map(
        lambda b: add_labels_column(b, processor.tokenizer, cfg.max_label_length),
        batched=True,
        batch_size=32,
        remove_columns=["text"],
    )

    data_collator = WhisperDataCollator(processor)

    # NOTE:
    # - If you set do_wer_eval=True, we must generate during eval -> can increase VRAM usage.
    # - If False, we keep eval loss only (safer).
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.num_train_epochs,
        eval_strategy=cfg.eval_strategy,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        fp16=(torch.cuda.is_available() and cfg.use_fp16_if_cuda),
        report_to="none",
        save_total_limit=cfg.save_total_limit,
        remove_unused_columns=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        generation_max_length=cfg.generation_max_length,
        predict_with_generate=cfg.do_wer_eval,
        prediction_loss_only=(not cfg.do_wer_eval),
    )

    compute_metrics = build_compute_metrics(processor) if cfg.do_wer_eval else None

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    processor.save_pretrained(cfg.output_dir)

    print(f"\n✅ Saved to: {cfg.output_dir}")


if __name__ == "__main__":
    main()
