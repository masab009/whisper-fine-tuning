from dataclasses import dataclass

@dataclass
class TrainConfig:
    # Data
    tsv_path: str = "./roman-urdu-dataset-20000/final_main_dataset_roman.tsv"
    audio_base_dir: str = "./roman-dataset-20000/limited_wav_files/limited_wav_files"
    test_size: float = 0.02
    seed: int = 42

    # Model
    model_name_or_path: str = "openai/whisper-small"

    # Tokenization
    max_label_length: int = 448  # Whisper labels length cap

    # Training output
    output_dir: str = "./whisper-RomanUrdu"

    # Trainer args (your defaults)
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    num_train_epochs: int = 3

    eval_strategy: str = "steps"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 25
    save_total_limit: int = 2

    # Generation / metrics
    generation_max_length: int = 256

    # Important toggles
    do_wer_eval: bool = False  # set True only if you want WER (uses generate -> can OOM)
    use_fp16_if_cuda: bool = True
    gradient_checkpointing: bool = False  # keep False (you were disabling it)
