Below is a solid, “drop-in” README you can paste into `README.md`, plus the exact steps to open-source your already-trained model folder on Hugging Face.

---

## README.md (copy/paste)

````md
# Whisper Roman-Urdu Fine-tuning (openai/whisper-small)

This repo fine-tunes OpenAI Whisper (small) for **Roman-Urdu transcription** using a TSV manifest + WAV files, and provides:
- Training script (`train_whisper.py`)
- Inference script for local WAVs (`transcribe_personal.py`)
- Modular code under `whisper_roman_urdu/`

## Project structure

.
├── train_whisper.py
├── transcribe_personal.py
├── whisper_roman_urdu/
│   ├── config.py
│   ├── data.py
│   ├── modeling.py
│   ├── collator.py
│   └── metrics.py
├── roman-urdu-dataset-20000/
│   ├── final_main_dataset_roman.tsv
│   └── limited_wav_files/limited_wav_files/*.wav
└── whisper-roman-urdu/   (trained model output)

## Requirements

- Linux / macOS (Windows should work too)
- Python 3.10+ recommended
- A CUDA GPU is recommended for training

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

## Dataset format

Your TSV should include at least these columns:

* `path` : filename (or relative path) of audio file
* `roman_sentence` : target transcription text

Example row:

```
path    roman_sentence
abc.wav yeh mera jumla hai
```

Audio files should exist under your audio base directory:

```
roman-urdu-dataset-20000/limited_wav_files/limited_wav_files/
```

## Training

Run training with explicit paths:

```bash
python train_whisper.py \
  --tsv_path "./roman-urdu-dataset-20000/final_main_dataset_roman.tsv" \
  --audio_base_dir "./roman-urdu-dataset-20000/limited_wav_files/limited_wav_files" \
  --output_dir "./whisper-roman-urdu"
```

### WER evaluation (optional)

WER evaluation uses generation during eval and can increase VRAM usage. Enable it like:

```bash
python train_whisper.py --do_wer_eval \
  --tsv_path "./roman-urdu-dataset-20000/final_main_dataset_roman.tsv" \
  --audio_base_dir "./roman-urdu-dataset-20000/limited_wav_files/limited_wav_files" \
  --output_dir "./whisper-roman-urdu"
```

## Inference on your own WAV files

Put WAVs into `./personal_recordings/` then run:

```bash
python transcribe_recordings.py \
  --model_dir "./whisper-roman-urdu" \
  --wav_dir "./personal_recordings" \
  --num_samples 10
```

## Notes / Troubleshooting

* If you see “Dropping N samples with missing audio files”, your `--audio_base_dir` does not match where the WAV files are.
* If eval crashes with OOM, run training without `--do_wer_eval` first.

# whisper-fine-tuning
