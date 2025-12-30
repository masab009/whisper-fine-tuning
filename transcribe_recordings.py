import os
import argparse
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import os
import warnings

warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers.utils import logging
logging.set_verbosity_error()
logging.disable_progress_bar()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", default="./whisper-roman-urdu")
    p.add_argument("--wav_dir", default="./personal_recordings")
    p.add_argument("--num_samples", type=int, default=10)
    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--max_length", type=int, default=225)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")
    processor = WhisperProcessor.from_pretrained(args.model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir).to(device)
    model.eval()

    # Force Urdu transcription for inference
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="ur", task="transcribe")
    model.config.suppress_tokens = []

    wav_files = sorted([
        os.path.join(args.wav_dir, f)
        for f in os.listdir(args.wav_dir)
        if f.lower().endswith(".wav")
    ])[: args.num_samples]

    if not wav_files:
        raise RuntimeError("No WAV files found!")

    print(f"Found {len(wav_files)} wav files\n")

    for idx, wav_path in enumerate(wav_files):
        print("=" * 90)
        print(f"Sample {idx}: {os.path.basename(wav_path)}")

        audio, _ = librosa.load(wav_path, sr=args.target_sr)

        inputs = processor(audio, sampling_rate=args.target_sr, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_features,
                max_length=args.max_length,
                num_beams=1,
                do_sample=False,
            )

        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("ASR:", transcription)

    print("\nDone.")


if __name__ == "__main__":
    main()
