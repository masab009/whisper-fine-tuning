from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from transformers import WhisperProcessor


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        audio_arrays = [f["audio"]["array"] for f in features]
        sampling_rate = features[0]["audio"]["sampling_rate"]

        input_features = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        ).input_features

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )

        # remove BOS if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]

        return {"input_features": input_features, "labels": labels}
