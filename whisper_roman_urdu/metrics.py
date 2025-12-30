import evaluate


def build_compute_metrics(processor):
    wer = evaluate.load("wer")

    def compute_metrics(pred):
        # If predict_with_generate=False, pred.predictions may be loss-only / empty
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # HF sometimes returns tuples
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        return {"wer": wer.compute(predictions=pred_str, references=label_str)}

    return compute_metrics
