from transformers import WhisperProcessor, WhisperForConditionalGeneration


def load_processor_and_model(model_name_or_path: str):
    # Keep your setup: language is the AUDIO language; task transcribe
    processor = WhisperProcessor.from_pretrained(
        model_name_or_path,
        language="urdu",
        task="transcribe",
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)

    # IMPORTANT for Roman Urdu training:
    # Do NOT force decoder prompt -> lets model adapt script
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False  # needed if enabling checkpointing; also safe generally

    return processor, model


def apply_gradient_checkpointing(model, enable: bool):
    # You had issues with checkpointing; keep it OFF by default.
    if enable:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
        model.config.use_cache = False
    return model
