from __future__ import annotations

from pytorch_lightning.callbacks import Callback


class LoRASaveCallback(Callback):
    def __init__(self, save_path: str):
        self.save_path = save_path

    def on_train_end(self, trainer, pl_module):
        # Save only the LoRA adapters at the end of training
        if hasattr(pl_module, "lora_model"):
            pl_module.lora_model.save_pretrained(self.save_path)
        else:
            raise ValueError("The model does not have a `lora_model` attribute.")
