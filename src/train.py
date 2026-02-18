import torch
import os
from torch.utils.data import Dataset
from config import Config
from model import get_model
from transformers import Trainer, TrainingArguments
from torch.nn.attention import sdpa_kernel, SDPBackend

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class CipherPlainData(Dataset):
    def __init__(self):
        # TODO: here wer should load recurrence encoding first, then plaintext
        print("Loading dataset...")
        # TODO: overwrite below with the actual length of the dataset
        self.data_len = 1000

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        # TODO: Overwrite below with actual tensor data here, once we have it
        # Below can be deleted then, it is just for testing
        seq = torch.randint(0, Config.vocab_size, (Config.max_context,))
        return {
            "input_ids": seq,
            "labels": seq.clone(),
        }


def train():
    model = get_model()

    args = TrainingArguments(
        output_dir=Config.output_dir,
        num_train_epochs=Config.epochs,
        per_device_train_batch_size=Config.batch_size,
        gradient_accumulation_steps=Config.grad_accum,
        learning_rate=Config.learning_rate,
        gradient_checkpointing=Config.grad_checkpoint,
        logging_steps=Config.log_steps,
        save_steps=Config.save_steps,
        # OOM without below
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=CipherPlainData(),
    )

    print(f"Training on {torch.cuda.get_device_name(0)}...")

    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        trainer.train()

    trainer.save_model(f"{Config.output_dir}/model")


if __name__ == "__main__":
    train()
