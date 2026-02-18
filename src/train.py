import json
import glob
import os
import torch
from torch.utils.data import Dataset
from config import Config

class CipherPlainData(Dataset):
    def __init__(self, directory_path):
        self.file_paths = glob.glob(os.path.join(directory_path, "*.json"))
        print(f"Loaded {len(self.file_paths)} cipher files.")
        
        # Mapping: 
        # Let's assume ciphertext symbols are tokens 0-500
        # Let's assume plaintext 'a'-'z' are tokens 501-526
        # Token 527 is our separator/SOS token
        self.char_offset = 501
        self.sep_token = 527

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        with open(self.file_paths[idx], 'r') as f:
            data = json.load(f)

        # 1. Parse Ciphertext: "18 3 12..." -> [18, 3, 12, ...]
        cipher_ids = [int(x) for x in data["ciphertext"].split()]
        
        # 2. Parse Plaintext: "baronne..." -> [1, 0, 17, ...] + offset
        # ord('a') is 97, so ord(char) - 97 maps 'a' to 0.
        plain_ids = [(ord(c) - 97) + self.char_offset for c in data["plaintext"]]

        # 3. Combine: [Cipher] + [SEP] + [Plain]
        # We want the model to predict [Plain] given [Cipher]
        full_seq = cipher_ids + [self.sep_token] + plain_ids
        
        # Truncate or pad to max_context
        full_seq = full_seq[:Config.max_context]
        
        # 4. Create Labels
        # In CausalLM, we mask the "prompt" (ciphertext) so the model 
        # only gets graded on its ability to predict the plaintext.
        # -100 is the default ignore_index for CrossEntropyLoss
        labels = ([-100] * (len(cipher_ids) + 1)) + plain_ids
        labels = labels[:Config.max_context]

        # 5. Padding (if necessary, though Trainer handles this with a collator)
        padding_length = Config.max_context - len(full_seq)
        input_ids = full_seq + [0] * padding_length
        labels = labels + [-100] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }