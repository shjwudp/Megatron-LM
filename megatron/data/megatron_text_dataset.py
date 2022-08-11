import torch
import numpy as np

class MegatronTextDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, seq_len=2048) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.seq_len = seq_len

        raw_data = []
        for doc in data:
            raw_data.append(doc["text"])
        encoded_data = tokenizer(raw_data)
        self.flat_data = []
        for encoded_sentence in encoded_data["input_ids"]:
            self.flat_data += encoded_sentence + [tokenizer.eos_token_id]

        tokenizer.pad_token = tokenizer.unk_token
        encoded_data = tokenizer(raw_data)
        self.data = encoded_data['input_ids']
        self.attention_mask = encoded_data['attention_mask']

    def __len__(self):
        return (len(self.flat_data) - 1) // self.seq_len

    def __getitem__(self, index):
        st = index * self.seq_len
        sample = self.flat_data[st: st + self.seq_len + 1]
        return {'text': np.array(sample, dtype=np.int64)}
