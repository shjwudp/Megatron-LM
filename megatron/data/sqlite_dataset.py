"""GPT dataset backend by SQLite, provides text preprocessing functions and datasets.

Example of preprocessing jsonl text:
```bash
python sqlite_dataset.py \
    --input $(ls /cephfs/gpt/dataset/OpenWebText2/2020-*.jsonl) \
    --tokenizer_path /cephfs/gpt/huggingface.co/bigscience/bloom \
    --output gpt.sqlite
```
"""

import sqlite3
import torch
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import functools
import argparse


def preprocess_data(data, tokenizer_path, dbfile, bucket_size=32 * 1024 ** 2, tokenizer_batch=320 * 1024 ** 2):
    conn = sqlite3.connect(dbfile, isolation_level='DEFERRED')

    # turn `synchronous` and `journal_mode` to off can significantly improve performance
    conn.cursor().execute('''PRAGMA synchronous = OFF''')
    conn.cursor().execute('''PRAGMA journal_mode = OFF''')

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def batch_text_to_df(text_batch, tokens, offset, tokenizer=tokenizer, bucket_size=bucket_size):
        """Tokenize the input batch of text into buckets and output them in the form of pandas.DataFrame.

        Args:
            text_batch (List[str]): Input text list.
            tokens (List[int]): Due to the design of buckets, there will be some token residues each time, and `tokens` save the tokens that were processed before.
            offset (int): The offset of the current data in the global.
            bucket_size (int, optional): Bucket size.
            tokenizer (_type_, optional): Tokenizer.

        Returns:
            (pandas.DataFrame, List[int], int): Bucketed tokens, residue tokens and offset.
        """
        encoded_data = tokenizer(text_batch)
        text_batch = []
        for encoded_text in encoded_data["input_ids"]:
            tokens += encoded_text + [tokenizer.eos_token_id]

        df_list = []
        while len(tokens) > bucket_size:
            df_list.append({
                "tokens": np.asarray(tokens[:bucket_size], dtype=np.int64),
                "coord_start": offset,
                "coord_end": offset + bucket_size,
            })

            tokens = tokens[bucket_size:]
            offset += bucket_size

        return pd.DataFrame(df_list), tokens, offset

    tokens = []
    offset = 0
    text_batch = []
    cumulative_text_size = 0
    for text in tqdm(data):
        text_batch.append(text)
        cumulative_text_size += len(text)
        if cumulative_text_size > tokenizer_batch:
            df, tokens, offset = batch_text_to_df(text_batch, tokens, offset)
            text_batch = []
            cumulative_text_size = 0

            if len(df) > 0:
                df.to_sql("gpt_tokens", conn, if_exists="append", method="multi")

    # Save the remaining text
    if len(text_batch) > 0:
        df, tokens, offset = batch_text_to_df(text_batch, tokens, offset)
        text_batch = []

        if df.size > 0:
            df.to_sql("gpt_tokens", conn, if_exists="append", method="multi")

    # Save the remaining tokens
    if len(tokens) > 0:
        df = pd.DataFrame([{
            "tokens": np.asarray(tokens[:], dtype=np.int64),
            "coord_start": offset,
            "coord_end": offset + len(tokens),
        }])
        df.to_sql("gpt_tokens", conn, if_exists="append", method="multi")


class GPTSqliteDataset(torch.utils.data.Dataset):

    def __init__(self, dbfile, seq_len=1) -> None:
        super().__init__()
        self.conn = sqlite3.connect(dbfile, isolation_level='DEFERRED')
        self.cursor = self.conn.cursor()
        self.seq_len = seq_len

        self.cursor.execute('''PRAGMA synchronous = OFF''')
        self.cursor.execute('''PRAGMA journal_mode = OFF''')
        self.cursor.execute('''PRAGMA mmap_size=268435456;''')

    def __len__(self):
        coord_end = self.cursor.execute(
            "SELECT coord_end FROM gpt_tokens ORDER BY coord_end DESC LIMIT 1;").fetchone()
        # -1 due to we always fetch seq_len + 1 tokens
        return (coord_end - 1) // self.seq_len

    @functools.lru_cache(maxsize=2)
    def get_tokens(self, coord_start):
        tokens = self.cursor.execute(f'''SELECT coord_start, tokens FROM gpt_tokens WHERE coord_start={coord_start}''').fetchone()
        return tokens

    def __getitem__(self, index):
        """Fetch seq_len + 1 tokens

        Args:
            index (int): Item ID.

        Returns:
            torch.Tensor: Tensor contains seq_len + 1 token.
        """
        # Fetch seq_len + 1 tokens
        st, ed = index * self.seq_len, (index + 1) * self.seq_len + 1

        index = self.cursor.execute(f'''SELECT coord_start FROM gpt_tokens
            WHERE coord_start < {ed} AND {st} < coord_end;
            ''')

        data = [self.get_tokens(i[0]) for i in index]
        chunks = []
        for offset, chunk in data:
            chunk = np.frombuffer(chunk, dtype=np.int64)
            if offset < st:
                chunk = chunk[st - offset:]
            if ed < offset + len(chunk):
                chunk = chunk[:ed - offset]
            chunks.append(chunk)
        chunks = np.concatenate(chunks)

        return {"text": chunks[:self.seq_len + 1]}


def drop_and_rebuild_table_gpt_tokens(dbfile):
    # Connecting to sqlite
    # connection object
    connection_obj = sqlite3.connect(dbfile)

    # cursor object
    cursor_obj = connection_obj.cursor()

    # Drop gpt_tokens table if exists
    cursor_obj.execute("DROP TABLE IF EXISTS gpt_tokens;")

    # Creating table
    table = """
    CREATE TABLE IF NOT EXISTS "gpt_tokens" (
        "index" INTEGER,
        "tokens" TEXT,
        "coord_start" INTEGER,
        "coord_end" INTEGER
    );
    """

    cursor_obj.execute(table)
    cursor_obj.execute("""CREATE INDEX "ix_gpt_tokens_quick_search" ON "gpt_tokens" ("coord_start", "coord_end");""")
    cursor_obj.close()


def open_web_text2_gen(file_list):
    for file in file_list:
        for line in open(file):
            text = json.loads(line)["text"]
            yield text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs='+')
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    file_list = args.input
    tokenizer_path = args.tokenizer_path
    dbfile = args.output

    open_web_text2 = open_web_text2_gen(file_list)

    drop_and_rebuild_table_gpt_tokens(dbfile)
    preprocess_data(open_web_text2, tokenizer_path, dbfile)


if __name__ == "__main__":
    main()
