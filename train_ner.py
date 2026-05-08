# train_ner.py
import json
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# ====== 修改 1：指定本地 BERT 模型路径 ======
BERT_MODEL_PATH = "./model/bert-base-chinese"  # ← 新增这一行

NUM_LABELS = 11  # O, B/I for NAME, ADDR, PHONE, ID, BANK

label_list = [
    "O",
    "B-PER", "I-PER",      # 人名
    "B-LOC", "I-LOC",      # 地址
    "B-PHONE", "I-PHONE",  # 电话（可选）
    "B-ID", "I-ID",        # 身份证（可选）
    "B-BANK", "I-BANK"     # 银行卡（可选）
]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for i, label in enumerate(label_list)}

class PIIDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_len=128):
        with open(data_file, encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        labels = [label2id.get(l, 0) for l in item["labels"]]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids()
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label_ids, dtype=torch.long)
        }

def main():
    # ====== 修改 2：从本地加载，禁止联网 ======
    tokenizer = BertTokenizerFast.from_pretrained(
        BERT_MODEL_PATH,           # ← 改这里
        local_files_only=True      # ← 关键！强制离线
    )
    model = BertForTokenClassification.from_pretrained(
        BERT_MODEL_PATH,           # ← 改这里
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
        local_files_only=True      # ← 关键！强制离线
    )

    train_dataset = PIIDataset("data/train_data_labeled.json", tokenizer)

    training_args = TrainingArguments(
        output_dir="./model/ner_model",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        save_steps=100,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=1,
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.config.id2label = id2label
    model.config.label2id = label2id
    model.save_pretrained("./model/ner_model")
    tokenizer.save_pretrained("./model/ner_model")
    print("✅ 模型训练完成，已保存至 ./model/ner_model")

if __name__ == "__main__":
    main()