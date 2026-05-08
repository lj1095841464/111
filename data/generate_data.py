# data/generate_data.py （可选运行，也可直接使用下方提供的 JSON）
import json
import random

names = ["张三", "李四", "王五", "赵六", "陈七"]
cities = ["北京市", "上海市", "广州市", "深圳市", "杭州市"]
phones = [f"138{random.randint(1000,9999):04d}{random.randint(1000,9999):04d}" for _ in range(20)]
ids = [f"{random.randint(110000,119999)}19900101{random.randint(1000,9999):04d}" for _ in range(20)]
banks = [f"622208{random.randint(100000000000,999999999999):012d}" for _ in range(20)]

def create_sample():
    text = f"{random.choice(names)}住在{random.choice(cities)}，电话是{random.choice(phones)}，身份证{random.choice(ids)}，银行卡{random.choice(banks)}。"
    return text

samples = [create_sample() for _ in range(100)]

# 简单标注（仅用于演示，实际应精细标注）
def label_text(text):
    tokens = list(text)
    labels = ['O'] * len(tokens)
    # 手动匹配关键词（简化版）
    for i in range(len(tokens)):
        substr = ''.join(tokens[i:i+3])
        if substr in names:
            for j in range(i, i+2):
                labels[j] = 'B-NAME' if j == i else 'I-NAME'
        elif substr in cities:
            for j in range(i, min(i+3, len(tokens))):
                labels[j] = 'B-ADDR' if j == i else 'I-ADDR'
        # 其他类型略（实际项目建议用正则+人工校验）
    return tokens, labels

data = []
for s in samples:
    tokens, labels = label_text(s)
    data.append({"tokens": tokens, "labels": labels})

with open("data/train_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)