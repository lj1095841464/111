# auto_label_names.py - 支持 JSON 数组格式
import json

# 1. 读取人名列表
with open("names.txt", "r", encoding="utf-8") as f:
    names = set(line.strip() for line in f if line.strip())

# 2. 读取整个训练数据（作为 JSON 数组）
with open("train_data.json", "r", encoding="utf-8") as f:
    data_list = json.load(f)  # ← 关键：用 json.load 读整个数组

# 3. 遍历每条数据
for data in data_list:
    tokens = data["tokens"]
    labels = data["labels"]
    text = "".join(tokens)
    
    # 初始化新标签（保留原标签，只覆盖人名部分）
    new_labels = labels[:]
    
    # 遍历每个人名进行匹配
    for name in names:
        start = 0
        while True:
            pos = text.find(name, start)
            if pos == -1 or len(name) == 0:
                break
            # 打上 B-PER / I-PER
            new_labels[pos] = "B-PER"
            for i in range(1, len(name)):
                if pos + i < len(new_labels):
                    new_labels[pos + i] = "I-PER"
            start = pos + 1
    
    data["labels"] = new_labels

# 4. 保存回文件（保持原格式：JSON 数组）
with open("train_data_labeled.json", "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print("✅ 自动标注完成！新文件：train_data_labeled.json")