import re
import torch
from transformers import BertTokenizerFast, BertForTokenClassification

# ===== 规则脱敏 =====
def rule_based_desensitize(text):
    # 先标记出所有11位数字（包括学号）
    def replace_long_digits(m):
        num = m.group()
        # 如果是合法手机号，用部分脱敏
        if re.match(r'1[3-9]\d{9}', num):
            return num[:3] + '****' + num[-4:]
        else:
            # 学号、工号等 → 全部脱敏
            return '*' * len(num)

    # 匹配所有11位数字
    text = re.sub(r'\b\d{11}\b', replace_long_digits, text)

    # 身份证
    text = re.sub(r'\d{17}[\dXx]', lambda m: m.group()[:6] + '**************' + m.group()[-4:], text)
    
    # 银行卡
    text = re.sub(r'\d{16,19}', lambda m: m.group()[:4] + ' **** **** ' + m.group()[-4:], text)
    
    return text
# ===== AI 脱敏 =====
#print("Merged tokens:", merged_tokens)
#print("Merged labels:", merged_labels)
def ai_desensitize(text):
    try:
        tokenizer = BertTokenizerFast.from_pretrained("./model/ner_model")
        model = BertForTokenClassification.from_pretrained("./model/ner_model")
        model.eval()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs).logits

        predictions = torch.argmax(outputs, dim=2)[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        word_ids = inputs.word_ids(batch_index=0)

        # 合并子词（如 "##三"）
        merged_tokens = []
        merged_labels = []
        current_word = None

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != current_word:
                current_word = word_id
                token = tokens[idx].replace("##", "")
                label = model.config.id2label[predictions[idx]]
                merged_tokens.append(token)
                merged_labels.append(label)
            else:
                merged_tokens[-1] += tokens[idx].replace("##", "")

        # 合并连续实体并脱敏
        result_chars = []
        i = 0
        while i < len(merged_tokens):
            label = merged_labels[i]
            if label in ("B-PER", "I-PER"):
                start = i
                while i < len(merged_tokens) and merged_labels[i] in ("B-PER", "I-PER"):
                    i += 1
                name = "".join(merged_tokens[start:i])
                result_chars.append(name[0] + "*" * (len(name) - 1))
            elif label in ("B-LOC", "I-LOC"):
                start = i
                while i < len(merged_tokens) and merged_labels[i] in ("B-LOC", "I-LOC"):
                    i += 1
                loc = "".join(merged_tokens[start:i])
                result_chars.append("*" * (len(loc) - 1))
            else:
                result_chars.append(merged_tokens[i])
                i += 1

        return "".join(result_chars)

    except Exception as e:
        print(f"⚠️ AI脱敏失败: {e}")
        return text  # 回退到原始文本

# ===== 主函数 =====
def desensitize_text(text):
    text = ai_desensitize(text)      # 先处理人名、地址
    text = rule_based_desensitize(text)  # 再处理电话、身份证等
    return text