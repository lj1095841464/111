# debug_ner.py
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("./model/ner_model")
model = AutoModelForTokenClassification.from_pretrained("./model/ner_model")

text = "张三住在北京"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs).logits.argmax(-1)
preds = [model.config.id2label[i] for i in outputs[0].tolist()]

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
for tok, lab in zip(tokens, preds):
    if tok not in ["[CLS]", "[SEP]"]:
        print(f"{tok} -> {lab}")