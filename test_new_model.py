# test_new_model.py
from desensitize import ai_desensitize

# 测试多个典型例子
test_cases = [
    "张三住在北京，电话13812345678",
    "李四的身份证是110101199003072316，银行卡6222081234567890123",
    "王五在上海市浦东新区工作",
    "联系人：赵六，手机：15987654321"
]

for text in test_cases:
    result = ai_desensitize(text)
    print(f"原文: {text}")
    print(f"脱敏: {result}")
    print("-" * 50)