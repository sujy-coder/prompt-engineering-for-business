import pandas as pd
from rag_qa import qa_chain  # 复用上面的链

# 模拟测试集
test_cases = [
    {"question": "员工年假有多少天？", "expected": "15天"},
    {"question": "试用期多久？", "expected": "3个月"},
    {"question": "如何申请加班？", "expected": "通过OA系统提交申请"},
]

results = []
for case in test_cases:
    output = qa_chain({"query": case["question"]})["result"]
    # 简单判断是否包含关键词（实际可用 rouge/bert-score）
    is_correct = case["expected"] in output
    results.append({
        "question": case["question"],
        "expected": case["expected"],
        "actual": output,
        "correct": is_correct
    })

df = pd.DataFrame(results)
accuracy = df["correct"].mean()
print(f"准确率: {accuracy:.2%}")

# 保存结果
df.to_csv("evaluation_results.csv", index=False)