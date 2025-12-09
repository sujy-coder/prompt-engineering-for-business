import pandas as pd
from rag_qa import qa_chain  # 复用上面的链

# 模拟测试集
test_cases = [
    {"question": "一家拥有专利技术的企业拒绝将其关键技术授权给竞争对手，是否必然构成违法？",
     "expected": "【结论】不必然违法；若存在正当理由（如保护知识产权、商业秘密等），则不构成滥用市场支配地位。 【依据】《中华人民共和国反垄断法》第二十二条第二款：“经营者依照有关知识产权的法律、行政法规规定行使知识产权的行为，不适用本法；但是，经营者滥用知识产权，排除、限制竞争的行为，适用本法。”"},
    {"question": "两家医疗器械企业合并，合计市场份额超过50%，但未向国务院反垄断执法机构申报，是否违法？",
     "expected": "【结论】如达到国务院规定的申报标准而未申报，则构成违法实施经营者集中。 【依据】《中华人民共和国反垄断法》第二十六条：“经营者集中达到国务院规定的申报标准的，经营者应当事先向国务院反垄断执法机构申报，未申报的不得实施集中。”"},
    {"question": "一家大型互联网公司以低于成本的价格在新业务领域持续补贴用户，意图排挤竞争对手，是否可能构成违法？",
     "expected": "【结论】若其具有市场支配地位且无正当理由，可能构成滥用市场支配地位。 【依据】《中华人民共和国反垄断法》第二十二条第一款第（二）项：“禁止具有市场支配地位的经营者从事下列滥用市场支配地位的行为：……（二）没有正当理由，以低于成本的价格销售商品。”"}
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