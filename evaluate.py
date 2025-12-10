import time
import pandas as pd
from rag_qa import qa_chain  # 复用上面的链
from typing import List, Dict, Callable
from rouge_score import rouge_scorer
import bert_score

import re

# 记录开始时间
start_time = time.time()

def load_txt_cases(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    blocks = re.split(r'\n\s*\n', content)
    cases = []
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        q = lines[0].replace("【问题】", "").strip()
        c = lines[1].replace("【结论】", "").strip()
        cases.append({"question": q, "expected": f"【结论】{c}"})
    return cases

# 测试集
test_cases = load_txt_cases("stu_test_cases.txt")

def keyword_match(expected: str, actual: str) -> bool:
    """最简单的包含判断（可扩展为模糊匹配等）"""
    return expected.strip().lower() in actual.strip().lower()


def rouge_l_match(expected: str, actual: str, threshold: float = 0.5) -> bool:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(expected, actual)['rougeL'].fmeasure
    return score >= threshold


def bertscore_match(expected: str, actual: str, threshold: float = 0.5, lang: str = 'en') -> bool:
    P, R, F1 = bert_score.score([actual], [expected], lang=lang, verbose=False)
    return F1.item() >= threshold


def evaluate_qa(
        test_cases: List[Dict],
        qa_chain,
        eval_fn: Callable[[str, str], bool] = keyword_match,
) -> pd.DataFrame:
    """
    通用 QA 评估函数

    Args:
        test_cases: 测试用例列表，每个包含 "question" 和 "expected"
        qa_chain: 接受 {"query": ...} 返回 {"result": ...} 的调用对象
        eval_fn: 评估函数，接收 (expected, actual) -> bool

    Returns:
        包含结果和准确率的 DataFrame
    """
    results = []
    for case in test_cases:
        question = case["question"]
        expected = case["expected"]
        actual = qa_chain.invoke({"query": question})["result"]

        try:
            correct = eval_fn(expected, actual)
        except Exception as e:
            print(f"评估出错（问题: {question[:50]}...）: {e}")
            correct = False

        results.append({
            "question": question,
            "expected": expected,
            "actual": actual,
            "correct": correct
        })

    df = pd.DataFrame(results)
    accuracy = df["correct"].mean()
    print(f"准确率: {accuracy:.2%}")
    # 保存结果
    df.to_csv("evaluation_results.csv", index=False)

# # 默认：关键词匹配
# df1 = evaluate_qa(test_cases, qa_chain)

# 使用 ROUGE-L
df2 = evaluate_qa(test_cases, qa_chain,
                      eval_fn=lambda exp, act: rouge_l_match(exp, act, threshold=0.6))
#
# # 使用 BERTScore
# df3 = evaluate_qa(test_cases, qa_chain,
#                       eval_fn=lambda exp, act: bertscore_match(exp, act, threshold=0.7, lang='zh'))

# 记录结束时间
end_time = time.time()
# 计算运行时间
running_time = end_time - start_time
print()
print(f'程序运行时间：{running_time:.2f}秒')