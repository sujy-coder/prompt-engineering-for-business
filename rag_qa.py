import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA

load_dotenv()

# 1. 加载文档
docs = []
for file in os.listdir("docs"):
    if file.endswith('.txt'):
        loader = TextLoader(f"docs/{file}", encoding="utf-8")
        docs.extend(loader.load())

# 2. 分块
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=[
        "\n\n",        # 优先按空行分（段落）
        "\n",          # 再按换行
        "。", "！", "？",  # 中文句号
        "；", "：",       # 分号、冒号
        " ", ""
    ]
)
splits = text_splitter.split_documents(docs)

# 3. 向量化
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
vectorstore = FAISS.from_documents(splits, embeddings)

# 4. 加载 Prompt 模板
# 如果没有模板文件，这里使用默认模板
try:
    with open("prompt_templates/stu_prompt_v2.txt", "r", encoding="utf-8") as f:
        template = f.read()
except FileNotFoundError:
    # 默认的问答模板
    template = """请基于以下上下文信息回答问题：
{context}

问题：{question}
答案："""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# 5. 构建 QA 链
llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": PROMPT}
)

# 6. 测试问答
if __name__ == "__main__":
    # 记录开始时间
    start_time = time.time()
    # 执行的代码块
    question = "学生请假需要遵循哪些程序？"
    result = qa_chain.invoke({"query": question})
    print(f"Q:")
    print(f"{question}")
    print("A:")
    print(f"{result['result']}")
    # 记录结束时间
    end_time = time.time()
    # 计算运行时间
    running_time = end_time - start_time
    print()
    print(f'程序运行时间：{running_time:.2f}秒')
