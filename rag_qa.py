import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
splits = text_splitter.split_documents(docs)

# 3. 向量化（使用 OpenAI embedding）
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embeddings)

# 4. 加载 Prompt 模板
# 如果没有模板文件，这里使用默认模板
try:
    with open("prompt_templates/qa_prompt_v1.txt", "r", encoding="utf-8") as f:
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
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    chain_type_kwargs={"prompt": PROMPT}
)

# 6. 测试问答
if __name__ == "__main__":
    question = "有哪些垄断行为？"
    result = qa_chain.invoke({"query": question})
    print(f"Q: {question}")
    print(f"A: {result['result']}")
