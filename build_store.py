import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

load_dotenv()

# 1. 加载文档
docs = []
for file in os.listdir("docs"):
    if file.endswith('.txt'):
        loader = TextLoader(f"docs/{file}", encoding="utf-8")
        docs.extend(loader.load())

# 2. 分块
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

vectorstore.save_local("faiss_index")  # 持久化