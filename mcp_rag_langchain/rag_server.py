import asyncio
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class RAGSystem(object):
    def __init__(self, config):
        self.config = config
        self.llm = ChatOpenAI(
            model=os.getenv("MODEL", "qwen-plus"),
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY")
        )
        self.embedding = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBED_MODEL"),
        )
        self.vectorstore=Chroma(
            collection_name=self.config["collection_name"],
            embedding_function=self.embedding,
            persist_directory=self.config["persist_dir"]
        )

        self.retriver = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k":self.config.get("top_k", 5)
            }
        )
    def _load_documents(self, file_paths):
        docs=[]
        for path in file_paths:
            if path.endswith(".pdf"):
                loader = PyPDFLoader(path)
            elif path.endswith(".txt"):
                loader = TextLoader(path, encoding="utf-8")
            else:
                raise ValueError(f"跳过不支持的文件格式: {path}")
            docs.extend(loader.load())
        return docs

    def _chunk_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 50),
            length_function=len,
            is_separator_regex=True
        )
        return text_splitter.split_documents(docs)

    def build_knowledge(self, file_paths):
        #1.加载文档
        raw_docs = self._load_documents(file_paths)

        #2.切分文档-切块
        chunks = self._chunk_documents(docs = raw_docs)

        #3.生成向量并存储
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
        print(f"知识库构建完成,文档块数为:{len(chunks)}")

    def query(self, question):
        qa_chain=RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriver,
            return_source_documents=True
        )
        result = qa_chain.invoke({"query": question})
        return_result = {
            "answer": result["result"],
            "sources":[
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "N/A")
                }
                for doc in result["source_documents"]
            ]
        }
        return return_result
config = {
    "persist_dir": "./data/rag_db",
    "collection_name": "rag",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "top_k": 5
}

rag=RAGSystem(config)
rag.build_knowledge(
    file_paths=[
        "./data/doupocangqiong.txt"
    ]
)

from mcp.server.fastmcp import FastMCP
mcp = FastMCP("rag")

@mcp.tool()
async def rag_query(query):
    response = rag.query(query)
    return response["answer"]


async def search_demo():
    query = "xxx"
    response = await rag_query(query)
    print("response:",response)

if __name__ == '__main__':
    #asyncio.run(search_demo())
    mcp.run(transport="stdio")