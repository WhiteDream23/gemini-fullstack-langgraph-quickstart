import os
import faiss
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain_core.documents import Document
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LocalKnowledgeBase:
    """本地知识库管理类"""
    
    def __init__(
        self,
        knowledge_dir: str = "./knowledge",
        vector_store_path: str = "./vector_store",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.knowledge_dir = Path(knowledge_dir)
        self.vector_store_path = Path(vector_store_path)
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 创建目录
        self.knowledge_dir.mkdir(exist_ok=True)
        self.vector_store_path.mkdir(exist_ok=True)
        
        # 初始化embedding模型
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"已加载嵌入模型: {embedding_model}")
        except Exception as e:
            logger.error(f"加载嵌入模型失败: {e}")
            # 使用默认模型
            try:
                self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("使用默认嵌入模型: all-MiniLM-L6-v2")
            except Exception as e2:
                logger.error(f"加载默认嵌入模型也失败: {e2}")
                self.embedding_model = None
        
        # 初始化文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        # 向量数据库
        self.index = None
        self.documents = []
        self.document_metadata = []
        
        # 加载现有的向量数据库
        self.load_vector_store()
    
    def load_documents(self) -> List[Document]:
        """加载知识库中的所有文档"""
        documents = []
        
        if not self.knowledge_dir.exists():
            logger.warning(f"知识库目录不存在: {self.knowledge_dir}")
            return documents
        
        # 支持的文件类型
        file_loaders = {
            ".md": lambda path: TextLoader(str(path), encoding="utf-8"),
            ".txt": lambda path: TextLoader(str(path), encoding="utf-8"),
        }
        
        for file_path in self.knowledge_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in file_loaders:
                try:
                    loader = file_loaders[file_path.suffix.lower()](file_path)
                    docs = loader.load()
                    
                    # 添加文件路径元数据
                    for doc in docs:
                        doc.metadata["source_file"] = str(file_path.name)
                        doc.metadata["file_type"] = file_path.suffix
                        doc.metadata["full_path"] = str(file_path)
                    
                    documents.extend(docs)
                    logger.info(f"已加载文档: {file_path}")
                    
                except Exception as e:
                    logger.error(f"加载文档失败 {file_path}: {e}")
        
        logger.info(f"总共加载了 {len(documents)} 个文档")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """分割文档为小块"""
        return self.text_splitter.split_documents(documents)
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """创建文本嵌入"""
        if self.embedding_model is None:
            logger.error("嵌入模型未初始化")
            return np.array([])
            
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            return np.array(embeddings).astype('float32')
        except Exception as e:
            logger.error(f"创建嵌入失败: {e}")
            return np.array([])
    
    def build_vector_store(self, force_rebuild: bool = False):
        """构建向量数据库"""
        index_file = self.vector_store_path / "faiss_index.bin"
        metadata_file = self.vector_store_path / "metadata.pkl"
        
        if not force_rebuild and index_file.exists() and metadata_file.exists():
            logger.info("向量数据库已存在，跳过构建")
            return
        
        logger.info("开始构建向量数据库...")
        
        # 加载和分割文档
        documents = self.load_documents()
        if not documents:
            logger.warning("没有找到文档，创建空的向量数据库")
            # 创建一个空的向量数据库
            self.documents = []
            self.document_metadata = []
            self.index = None
            return
        
        chunks = self.split_documents(documents)
        if not chunks:
            logger.warning("文档分割后为空，创建空的向量数据库")
            self.documents = []
            self.document_metadata = []
            self.index = None
            return
        
        # 提取文本和元数据
        texts = [chunk.page_content for chunk in chunks]
        metadata = [chunk.metadata for chunk in chunks]
        
        # 创建嵌入
        embeddings = self.create_embeddings(texts)
        if embeddings.size == 0:
            logger.error("嵌入创建失败")
            return
        
        # 创建FAISS索引
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # 保存文档和元数据
        self.documents = texts
        self.document_metadata = metadata
        
        # 保存到磁盘
        try:
            faiss.write_index(self.index, str(index_file))
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.document_metadata
                }, f)
            logger.info(f"向量数据库构建完成，包含 {len(texts)} 个文档块")
        except Exception as e:
            logger.error(f"保存向量数据库失败: {e}")
    
    def load_vector_store(self):
        """加载向量数据库"""
        index_file = self.vector_store_path / "faiss_index.bin"
        metadata_file = self.vector_store_path / "metadata.pkl"
        
        if index_file.exists() and metadata_file.exists():
            try:
                self.index = faiss.read_index(str(index_file))
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.document_metadata = data['metadata']
                logger.info(f"已加载向量数据库，包含 {len(self.documents)} 个文档块")
            except Exception as e:
                logger.error(f"加载向量数据库失败: {e}")
                self.index = None
                self.documents = []
                self.document_metadata = []
        else:
            logger.info("向量数据库不存在，将在首次查询时构建")
            self.index = None
            self.documents = []
            self.document_metadata = []
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关文档"""
        if self.index is None or len(self.documents) == 0:
            logger.warning("向量数据库未初始化，尝试构建...")
            self.build_vector_store()
            if self.index is None or len(self.documents) == 0:
                logger.warning("向量数据库仍然为空")
                return []
        
        # 创建查询嵌入
        query_embedding = self.create_embeddings([query])
        if query_embedding.size == 0:
            return []
        
        try:
            # 搜索
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):  # 确保索引有效
                    results.append({
                        'content': self.documents[idx],
                        'metadata': self.document_metadata[idx],
                        'score': float(score),
                        'similarity': 1 / (1 + score)  # 转换为相似度分数
                    })
            
            return results
        except Exception as e:
            logger.error(f"搜索过程中出现错误: {e}")
            return []
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None):
        """添加新文档到知识库"""
        if metadata is None:
            metadata = {}
        
        # 分割文档
        doc = Document(page_content=content, metadata=metadata)
        chunks = self.text_splitter.split_documents([doc])
        
        if not chunks:
            return
        
        # 创建嵌入
        texts = [chunk.page_content for chunk in chunks]
        chunk_metadata = [chunk.metadata for chunk in chunks]
        embeddings = self.create_embeddings(texts)
        
        if embeddings.size == 0:
            return
        
        # 添加到现有索引
        if self.index is None:
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.documents = []
            self.document_metadata = []
        
        self.index.add(embeddings)
        self.documents.extend(texts)
        self.document_metadata.extend(chunk_metadata)
        
        # 保存更新后的索引
        self.save_vector_store()
        logger.info(f"已添加 {len(chunks)} 个文档块到知识库")
    
    def save_vector_store(self):
        """保存向量数据库"""
        if self.index is None:
            return
        
        index_file = self.vector_store_path / "faiss_index.bin"
        metadata_file = self.vector_store_path / "metadata.pkl"
        
        try:
            faiss.write_index(self.index, str(index_file))
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.document_metadata
                }, f)
            logger.info("向量数据库已保存")
        except Exception as e:
            logger.error(f"保存向量数据库失败: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return {
            'total_chunks': len(self.documents) if self.documents else 0,
            'index_size': self.index.ntotal if self.index else 0,
            'knowledge_dir': str(self.knowledge_dir),
            'vector_store_path': str(self.vector_store_path),
            'embedding_model': self.embedding_model_name
        }


# 全局知识库实例
knowledge_base = LocalKnowledgeBase()
