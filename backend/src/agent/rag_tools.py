from typing import List, Dict, Any, Type
from langchain.tools import BaseTool
# from langchain.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel,Field
from agent.knowledge_base import knowledge_base


class RetrieveInput(BaseModel):
    """检索工具输入"""
    query: str = Field(description="要检索的问题或关键词")
    top_k: int = Field(default=5, description="返回的最相关文档数量")


class LocalRAGTool(BaseTool):
    """本地 RAG 检索工具"""
    
    name: str = "retrieve_local_knowledge"
    description: str = """
    从本地知识库中检索相关信息。
    用于回答用户问题前，先查找本地是否有相关资料。
    输入应该是用户的问题或查询关键词。
    """
    args_schema: Type[BaseModel] = RetrieveInput
    
    def _run(self, query: str, top_k: int = 5) -> str:
        """执行本地知识库检索"""
        try:
            # 检索相关文档
            results = knowledge_base.search(query, top_k=top_k)
            
            if not results:
                return "本地知识库中未找到相关信息。"
            
            # 格式化检索结果
            formatted_results = []
            for i, result in enumerate(results, 1):
                content = result['content']
                metadata = result.get('metadata', {})
                source = metadata.get('source_file', '未知来源')
                similarity = result.get('similarity', 0)
                
                # 只包含相似度较高的结果
                if similarity > 0.3:  # 可调整阈值
                    formatted_results.append(f"""
【文档 {i}】(相似度: {similarity:.2f})
来源: {source}
内容: {content[:500]}{'...' if len(content) > 500 else ''}
""")
            
            if not formatted_results:
                return "本地知识库中未找到足够相关的信息。"
            
            return f"本地知识库检索结果:\n{''.join(formatted_results)}"
            
        except Exception as e:
            return f"检索本地知识库时出错: {str(e)}"
    
    async def _arun(self, query: str, top_k: int = 5) -> str:
        """异步执行"""
        return self._run(query, top_k)


def create_rag_tool() -> LocalRAGTool:
    """创建 RAG 检索工具"""
    return LocalRAGTool()


def evaluate_rag_sufficiency(rag_result: str, original_query: str) -> Dict[str, Any]:
    """评估 RAG 结果是否足够回答用户问题"""
    
    # 简单的启发式规则
    is_sufficient = False
    confidence = 0.0
    reason = ""
    
    if "未找到相关信息" in rag_result or "未找到足够相关" in rag_result:
        is_sufficient = False
        confidence = 0.0
        reason = "本地知识库中没有相关信息"
    elif "出错" in rag_result:
        is_sufficient = False
        confidence = 0.0
        reason = "检索过程中出现错误"
    else:
        # 检查内容质量
        content_length = len(rag_result)
        if content_length > 100:  # 内容足够长
            # 检查是否包含实质性信息
            if any(keyword in rag_result for keyword in ["相似度:", "来源:", "内容:"]):
                is_sufficient = True
                confidence = min(0.8, content_length / 1000)  # 基于内容长度的置信度
                reason = "本地知识库包含相关信息"
            else:
                is_sufficient = False
                confidence = 0.2
                reason = "检索到的信息质量不足"
        else:
            is_sufficient = False
            confidence = 0.1
            reason = "检索到的信息过少"
    
    return {
        "is_sufficient": is_sufficient,
        "confidence": confidence,
        "reason": reason,
        "rag_content": rag_result
    }
