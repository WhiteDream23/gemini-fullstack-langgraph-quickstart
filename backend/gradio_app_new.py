import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import gradio as gr
import asyncio
from langchain_core.messages import HumanMessage
from agent.graph import graph
from agent.knowledge_base import knowledge_base
import json
from typing import List, Dict, Any
import time
import tempfile
from pathlib import Path


class ResearchAgent:
    def __init__(self):
        self.conversation_history = []
    
    def format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """格式化搜索来源"""
        if not sources:
            return "无可用来源"
        
        formatted_sources = []
        for i, source in enumerate(sources, 1):
            title = source.get('title', '未知标题')
            url = source.get('value', source.get('url', ''))
            content_preview = source.get('content', '')[:200] + "..." if source.get('content') else ""
            
            formatted_sources.append(f"""
**来源 {i}: {title}**
- 链接: {url}
- 摘要: {content_preview}
""")
        
        return "\n".join(formatted_sources)
    
    def research_query(self, question: str, initial_queries: int = 3, max_loops: int = 2, reasoning_model: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
        """执行研究查询"""
        if not question.strip():
            return "请输入一个有效的问题。", "", "", ""
        
        try:
            # 准备输入状态
            state = {
                "messages": [HumanMessage(content=question)],
                "initial_search_query_count": initial_queries,
                "max_research_loops": max_loops,
                "reasoning_model": reasoning_model,
            }
            
            # 执行图
            result = graph.invoke(state)
            
            # 提取结果
            messages = result.get("messages", [])
            sources = result.get("sources_gathered", [])
            search_queries = result.get("search_query", [])
            rag_result = result.get("rag_result", "")
            use_local_knowledge = result.get("use_local_knowledge", False)
            
            # 获取最终答案
            final_answer = ""
            if messages:
                final_answer = messages[-1].content
            
            # 格式化来源
            formatted_sources = self.format_sources(sources)
            
            # 格式化搜索查询
            formatted_queries = ""
            if search_queries:
                formatted_queries = "执行的搜索查询:\n" + "\n".join([f"• {query}" for query in search_queries])
            
            # 格式化 RAG 信息
            rag_info = ""
            if use_local_knowledge:
                rag_info = f"✅ 基于本地知识库回答\n\n本地检索结果:\n{rag_result[:500]}{'...' if len(rag_result) > 500 else ''}"
            elif rag_result:
                rag_info = f"ℹ️ 本地知识库检索:\n{rag_result[:300]}{'...' if len(rag_result) > 300 else ''}\n\n🌐 继续网络搜索获取更多信息"
            
            # 保存到对话历史
            self.conversation_history.append({
                "question": question,
                "answer": final_answer,
                "sources": sources,
                "rag_used": use_local_knowledge,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return final_answer, formatted_sources, formatted_queries, rag_info
            
        except Exception as e:
            error_msg = f"处理查询时出错: {str(e)}"
            return error_msg, "", "", ""
    
    def upload_document(self, file):
        """上传文档到知识库"""
        if file is None:
            return "请选择要上传的文件"
        
        try:
            # 获取文件路径
            file_path = Path(file.name)
            
            # 检查文件类型
            supported_types = ['.md', '.txt', '.pdf', '.docx']
            if file_path.suffix.lower() not in supported_types:
                return f"不支持的文件类型: {file_path.suffix}。支持的类型: {', '.join(supported_types)}"
            
            # 复制文件到知识库目录
            knowledge_dir = knowledge_base.knowledge_dir
            target_path = knowledge_dir / file_path.name
            
            # 确保目标目录存在
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 复制文件
            import shutil
            shutil.copy2(file_path, target_path)
            
            # 重建向量索引
            knowledge_base.build_vector_store(force_rebuild=True)
            
            return f"✅ 文件 '{file_path.name}' 已成功添加到知识库并重建索引"
            
        except Exception as e:
            return f"❌ 上传文件失败: {str(e)}"
    
    def search_knowledge_base(self, query: str, top_k: int = 5):
        """搜索本地知识库"""
        if not query.strip():
            return "请输入搜索查询"
        
        try:
            results = knowledge_base.search(query, top_k=top_k)
            
            if not results:
                return "❌ 没有找到相关结果"
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                content = result['content']
                metadata = result.get('metadata', {})
                source = metadata.get('source_file', '未知来源')
                similarity = result.get('similarity', 0)
                
                formatted_results.append(f"""
**【结果 {i}】** (相似度: {similarity:.3f})
**来源:** {source}
**内容:** {content[:300]}{'...' if len(content) > 300 else ''}

---
""")
            
            return f"🔍 找到 {len(results)} 个相关结果:\n\n{''.join(formatted_results)}"
            
        except Exception as e:
            return f"❌ 搜索失败: {str(e)}"
    
    def get_knowledge_base_stats(self):
        """获取知识库统计信息"""
        try:
            stats = knowledge_base.get_stats()
            
            info = f"""
📊 **知识库统计信息**

- **总文档块数:** {stats['total_chunks']}
- **索引大小:** {stats['index_size']}
- **知识库目录:** {stats['knowledge_dir']}
- **向量存储目录:** {stats['vector_store_path']}

"""
            
            # 检查知识库目录中的文件
            knowledge_dir = Path(stats['knowledge_dir'])
            if knowledge_dir.exists():
                files = list(knowledge_dir.rglob("*"))
                doc_files = [f for f in files if f.is_file() and f.suffix.lower() in ['.md', '.txt', '.pdf', '.docx']]
                
                info += f"- **知识库文件数:** {len(doc_files)}\n\n"
                
                if doc_files:
                    info += "**文件列表:**\n"
                    for file in doc_files[:10]:  # 只显示前10个文件
                        info += f"  - {file.name}\n"
                    if len(doc_files) > 10:
                        info += f"  - ... 和其他 {len(doc_files) - 10} 个文件\n"
            
            return info
            
        except Exception as e:
            return f"❌ 获取统计信息失败: {str(e)}"
    
    def rebuild_index(self):
        """重建知识库索引"""
        try:
            knowledge_base.build_vector_store(force_rebuild=True)
            return "✅ 知识库索引重建完成"
        except Exception as e:
            return f"❌ 重建索引失败: {str(e)}"
    
    def get_conversation_history(self):
        """获取对话历史"""
        if not self.conversation_history:
            return "暂无对话历史"
        
        history_text = []
        for i, conv in enumerate(self.conversation_history, 1):
            rag_badge = "🔗 RAG" if conv.get('rag_used', False) else "🌐 WEB"
            history_text.append(f"""
**对话 {i} ({conv['timestamp']})** {rag_badge}
**问题:** {conv['question']}
**回答:** {conv['answer'][:300]}{'...' if len(conv['answer']) > 300 else ''}
---
""")
        
        return "\n".join(history_text)


# 创建研究代理实例
agent = ResearchAgent()


def create_gradio_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(
        title="Qwen 智能研究助手 + 本地知识库",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="main-header">
            🔍 Qwen 智能研究助手 + 📚 本地知识库
        </div>
        <p style="text-align: center; font-size: 1.1em; color: #666;">
            基于 Qwen 模型、Tavily 搜索和本地 RAG 的智能研究工具，优先使用本地知识库，必要时补充网络搜索
        </p>
        """)
        
        with gr.Tab("💭 智能问答"):
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="请输入您的问题",
                        placeholder="例如：什么是机器学习？人工智能的发展历程？",
                        lines=3,
                        max_lines=5
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("🚀 开始研究", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ 高级设置")
                    initial_queries = gr.Slider(
                        label="初始查询数量",
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        info="生成的初始搜索查询数量"
                    )
                    max_loops = gr.Slider(
                        label="最大研究循环",
                        minimum=1,
                        maximum=5,
                        value=2,
                        step=1,
                        info="最大反思和补充搜索次数"
                    )
                    reasoning_model = gr.Dropdown(
                        label="推理模型",
                        choices=["Qwen/Qwen3-30B-A3B-Instruct-2507"],
                        value="Qwen/Qwen3-30B-A3B-Instruct-2507",
                        info="用于分析和推理的模型"
                    )
            
            gr.Markdown("---")
            
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Textbox(
                        label="📝 研究结果",
                        lines=15,
                        max_lines=20,
                        show_copy_button=True
                    )
                
                with gr.Column():
                    with gr.Tab("📚 信息来源"):
                        sources_output = gr.Textbox(
                            label="网络搜索来源",
                            lines=8,
                            max_lines=12,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("🔍 搜索记录"):
                        queries_output = gr.Textbox(
                            label="执行的搜索查询",
                            lines=5,
                            max_lines=8
                        )
                    
                    with gr.Tab("🔗 RAG 信息"):
                        rag_output = gr.Textbox(
                            label="本地知识库检索",
                            lines=8,
                            max_lines=12,
                            show_copy_button=True
                        )
        
        with gr.Tab("📚 知识库管理"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📤 上传文档")
                    file_upload = gr.File(
                        label="选择文件",
                        file_types=[".md", ".txt", ".pdf", ".docx"],
                        type="filepath"
                    )
                    upload_btn = gr.Button("📁 上传到知识库", variant="primary")
                    upload_result = gr.Textbox(label="上传结果", lines=3)
                
                with gr.Column():
                    gr.Markdown("### 🔍 搜索知识库")
                    search_input = gr.Textbox(
                        label="搜索查询",
                        placeholder="输入要搜索的内容..."
                    )
                    search_topk = gr.Slider(
                        label="返回结果数量",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1
                    )
                    search_btn = gr.Button("🔍 搜索", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    search_results = gr.Textbox(
                        label="🔍 搜索结果",
                        lines=15,
                        max_lines=20,
                        show_copy_button=True
                    )
                
                with gr.Column():
                    gr.Markdown("### ⚙️ 知识库管理")
                    
                    stats_btn = gr.Button("📊 查看统计信息", variant="secondary")
                    rebuild_btn = gr.Button("🔄 重建索引", variant="secondary")
                    
                    kb_stats = gr.Textbox(
                        label="📊 知识库信息",
                        lines=12,
                        max_lines=15
                    )
        
        with gr.Tab("📜 对话历史"):
            history_display = gr.Textbox(
                label="对话记录",
                lines=20,
                max_lines=25,
                show_copy_button=True
            )
            refresh_history_btn = gr.Button("🔄 刷新历史记录", variant="secondary")
        
        with gr.Tab("ℹ️ 使用说明"):
            gr.Markdown("""
            ## 🌟 功能特点
            
            ### 🤖 智能研究流程
            1. **本地优先:** 首先搜索本地知识库
            2. **智能判断:** 评估本地信息是否足够回答问题
            3. **网络补充:** 必要时进行网络搜索获取最新信息
            4. **综合回答:** 整合本地和网络信息提供全面答案
            
            ### 📚 本地知识库
            - **文档支持:** Markdown、TXT、PDF、DOCX
            - **向量检索:** 基于 FAISS 的高效语义搜索
            - **实时更新:** 上传文档后自动重建索引
            - **语义理解:** 使用 Sentence Transformers 进行嵌入
            
            ### 🔍 搜索能力
            - **实时网络搜索:** 获取最新信息
            - **多源验证:** 交叉验证信息准确性
            - **智能筛选:** 过滤高质量内容
            - **来源追溯:** 提供详细引用链接
            
            ### 💡 使用技巧
            1. **构建知识库:** 上传相关文档建立专业知识库
            2. **明确问题:** 问题越具体，答案越精准
            3. **查看 RAG 信息:** 了解答案来源（本地 vs 网络）
            4. **管理知识库:** 定期更新和维护知识库内容
            
            ### 📝 示例问题
            - "什么是机器学习？"（如果知识库中有相关文档）
            - "2024年人工智能的最新突破是什么？"（网络搜索）
            - "深度学习的基本概念"（本地知识库）
            - "可再生能源技术的发展趋势如何？"（混合搜索）
            
            ### 📚 知识库管理
            - **上传文档:** 支持多种格式的文档上传
            - **搜索测试:** 测试知识库的检索效果
            - **统计信息:** 查看知识库的规模和状态
            - **重建索引:** 在添加大量文档后重建索引
            
            ### ⚠️ 注意事项
            - 确保已配置 `OPENAI_API_KEY` 和 `TAVILY_API_KEY`
            - 本地知识库优先，可能不包含最新信息
            - 大文档建议分段上传以提高检索精度
            - 定期备份知识库文件和索引
            """)
        
        # 事件绑定 - 智能问答
        submit_btn.click(
            fn=agent.research_query,
            inputs=[question_input, initial_queries, max_loops, reasoning_model],
            outputs=[answer_output, sources_output, queries_output, rag_output],
            show_progress=True
        )
        
        question_input.submit(
            fn=agent.research_query,
            inputs=[question_input, initial_queries, max_loops, reasoning_model],
            outputs=[answer_output, sources_output, queries_output, rag_output],
            show_progress=True
        )
        
        clear_btn.click(
            fn=lambda: ("", "", "", "", ""),
            outputs=[question_input, answer_output, sources_output, queries_output, rag_output]
        )
        
        # 事件绑定 - 知识库管理
        upload_btn.click(
            fn=agent.upload_document,
            inputs=[file_upload],
            outputs=[upload_result],
            show_progress=True
        )
        
        search_btn.click(
            fn=agent.search_knowledge_base,
            inputs=[search_input, search_topk],
            outputs=[search_results],
            show_progress=True
        )
        
        stats_btn.click(
            fn=agent.get_knowledge_base_stats,
            outputs=[kb_stats]
        )
        
        rebuild_btn.click(
            fn=agent.rebuild_index,
            outputs=[kb_stats],
            show_progress=True
        )
        
        # 事件绑定 - 对话历史
        refresh_history_btn.click(
            fn=agent.get_conversation_history,
            outputs=[history_display]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
