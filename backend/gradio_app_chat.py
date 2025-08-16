import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import graph
from agent.knowledge_base import knowledge_base
import json
from typing import List, Dict, Any, Tuple
import time
import tempfile
import asyncio
from pathlib import Path
import traceback


class ChatResearchAgent:
    def __init__(self):
        self.sessions = {}  # 存储多个会话
    
    def create_session(self, session_id: str = None) -> str:
        """创建新的会话"""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        self.sessions[session_id] = {
            "messages": [],
            "conversation_state": None,
            "is_waiting_for_clarification": False,
            "last_result": None,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        return session_id
    
    def get_session(self, session_id: str) -> Dict:
        """获取会话"""
        if session_id not in self.sessions:
            session_id = self.create_session(session_id)
        return self.sessions[session_id]
    
    async def chat_with_agent(self, message: str, history: List[List[str]], session_id: str = "default") -> Tuple[str, List[List[str]]]:
        """与代理进行对话"""
        if not message.strip():
            return "", history
        
        # 获取会话
        session = self.get_session(session_id)
        
        try:
            # 将用户消息添加到历史
            history.append([message, None])
            
            # 准备状态
            # 如果是新对话或者没有等待澄清，创建新的状态
            if session["conversation_state"] is None or not session["is_waiting_for_clarification"]:
                state = {
                    "messages": [HumanMessage(content=message)],
                    "initial_search_query_count": 3,
                    "max_research_loops": 2,
                    "reasoning_model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
                }
            else:
                # 如果正在等待澄清，将新消息添加到现有状态
                state = session["conversation_state"].copy()
                state["messages"].append(HumanMessage(content=message))
            
            # 执行图
            result = await graph.ainvoke(state)
            
            # 保存会话状态
            session["conversation_state"] = result
            session["last_result"] = result
            
            # 提取AI回复
            ai_response = ""
            if result.get("messages"):
                ai_response = result["messages"][-1].content
            
            # 检查是否需要澄清
            need_clarification = result.get("need_clarification", False)
            session["is_waiting_for_clarification"] = need_clarification
            
            # 如果需要澄清，添加提示
            if need_clarification:
                ai_response += "\n\n💡 *请提供更多信息以便我更好地帮助您*"
            else:
                # 添加额外信息
                extra_info = self._format_extra_info(result)
                if extra_info:
                    ai_response += f"\n\n---\n{extra_info}"
            
            # 更新历史记录
            history[-1][1] = ai_response
            
            # 保存到会话
            session["messages"].extend([
                {"role": "user", "content": message, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")},
                {"role": "assistant", "content": ai_response, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            ])
            
            return "", history
            
        except Exception as e:
            error_msg = f"❌ 处理消息时出错: {str(e)}"
            print(f"Error details: {traceback.format_exc()}")
            
            # 更新历史记录显示错误
            history[-1][1] = error_msg
            return "", history
    
    def _format_extra_info(self, result: Dict) -> str:
        """格式化额外信息"""
        info_parts = []
        
        # RAG信息
        if result.get("use_local_knowledge"):
            info_parts.append("🔗 **信息来源**: 本地知识库")
        elif result.get("rag_result"):
            info_parts.append("🔍 **检索状态**: 本地知识库 → 网络搜索")
        
        # 搜索查询
        if result.get("search_query"):
            queries = result["search_query"]
            if isinstance(queries, list) and queries:
                query_list = "\n".join([f"• {q}" for q in queries[:3]])
                info_parts.append(f"🔎 **搜索查询**:\n{query_list}")
        
        # 信息来源
        if result.get("sources_gathered"):
            sources = result["sources_gathered"][:3]  # 只显示前3个来源
            source_list = []
            for i, source in enumerate(sources, 1):
                title = source.get('title', '未知标题')[:50]
                url = source.get('value', source.get('url', ''))
                source_list.append(f"{i}. [{title}]({url})")
            
            if source_list:
                info_parts.append(f"📚 **参考来源**:\n" + "\n".join(source_list))
        
        return "\n\n".join(info_parts) if info_parts else ""
    
    def clear_session(self, session_id: str = "default"):
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        return []
    
    def get_session_info(self, session_id: str = "default") -> str:
        """获取会话信息"""
        session = self.get_session(session_id)
        
        info = f"""
## 📊 会话信息

- **会话ID**: {session_id}
- **创建时间**: {session.get('created_at', '未知')}
- **消息数量**: {len(session.get('messages', []))}
- **等待澄清**: {'是' if session.get('is_waiting_for_clarification', False) else '否'}
- **最后活动**: {session['messages'][-1]['timestamp'] if session.get('messages') else '无'}
        """
        
        return info.strip()
    
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


# 创建全局代理实例
chat_agent = ChatResearchAgent()


def create_chat_interface():
    """创建ChatGPT风格的对话界面"""
    
    with gr.Blocks(
        title="🤖 智能研究助手 - 对话模式",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.2em;
            font-weight: bold;
            margin-bottom: 15px;
        }
        .chat-container {
            height: 600px;
        }
        """
    ) as demo:
        
        # 会话ID状态
        session_id = gr.State("default")
        
        gr.HTML("""
        <div class="main-header">
            🤖 智能研究助手
        </div>
        <p style="text-align: center; font-size: 1.1em; color: #666;">
            基于 Qwen 模型的智能对话助手，支持意图澄清、本地知识库检索和实时网络搜索
        </p>
        """)
        
        with gr.Tab("💬 智能对话"):
            with gr.Row():
                with gr.Column(scale=4):
                    # 主对话界面
                    chatbot = gr.Chatbot(
                        label="对话历史",
                        height=500,
                        show_copy_button=True,
                        bubble_full_width=False,
                        avatar_images=("👤", "🤖")
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="",
                            placeholder="输入您的问题，我会首先确认理解您的意图，然后搜索本地知识库，必要时进行网络搜索...",
                            lines=2,
                            max_lines=5,
                            scale=4,
                            show_label=False
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)
                    
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
                        session_info_btn = gr.Button("📊 会话信息", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 💡 使用说明")
                    gr.Markdown("""
                    **对话流程**：
                    1. 🔍 意图澄清
                    2. 📋 生成研究简报  
                    3. 🔗 本地知识库检索
                    4. 🌐 网络搜索（如需要）
                    5. 📝 综合答案生成
                    
                    **特色功能**：
                    - 智能意图理解
                    - 多轮澄清对话
                    - 本地优先策略
                    - 来源可追溯
                    """)
                    
                    session_display = gr.Textbox(
                        label="会话信息",
                        lines=8,
                        interactive=False
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
        
        with gr.Tab("ℹ️ 使用说明"):
            gr.Markdown("""
            ## 🌟 智能对话助手特点
            
            ### 🤖 智能对话流程
            1. **意图澄清**: 自动分析问题是否需要澄清
            2. **研究规划**: 生成详细的研究简报
            3. **本地优先**: 首先搜索本地知识库
            4. **网络补充**: 必要时进行实时网络搜索
            5. **综合回答**: 整合多源信息生成答案
            
            ### 💬 对话交互
            - **多轮对话**: 支持连续对话和上下文理解
            - **智能澄清**: 自动识别模糊问题并主动澄清
            - **即时反馈**: 实时显示处理状态和信息来源
            - **会话管理**: 支持清空对话和查看会话信息
            
            ### 📚 知识库功能
            - **文档上传**: 支持 Markdown、TXT、PDF、DOCX
            - **智能检索**: 基于语义的向量搜索
            - **实时更新**: 上传后自动重建索引
            - **统计监控**: 查看知识库规模和状态
            
            ### 🔍 搜索策略
            - **本地优先**: 优先使用本地知识库
            - **智能判断**: 自动评估本地信息充分性
            - **网络补充**: 实时搜索最新信息
            - **来源追溯**: 提供详细的信息来源
            
            ### 💡 使用技巧
            1. **明确问题**: 具体的问题能获得更精准的答案
            2. **多轮交互**: 可以基于回答继续深入提问
            3. **澄清配合**: 遇到澄清问题时请提供详细信息
            4. **知识库建设**: 上传相关文档提升回答质量
            
            ### 📝 示例对话
            
            **用户**: "我想了解AI"
            **助手**: "您想了解AI的哪个方面呢？比如：基本概念、发展历史、技术原理、应用领域还是发展趋势？"
            
            **用户**: "我想了解AI在医疗领域的最新应用"
            **助手**: "我理解您想了解AI在医疗领域的最新应用，让我为您搜索相关信息..."
            
            ### ⚠️ 注意事项
            - 确保已正确配置API密钥
            - 本地知识库内容会影响回答质量
            - 网络搜索需要稳定的网络连接
            - 建议定期更新知识库内容
            """)
        
        # 事件绑定
        def handle_send(message, history, session_id):
            return chat_agent.chat_with_agent(message, history, session_id)
        def handle_send_async(message, history, session_id):
            """异步事件处理包装器"""
            return asyncio.run(chat_agent.chat_with_agent(message, history, session_id))

        
        def handle_clear(session_id):
            chat_agent.clear_session(session_id)
            return []
        
        def handle_session_info(session_id):
            return chat_agent.get_session_info(session_id)
        
        # 发送消息
        send_btn.click(
            fn=handle_send_async,
            inputs=[msg_input, chatbot, session_id],
            outputs=[msg_input, chatbot],
            show_progress=True
        )
        
        msg_input.submit(
            fn=handle_send,
            inputs=[msg_input, chatbot, session_id],
            outputs=[msg_input, chatbot],
            show_progress=True
        )
        
        # 清空对话
        clear_btn.click(
            fn=handle_clear,
            inputs=[session_id],
            outputs=[chatbot]
        )
        
        # 会话信息
        session_info_btn.click(
            fn=handle_session_info,
            inputs=[session_id],
            outputs=[session_display]
        )
        
        # 知识库管理事件
        upload_btn.click(
            fn=chat_agent.upload_document,
            inputs=[file_upload],
            outputs=[upload_result],
            show_progress=True
        )
        
        search_btn.click(
            fn=chat_agent.search_knowledge_base,
            inputs=[search_input, search_topk],
            outputs=[search_results],
            show_progress=True
        )
        
        stats_btn.click(
            fn=chat_agent.get_knowledge_base_stats,
            outputs=[kb_stats]
        )
        
        rebuild_btn.click(
            fn=chat_agent.rebuild_index,
            outputs=[kb_stats],
            show_progress=True
        )
    
    return demo


if __name__ == "__main__":
    # 初始化知识库
    try:
        print("🔄 正在初始化知识库...")
        knowledge_base.build_vector_store()
        print("✅ 知识库初始化完成")
    except Exception as e:
        print(f"⚠️ 知识库初始化失败: {e}")
    
    # 启动应用
    demo = create_chat_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False
    )
