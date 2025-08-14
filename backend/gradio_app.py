import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import gradio as gr
import asyncio
from langchain_core.messages import HumanMessage
from agent.graph import graph
import json
from typing import List, Dict, Any
import time


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
            return "请输入一个有效的问题。", "", ""
        
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
            
            # 保存到对话历史
            self.conversation_history.append({
                "question": question,
                "answer": final_answer,
                "sources": sources,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return final_answer, formatted_sources, formatted_queries
            
        except Exception as e:
            error_msg = f"处理查询时出错: {str(e)}"
            return error_msg, "", ""
    
    def get_conversation_history(self):
        """获取对话历史"""
        if not self.conversation_history:
            return "暂无对话历史"
        
        history_text = []
        for i, conv in enumerate(self.conversation_history, 1):
            history_text.append(f"""
**对话 {i} ({conv['timestamp']})**
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
        title="Qwen 智能研究助手",
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
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="main-header">
            🔍 Qwen 智能研究助手
        </div>
        <p style="text-align: center; font-size: 1.1em; color: #666;">
            基于 Qwen 模型和 Tavily 搜索的智能研究工具，帮您快速获取准确、全面的信息
        </p>
        """)
        
        with gr.Tab("💭 智能问答"):
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="请输入您的问题",
                        placeholder="例如：什么是量子计算的最新发展？",
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
                            label="搜索来源",
                            lines=10,
                            max_lines=15,
                            show_copy_button=True
                        )
                    
                    with gr.Tab("🔍 搜索记录"):
                        queries_output = gr.Textbox(
                            label="执行的搜索查询",
                            lines=5,
                            max_lines=10
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
            
            ### 🤖 智能研究
            - 基于 **Qwen3-30B** 大语言模型
            - 集成 **Tavily** 实时网络搜索
            - 自动生成多角度搜索查询
            - 智能分析和信息整合
            
            ### 🔍 搜索能力
            - **实时网络搜索**：获取最新信息
            - **多源验证**：交叉验证信息准确性
            - **智能筛选**：过滤高质量内容
            - **来源追溯**：提供详细引用链接
            
            ### 💡 使用技巧
            1. **明确问题**：问题越具体，答案越精准
            2. **调整参数**：根据需要调整搜索深度
            3. **查看来源**：验证信息的可靠性
            4. **多轮对话**：基于前面结果继续深入
            
            ### 📝 示例问题
            - "2024年人工智能的最新突破是什么？"
            - "可再生能源技术的发展趋势如何？"
            - "量子计算在实际应用中有哪些进展？"
            - "区块链技术在金融领域的应用现状"
            
            ### ⚠️ 注意事项
            - 确保已配置 `OPENAI_API_KEY` 和 `TAVILY_API_KEY`
            - 搜索结果基于网络公开信息
            - 建议验证重要信息的准确性
            """)
        
        # 事件绑定
        submit_btn.click(
            fn=agent.research_query,
            inputs=[question_input, initial_queries, max_loops, reasoning_model],
            outputs=[answer_output, sources_output, queries_output],
            show_progress=True
        )
        
        question_input.submit(
            fn=agent.research_query,
            inputs=[question_input, initial_queries, max_loops, reasoning_model],
            outputs=[answer_output, sources_output, queries_output],
            show_progress=True
        )
        
        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            outputs=[question_input, answer_output, sources_output, queries_output]
        )
        
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
