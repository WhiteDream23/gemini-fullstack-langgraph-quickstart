#!/usr/bin/env python3
"""
简化的 Gradio 启动脚本
"""

import os
import sys

# 添加源代码路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

# 设置环境变量（如果需要）
os.environ.setdefault('PYTHONPATH', src_path)

if __name__ == "__main__":
    try:
        # 导入并运行 Gradio 应用
        from gradio_app import create_gradio_interface
        
        print("🚀 启动 Qwen 智能研究助手...")
        print("📍 访问地址: http://localhost:7860")
        print("💡 按 Ctrl+C 停止服务")
        print("-" * 50)
        
        demo = create_gradio_interface()
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            quiet=False,
            inbrowser=True  # 自动打开浏览器
        )
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        print("请检查:")
        print("1. 是否已配置环境变量 OPENAI_API_KEY 和 TAVILY_API_KEY")
        print("2. 是否已安装所有依赖包")
        print("3. 网络连接是否正常")
