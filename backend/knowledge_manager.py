#!/usr/bin/env python3
"""
知识库管理脚本
用于管理本地知识库：添加文档、构建索引、查询测试等
"""

import os
import sys
import argparse
from pathlib import Path

# 添加源代码路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

from agent.knowledge_base import knowledge_base


def add_document(file_path: str):
    """添加文档到知识库"""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    try:
        # 复制文件到知识库目录
        knowledge_dir = knowledge_base.knowledge_dir
        target_path = knowledge_dir / file_path.name
        
        # 确保目标目录存在
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        import shutil
        shutil.copy2(file_path, target_path)
        
        print(f"✅ 文件已添加到知识库: {target_path}")
        
        # 重建向量索引
        print("🔄 重建向量索引...")
        knowledge_base.build_vector_store(force_rebuild=True)
        print("✅ 向量索引重建完成")
        
    except Exception as e:
        print(f"❌ 添加文档失败: {e}")


def build_index():
    """构建向量索引"""
    try:
        print("🔄 开始构建向量索引...")
        knowledge_base.build_vector_store(force_rebuild=True)
        print("✅ 向量索引构建完成")
    except Exception as e:
        print(f"❌ 构建索引失败: {e}")


def search_knowledge(query: str, top_k: int = 5):
    """搜索知识库"""
    try:
        print(f"🔍 搜索: {query}")
        results = knowledge_base.search(query, top_k=top_k)
        
        if not results:
            print("❌ 没有找到相关结果")
            return
        
        print(f"\n📚 找到 {len(results)} 个相关结果:")
        for i, result in enumerate(results, 1):
            content = result['content']
            metadata = result.get('metadata', {})
            source = metadata.get('source_file', '未知来源')
            similarity = result.get('similarity', 0)
            
            print(f"\n【结果 {i}】")
            print(f"来源: {source}")
            print(f"相似度: {similarity:.3f}")
            print(f"内容: {content[:200]}{'...' if len(content) > 200 else ''}")
            print("-" * 50)
            
    except Exception as e:
        print(f"❌ 搜索失败: {e}")


def show_stats():
    """显示知识库统计信息"""
    try:
        stats = knowledge_base.get_stats()
        print("📊 知识库统计信息:")
        print(f"  总文档块数: {stats['total_chunks']}")
        print(f"  索引大小: {stats['index_size']}")
        print(f"  知识库目录: {stats['knowledge_dir']}")
        print(f"  向量存储目录: {stats['vector_store_path']}")
        
        # 检查知识库目录中的文件
        knowledge_dir = Path(stats['knowledge_dir'])
        if knowledge_dir.exists():
            files = list(knowledge_dir.rglob("*"))
            doc_files = [f for f in files if f.is_file() and f.suffix.lower() in ['.md', '.txt', '.pdf', '.docx']]
            print(f"  知识库文件数: {len(doc_files)}")
            
            if doc_files:
                print("  文件列表:")
                for file in doc_files[:10]:  # 只显示前10个文件
                    print(f"    - {file.name}")
                if len(doc_files) > 10:
                    print(f"    ... 和其他 {len(doc_files) - 10} 个文件")
        
    except Exception as e:
        print(f"❌ 获取统计信息失败: {e}")


def create_sample_docs():
    """创建示例文档"""
    knowledge_dir = knowledge_base.knowledge_dir
    
    sample_docs = [
        {
            "filename": "ai_introduction.md",
            "content": """# 人工智能简介

## 什么是人工智能

人工智能（Artificial Intelligence，AI）是指让机器表现出人类智能特征的技术和方法。

## 主要分类

### 1. 机器学习
机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习和改进。

### 2. 深度学习
深度学习是机器学习的一个子集，它模仿人脑神经网络的结构和功能。

### 3. 自然语言处理
自然语言处理（NLP）使计算机能够理解、解释和生成人类语言。

## 应用领域

- 图像识别
- 语音识别
- 推荐系统
- 自动驾驶
- 医疗诊断
"""
        },
        {
            "filename": "machine_learning_basics.md",
            "content": """# 机器学习基础

## 监督学习

监督学习使用标记的训练数据来学习输入和输出之间的映射关系。

### 常见算法
- 线性回归
- 逻辑回归
- 决策树
- 随机森林
- 支持向量机

## 无监督学习

无监督学习在没有标记数据的情况下发现数据中的隐藏模式。

### 常见算法
- K-means聚类
- 层次聚类
- 主成分分析（PCA）
- DBSCAN

## 强化学习

强化学习通过与环境交互来学习最优行为策略。

### 核心概念
- 智能体（Agent）
- 环境（Environment）
- 状态（State）
- 动作（Action）
- 奖励（Reward）
"""
        },
        {
            "filename": "deep_learning_overview.md",
            "content": """# 深度学习概述

## 神经网络基础

人工神经网络是深度学习的基础，它模仿生物神经网络的结构。

### 基本组件
- 神经元（节点）
- 权重和偏置
- 激活函数
- 层（输入层、隐藏层、输出层）

## 常见网络架构

### 1. 前馈神经网络
最简单的神经网络类型，信息只向前传播。

### 2. 卷积神经网络（CNN）
主要用于图像处理和计算机视觉任务。

### 3. 循环神经网络（RNN）
适合处理序列数据，如文本和时间序列。

### 4. 变压器（Transformer）
现代自然语言处理的主流架构。

## 训练过程

1. 前向传播
2. 计算损失
3. 反向传播
4. 参数更新
"""
        }
    ]
    
    try:
        for doc in sample_docs:
            file_path = knowledge_dir / doc["filename"]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc["content"])
            print(f"✅ 创建示例文档: {doc['filename']}")
        
        print("🔄 构建向量索引...")
        knowledge_base.build_vector_store(force_rebuild=True)
        print("✅ 示例知识库创建完成！")
        
    except Exception as e:
        print(f"❌ 创建示例文档失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="知识库管理工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 添加文档命令
    add_parser = subparsers.add_parser('add', help='添加文档到知识库')
    add_parser.add_argument('file_path', help='要添加的文件路径')
    
    # 构建索引命令
    subparsers.add_parser('build', help='构建向量索引')
    
    # 搜索命令
    search_parser = subparsers.add_parser('search', help='搜索知识库')
    search_parser.add_argument('query', help='搜索查询')
    search_parser.add_argument('--top-k', type=int, default=5, help='返回结果数量')
    
    # 统计信息命令
    subparsers.add_parser('stats', help='显示知识库统计信息')
    
    # 创建示例文档命令
    subparsers.add_parser('init', help='创建示例知识库')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print("🚀 知识库管理工具")
    print("-" * 40)
    
    if args.command == 'add':
        add_document(args.file_path)
    elif args.command == 'build':
        build_index()
    elif args.command == 'search':
        search_knowledge(args.query, args.top_k)
    elif args.command == 'stats':
        show_stats()
    elif args.command == 'init':
        create_sample_docs()


if __name__ == "__main__":
    main()
