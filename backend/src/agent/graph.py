import os
import asyncio
from agent.tools_and_schemas import SearchQueryList, Reflection,ClarifyWithUser,ResearchQuestion
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from tavily import TavilyClient
from typing_extensions import Literal
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command,interrupt
from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
    RAGState,
)
from agent.configuration import Configuration
from agent.prompts import (
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
    clarify_with_user_instructions,
    transform_messages_into_research_topic_prompt,
)

from langchain_openai import ChatOpenAI
from agent.utils import (
    get_research_topic,
    get_today_str,
    get_buffer_string,
)

from agent.agent_tools import create_rag_tool, evaluate_rag_sufficiency,save_file

load_dotenv()

if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY is not set")

if os.getenv("TAVILY_API_KEY") is None:
    raise ValueError("TAVILY_API_KEY is not set")

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Initialize memory for the RAG agent
memory = MemorySaver()


# Nodes

def clarify_with_user(state: OverallState, config: RunnableConfig)  -> Command[Literal["write_research_brief", "__end__"]]:
    """
    用户意图澄清节点
    
    分析用户的问题是否包含足够信息进行研究，
    如果不够明确则生成澄清问题，否则确认理解。
    """
    configurable = Configuration.from_runnable_config(config)
    
    # 初始化模型
    llm = ChatOpenAI(
        model=configurable.reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
    )
    
    # 设置结构化输出
    structured_output_model = llm.with_structured_output(ClarifyWithUser)
    
    # 调用模型进行意图澄清
    response = structured_output_model.invoke([
        HumanMessage(content=clarify_with_user_instructions.format(
            messages=get_buffer_string(messages=state.get("messages", [])), 
            date=get_today_str()
        ))
    ])
    
    # 更新状态
    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)],
                    "need_clarification": True,}
        )
    else:
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)],
                     "need_clarification": False}
        )


def write_research_brief(state: OverallState, config: RunnableConfig) -> OverallState:
    """
    研究简报生成节点
    
    将对话历史转换为详细的研究简报，
    为后续的RAG检索和网络搜索提供明确指导。
    """
    configurable = Configuration.from_runnable_config(config)
    
    # 初始化模型
    llm = ChatOpenAI(
        model=configurable.reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
    )
    
    # 设置结构化输出
    structured_output_model = llm.with_structured_output(ResearchQuestion)
    
    # 生成研究简报
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])
    
    # 更新状态
    return {
        "research_brief": response.research_brief,
        "supervisor_messages": [HumanMessage(content=f"研究简报：{response.research_brief}")],
        "messages": [AIMessage(content=f"📋 研究简报已生成：\n\n{response.research_brief}")]
    }


def local_rag_search(state: OverallState, config: RunnableConfig) -> RAGState:
    """本地 RAG 检索节点
    
    首先在本地知识库中搜索相关信息，如果找到足够的信息，
    则可以直接基于本地知识回答，否则继续网络搜索流程。
    """
    configurable = Configuration.from_runnable_config(config)
    
    # 获取用户问题
    user_question = get_research_topic(state["research_brief"])
    
    # 创建 LLM 实例
    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
    )
    
    # 创建 RAG 工具
    retrieve_tool = create_rag_tool()
    
    # 创建 React Agent
    agent_executor = create_react_agent(llm, [retrieve_tool])
    
    try:
        # 执行 RAG 检索
        rag_prompt = f"""请使用本地知识库检索工具回答以下问题：{user_question}
        
如果本地知识库中有相关信息，请基于检索到的内容提供详细回答。
如果本地知识库中没有足够的相关信息，请明确说明需要进一步的网络搜索。"""
        
        result = agent_executor.invoke(
            {"messages": [HumanMessage(content=rag_prompt)]},
            config={"configurable": {"thread_id": "rag_search"}}
        )
        
        # 提取结果
        rag_result = ""
        if result.get("messages"):
            rag_result = result["messages"][-1].content
        
        # 评估 RAG 结果的充分性
        evaluation = evaluate_rag_sufficiency(rag_result, user_question)
        
        return {
            "rag_result": rag_result,
            "rag_sufficient": evaluation["is_sufficient"],
            "rag_confidence": evaluation["confidence"],
            "evaluation_reason": evaluation["reason"],
            "use_local_knowledge": evaluation["is_sufficient"]
        }
        
    except Exception as e:
        return {
            "rag_result": f"本地 RAG 检索失败: {str(e)}",
            "rag_sufficient": False,
            "rag_confidence": 0.0,
            "evaluation_reason": "RAG 检索过程中出现错误",
            "use_local_knowledge": False
        }


def evaluate_rag_result(state: OverallState) -> str:
    """评估 RAG 结果，决定下一步流程"""
    if state.get("rag_sufficient", False) and state.get("rag_confidence", 0) > 0.5:
        return "finalize_answer_with_rag"
    else:
        return "generate_query"


def finalize_answer_with_rag(state: OverallState, config: RunnableConfig):
    """基于 RAG 结果生成最终答案"""
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model
    
    # 获取用户问题和 RAG 结果
    user_question = get_research_topic(state["research_brief"])
    rag_content = state.get("rag_result", "")
    
    # 创建基于 RAG 的回答提示
    rag_answer_prompt = f"""
基于以下本地知识库的检索结果，请为用户问题提供详细、准确的回答。

用户问题: {user_question}

本地知识库检索结果:
{rag_content}

请注意:
1. 基于提供的本地知识库内容进行回答
2. 如果信息不完整，请说明已知信息的局限性
3. 保持回答的准确性和相关性
4. 适当引用来源信息

当前日期: {get_today_str()}
"""
    
    # 初始化 LLM
    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
    )
    
    result = llm.invoke(rag_answer_prompt)
    
    # 添加 RAG 来源标注
    final_answer = f"{result.content}\n\n---\n💡 此回答基于本地知识库内容生成"
    
    return {
        "messages": [AIMessage(content=final_answer)],
        "sources_gathered": [],  # RAG 不提供网络来源
    }


def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """基于用户问题生成搜索查询的 LangGraph 节点。

    当本地 RAG 检索不足时，使用 Qwen 模型创建网络搜索查询。

    Args:
        state: 包含用户问题的当前图状态
        config: 可运行对象的配置，包括 LLM 提供商设置

    Returns:
        包含状态更新的字典，包括含有生成查询的 search_query 键
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    # init Qwen model via ModelScope
    llm = ChatOpenAI(
        model=configurable.query_generator_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
        extra_body={"enable_thinking": False},
    )
    structured_llm = llm.with_structured_output(SearchQueryList)

    # Format the prompt
    current_date = get_today_str()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["research_brief"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """将搜索查询发送到网络研究节点的 LangGraph 节点。

    用于生成 n 个网络研究节点，每个搜索查询对应一个节点。
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """使用 Tavily 搜索 API 执行网络研究的 LangGraph 节点。

    使用 Tavily API 执行网络搜索，然后使用 Qwen 模型分析
    和总结搜索结果。

    Args:
        state: 包含搜索查询和研究循环计数的当前图状态
        config: 可运行对象的配置，包括搜索 API 设置

    Returns:
        包含状态更新的字典，包括 sources_gathered、research_loop_count 和 web_research_results
    """
    # Configure
    configurable = Configuration.from_runnable_config(config)
    search_query = state["search_query"]
    
    try:
        # Perform web search using Tavily
        search_results = tavily_client.search(
            query=search_query,
            search_depth="advanced",
            max_results=5,
            include_domains=None,
            exclude_domains=None,
            include_answer=True,
            include_raw_content=False,
            include_images=False
        )
        
        # Extract relevant information from search results
        search_content = []
        sources_gathered = []
        
        if search_results.get("results"):
            for idx, result in enumerate(search_results["results"]):
                source_info = {
                    "short_url": f"[source_{state['id']}_{idx}]",
                    "value": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", "")
                }
                sources_gathered.append(source_info)
                
                # Format content for analysis
                search_content.append(f"来源: {result.get('title', 'Unknown')}\n"
                                    f"URL: {result.get('url', '')}\n"
                                    f"内容: {result.get('content', '')}")
        
        # Use Qwen model to analyze and synthesize the search results
        llm = ChatOpenAI(
            model=configurable.query_generator_model,
            temperature=0,
            max_retries=2,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
            extra_body={"enable_thinking": False},
        )
        
        # Enhanced prompt for analysis
        analysis_prompt = f"""
        {web_searcher_instructions.format(
            current_date=get_today_str(),
            research_topic=search_query,
        )}
        
        以下是搜索到的相关信息：
        
        {chr(10).join(search_content)}
        
        请基于以上搜索结果，提供关于"{search_query}"的全面分析和总结。
        请在回答中引用相关的源链接，格式为 [source_{state['id']}_X]。
        """
        
        response = llm.invoke(analysis_prompt)
        modified_text = response.content
        
        # Add Tavily's direct answer if available
        if search_results.get("answer"):
            modified_text = f"快速回答: {search_results['answer']}\n\n详细分析:\n{modified_text}"
        
    except Exception as e:
        # Fallback to knowledge-based response if Tavily fails
        print(f"Tavily搜索失败: {e}")
        
        llm = ChatOpenAI(
            model=configurable.query_generator_model,
            temperature=0,
            max_retries=2,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
            extra_body={"enable_thinking": False},
        )
        
        fallback_prompt = f"""
        {web_searcher_instructions.format(
            current_date=get_today_str(),
            research_topic=search_query,
        )}
        
        注意：由于网络搜索服务暂时不可用，请基于您的知识库提供关于"{search_query}"的详细信息。
        """
        
        response = llm.invoke(fallback_prompt)
        modified_text = f"[基于知识库回答] {response.content}"
        sources_gathered = []

    return {
        "sources_gathered": sources_gathered,
        "search_query": [search_query],
        "web_research_result": [modified_text],
    }


def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """识别知识空白并生成潜在后续查询的 LangGraph 节点。

    分析当前摘要以识别需要进一步研究的领域，并生成
    潜在的后续查询。使用结构化输出以 JSON 格式提取
    后续查询。

    Args:
        state: 包含运行摘要和研究主题的当前图状态
        config: 可运行对象的配置，包括 LLM 提供商设置

    Returns:
        包含状态更新的字典，包括含有生成的后续查询的 search_query 键
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_today_str()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["research_brief"]),
        summaries="\n\n---\n\n".join(state["web_research_result"]),
    )
    # init Qwen Model
    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=1.0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
        extra_body={"enable_thinking": False},
    )
    result = llm.with_structured_output(Reflection).invoke(formatted_prompt)

    return {
        "is_sufficient": result.is_sufficient,
        "knowledge_gap": result.knowledge_gap,
        "follow_up_queries": result.follow_up_queries,
        "research_loop_count": state["research_loop_count"],
        "number_of_ran_queries": len(state["search_query"]),
    }


def evaluate_research(
    state: ReflectionState,
    config: RunnableConfig,
) -> OverallState:
    """决定研究流程下一步的 LangGraph 路由函数。

    通过决定是否继续收集信息或基于配置的最大研究循环次数
    完成摘要来控制研究循环。

    Args:
        state: 包含研究循环计数的当前图状态
        config: 可运行对象的配置，包括 max_research_loops 设置

    Returns:
        指示下一个要访问的节点的字符串字面量（"web_research" 或 "finalize_summary"）
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    if state["is_sufficient"] or state["research_loop_count"] >= max_research_loops:
        return "finalize_answer"
    else:
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state["number_of_ran_queries"] + int(idx),
                },
            )
            for idx, follow_up_query in enumerate(state["follow_up_queries"])
        ]


def finalize_answer(state: OverallState, config: RunnableConfig):
    """完成研究摘要的 LangGraph 节点。

    通过去重和格式化来源来准备最终输出，然后
    将它们与运行摘要结合以创建结构良好的
    带有适当引用的研究报告。

    Args:
        state: 包含运行摘要和收集来源的当前图状态

    Returns:
        包含状态更新的字典，包括含有格式化的最终摘要和来源的 running_summary 键
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_today_str()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["research_brief"]),
        summaries="\n---\n\n".join(state["web_research_result"]),
    )

    # init Qwen Model, default to Qwen3-30B
    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
    )
    result = llm.invoke(formatted_prompt)

    # Replace the short urls with the original urls and add all used urls to the sources_gathered
    unique_sources = []
    for source in state["sources_gathered"]:
        if source["short_url"] in result.content:
            result.content = result.content.replace(
                source["short_url"], source["value"]
            )
            unique_sources.append(source)

    return {
        "messages": [AIMessage(content=result.content)],
        "sources_gathered": unique_sources,
    }



async def save_file_node(state: OverallState, config: RunnableConfig) -> OverallState:
    """将研究报告保存到文件的 LangGraph 节点。

    Args:
        state: 包含运行摘要和收集来源的当前图状态

    Returns:
        指示工作流结束的命令
    """
    # Implement file saving logic here
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # init Qwen Model, default to Qwen3-30B
    llm = ChatOpenAI(
        model=reasoning_model,
        temperature=0,
        max_retries=2,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api-inference.modelscope.cn/v1"),
    )
    client = MultiServerMCPClient(
    {
        "GITHUB": {
            "url": os.getenv("GITHUB_TOKEN_KEY"),
            "transport": "sse"
        }

    }
)
    tools = await asyncio.wait_for(client.get_tools(), timeout=25)
    tools.append(save_file)
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)

    try:
        # 执行 RAG 检索
        save_prompt = f"""请将下面的内容保存到knowledge文件夹中：{state['messages'][-1].content},保存文件名为{state['research_brief'][:10]}.md,
        同时在我的mydeepresearch_note github仓库中创建相同的文件
        """
        
        result = await agent_executor.ainvoke(
            {"messages": [HumanMessage(content=save_prompt)]},
        )
        
        # 提取结果
        save_result = ""
        if result.get("messages"):
            save_result = result["messages"][-1].content
        
        return {
            "save_result": save_result,
        }
        
    except Exception as e:
        return {
            "save_result": f"本地保存文件检索失败: {str(e)}",
        }




def ask_save_permission(state: OverallState) -> Command[Literal["save_file", "__end__"]]:
    """处理用户的保存决定"""
    state["awaiting_user_decision"] = True
    save_permission = interrupt({
        "question": "是否保存文件?"
    })
    state["awaiting_user_decision"] = False
    if save_permission["save_permission"] == "save":
        return Command(goto="save_file", update={"save_permission": "save"})
    else:
        return Command(goto="__end__", update={"save_permission": "skip"})              


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define all nodes in the workflow
builder.add_node("clarify_with_user", clarify_with_user)
builder.add_node("write_research_brief", write_research_brief)
builder.add_node("local_rag_search", local_rag_search)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)
builder.add_node("ask_save_permission", ask_save_permission)
builder.add_node("save_file", save_file_node)
builder.add_node("finalize_answer_with_rag", finalize_answer_with_rag)

# 新的工作流程：
# 1. 首先进行意图澄清
builder.add_edge(START, "clarify_with_user")

builder.add_edge(
    "write_research_brief",
    "local_rag_search"
)

# 5. 本地 RAG 检索（原有逻辑保持不变）
builder.add_conditional_edges(
    "local_rag_search", 
    evaluate_rag_result, 
    ["generate_query", "finalize_answer_with_rag"]
)

# 6. 网络搜索流程（原有逻辑保持不变）
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
builder.add_edge("web_research", "reflection")
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)

# 7. 结束节点
builder.add_edge("finalize_answer", "ask_save_permission")

builder.add_edge("save_file", END)
builder.add_edge("finalize_answer_with_rag", END)

graph = builder.compile(name="intent-clarification-rag-enhanced-search-agent",checkpointer=memory)

# 保存图形结构
try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("langgraph_structure.png", "wb") as f:
        f.write(png_data)
    print("✅ 图形结构已保存为 'langgraph_structure.png'")
except Exception as e:
    print(f"⚠️  无法生成图形: {e}")