import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from tavily import TavilyClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
    RAGState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from langchain_openai import ChatOpenAI
from agent.utils import (
    get_research_topic,
)
from agent.rag_tools import create_rag_tool, evaluate_rag_sufficiency

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
def local_rag_search(state: OverallState, config: RunnableConfig) -> RAGState:
    """本地 RAG 检索节点
    
    首先在本地知识库中搜索相关信息，如果找到足够的信息，
    则可以直接基于本地知识回答，否则继续网络搜索流程。
    """
    configurable = Configuration.from_runnable_config(config)
    
    # 获取用户问题
    user_question = get_research_topic(state["messages"])
    
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
    agent_executor = create_react_agent(llm, [retrieve_tool], checkpointer=memory)
    
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
    user_question = get_research_topic(state["messages"])
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

当前日期: {get_current_date()}
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
    """LangGraph node that generates search queries based on the User's question.

    当本地 RAG 检索不足时，使用 Qwen 模型创建网络搜索查询。

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
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
    current_date = get_current_date()
    formatted_prompt = query_writer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
        number_queries=state["initial_search_query_count"],
    )
    # Generate the search queries
    result = structured_llm.invoke(formatted_prompt)
    return {"search_query": result.query}


def continue_to_web_research(state: QueryGenerationState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    return [
        Send("web_research", {"search_query": search_query, "id": int(idx)})
        for idx, search_query in enumerate(state["search_query"])
    ]


def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using Tavily Search API.

    Executes a web search using Tavily API and then uses Qwen model to analyze
    and summarize the search results.

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search API settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
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
            current_date=get_current_date(),
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
            current_date=get_current_date(),
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
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = reflection_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
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
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
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
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model

    # Format the prompt
    current_date = get_current_date()
    formatted_prompt = answer_instructions.format(
        current_date=current_date,
        research_topic=get_research_topic(state["messages"]),
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


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("local_rag_search", local_rag_search)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)
builder.add_node("finalize_answer_with_rag", finalize_answer_with_rag)

# Set the entrypoint as `local_rag_search`
# 首先进行本地 RAG 检索
builder.add_edge(START, "local_rag_search")

# 根据 RAG 结果决定下一步
builder.add_conditional_edges(
    "local_rag_search", 
    evaluate_rag_result, 
    ["generate_query", "finalize_answer_with_rag"]
)

# 如果 RAG 不充分，继续原有的搜索流程
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)
builder.add_edge("finalize_answer_with_rag", END)

graph = builder.compile(name="rag-enhanced-search-agent")

# 保存图形结构
try:
    png_data = graph.get_graph().draw_mermaid_png()
    with open("langgraph_structure.png", "wb") as f:
        f.write(png_data)
    print("✅ 图形结构已保存为 'langgraph_structure.png'")
except Exception as e:
    print(f"⚠️  无法生成图形: {e}")