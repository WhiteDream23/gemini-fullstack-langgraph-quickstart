# Qwen Fullstack LangGraph Quickstart

This project demonstrates an intelligent research application using Qwen models via ModelScope API and Tavily web search, powered by LangGraph. The application provides both a modern Gradio web interface and command-line interface for comprehensive research tasks.

<img src="./app.png" title="Qwen Fullstack LangGraph" alt="Qwen Fullstack LangGraph" width="90%">

## 🌟 Features

- 💬 **Modern Web UI**: Beautiful Gradio interface with real-time interaction
- 🧠 **Intelligent Research**: LangGraph-powered agent for advanced research workflows
- 🔍 **Smart Query Generation**: Dynamic search query generation using Qwen models
- 🌐 **Real-time Web Search**: Integrated with Tavily Search API for latest information
- 🤔 **Reflective Reasoning**: Intelligent knowledge gap analysis and iterative refinement
- 📄 **Comprehensive Answers**: Well-structured responses with real citations
- 📊 **Conversation History**: Track and review previous research sessions
- ⚙️ **Configurable Parameters**: Adjust search depth and reasoning complexity

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- **API Keys**:
  - **ModelScope API Key**: For accessing Qwen models
  - **Tavily API Key**: For real-time web search functionality

### 1. Setup Environment

1. **Clone and navigate to the project:**
   ```bash
   cd backend
   ```

2. **Create and configure environment file:**
   ```bash
   cp .env.example .env
   ```
   
3. **Edit `.env` file with your API keys:**
   ```env
   OPENAI_API_KEY="your_modelscope_api_key_here"
   OPENAI_BASE_URL="https://api-inference.modelscope.cn/v1"
   TAVILY_API_KEY="your_tavily_api_key_here"
   ```

### 2. Install Dependencies

```bash
cd backend
pip install .
```

### 3. Launch the Application

#### Option A: Gradio Web Interface (Recommended)

**Windows:**
```bash
# Double-click the batch file
start_gradio.bat

# Or run manually
python run_gradio.py
```

**Linux/Mac:**
```bash
python run_gradio.py
```

**Access the web interface at:** http://localhost:7860

#### Option B: Command Line Interface

```bash
# Set Python path
export PYTHONPATH="./src"  # Linux/Mac
# or
set PYTHONPATH=.\src       # Windows

python examples/cli_research.py "Your research question here"
```

#### Option C: LangGraph Development Server

```bash
langgraph dev
```

## 📋 Usage Examples

### Gradio Web Interface

**Access at:** http://localhost:7860

1. **Technology Research:**
   - "What are the latest developments in quantum computing?"
   - "How is AI being used in healthcare in 2024?"

2. **Market Analysis:**
   - "What are the current trends in renewable energy investments?"
   - "How has the cryptocurrency market evolved recently?"

3. **Academic Research:**
   - "What are the recent breakthroughs in gene therapy?"
   - "How is climate change affecting global agriculture?"

### CLI Examples

```bash
# Basic research
python examples/cli_research.py "What are the benefits of electric vehicles?"

# Advanced research with custom parameters
python examples/cli_research.py "Artificial intelligence trends" --initial-queries 5 --max-loops 3
```

### LangGraph Studio

Access the visual agent builder at: http://localhost:8123

## 🔧 Configuration

### Research Parameters

- **Initial Queries**: Number of initial search queries (1-5)
- **Max Research Loops**: Maximum iterative refinement cycles (1-5)
- **Reasoning Model**: Qwen model variant for analysis

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | ModelScope API key for Qwen models | Required |
| `OPENAI_BASE_URL` | ModelScope API endpoint | `https://api-inference.modelscope.cn/v1` |
| `TAVILY_API_KEY` | Tavily search API key | Required |
| `LANGSMITH_API_KEY` | LangSmith tracing (optional) | Optional |

## 🏗️ Architecture

The core of the backend is a LangGraph agent defined in `backend/src/agent/graph.py`. It follows these steps:

<img src="./agent.png" title="Agent Flow" alt="Agent Flow" width="50%">

1.  **Generate Initial Queries:** Based on your input, it generates a set of initial search queries using Qwen model via ModelScope.
2.  **Web Research:** For each query, it uses Tavily API to perform real-time web search and then analyzes the results using Qwen model.
3.  **Reflection & Knowledge Gap Analysis:** The agent analyzes the search results to determine if the information is sufficient or if there are knowledge gaps. It uses Qwen model for this reflection process.
4.  **Iterative Refinement:** If gaps are found or the information is insufficient, it generates follow-up queries and repeats the web research and reflection steps (up to a configured maximum number of loops).
5.  **Finalize Answer:** Once the research is deemed sufficient, the agent synthesizes the gathered information into a coherent answer with real citations from web sources, using Qwen model.

## CLI Example

For quick one-off questions you can execute the agent from the command line. The
script `backend/examples/cli_research.py` runs the LangGraph agent and prints the
final answer:

```bash
cd backend
python examples/cli_research.py "What are the latest trends in renewable energy?"
```


## Deployment

In production, the backend server serves the optimized static frontend build. LangGraph requires a Redis instance and a Postgres database. Redis is used as a pub-sub broker to enable streaming real time output from background runs. Postgres is used to store assistants, threads, runs, persist thread state and long term memory, and to manage the state of the background task queue with 'exactly once' semantics. For more details on how to deploy the backend server, take a look at the [LangGraph Documentation](https://langchain-ai.github.io/langgraph/concepts/deployment_options/). Below is an example of how to build a Docker image that includes the optimized frontend build and the backend server and run it via `docker-compose`.

_Note: For the docker-compose.yml example you need a LangSmith API key, you can get one from [LangSmith](https://smith.langchain.com/settings)._

_Note: If you are not running the docker-compose.yml example or exposing the backend server to the public internet, you should update the `apiUrl` in the `frontend/src/App.tsx` file to your host. Currently the `apiUrl` is set to `http://localhost:8123` for docker-compose or `http://localhost:2024` for development._

**1. Build the Docker Image:**

   Run the following command from the **project root directory**:
   ```bash
   docker build -t qwen-fullstack-langgraph -f Dockerfile .
   ```
**2. Run the Production Server:**

   ```bash
   OPENAI_API_KEY=<your_modelscope_api_key> OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1 TAVILY_API_KEY=<your_tavily_api_key> LANGSMITH_API_KEY=<your_langsmith_api_key> docker-compose up
   ```

Open your browser and navigate to `http://localhost:8123/app/` to see the application. The API will be available at `http://localhost:8123`.

## Technologies Used

- [React](https://reactjs.org/) (with [Vite](https://vitejs.dev/)) - For the frontend user interface.
- [Tailwind CSS](https://tailwindcss.com/) - For styling.
- [Shadcn UI](https://ui.shadcn.com/) - For components.
- [LangGraph](https://github.com/langchain-ai/langgraph) - For building the backend research agent.
- [Qwen Models](https://modelscope.cn/) - LLM for query generation, reflection, and answer synthesis via ModelScope API.
- [Tavily](https://tavily.com/) - Real-time web search API.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details. 
