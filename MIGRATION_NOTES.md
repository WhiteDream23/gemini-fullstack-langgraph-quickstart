# Qwen + Tavily Migration Notes

This project has been migrated from Google Gemini API to use Qwen models via ModelScope API and integrated with Tavily for real-time web search. Here are the key changes:

## 主要变化

### 1. API 提供商
- **旧**: Google Gemini API
- **新**: ModelScope API (for Qwen models) + Tavily API (for web search)

### 2. 环境变量
- **旧**: `GEMINI_API_KEY`
- **新**: 
  - `OPENAI_API_KEY` (用于 ModelScope API)
  - `OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1`
  - `TAVILY_API_KEY` (用于实时网络搜索)

### 3. 默认模型
- **旧**: 
  - `gemini-2.0-flash` (查询生成)
  - `gemini-2.5-flash` (反思)
  - `gemini-2.5-pro` (答案生成)
- **新**: 
  - `Qwen/Qwen3-30B-A3B-Instruct-2507` (所有任务)

### 4. 依赖项变化
移除的包:
- `langchain-google-genai`
- `google-genai`

添加的包:
- `langchain-openai`
- `openai`
- `tavily-python`

### 5. Web 搜索功能升级
- **旧**: 使用 Google Search API（通过 Gemini）
- **新**: 使用 Tavily Search API 进行真实的实时网络搜索
- **优势**: 
  - 独立的搜索服务，更稳定
  - 更好的搜索结果质量
  - 支持高级搜索选项
  - 真实的网络引用和链接

## 设置说明

1. 获取 ModelScope API 密钥:
   - 访问 https://modelscope.cn/
   - 注册账号并获取 API 密钥

2. 获取 Tavily API 密钥:
   - 访问 https://tavily.com/
   - 注册账号并获取 API 密钥

3. 更新环境变量:
   ```bash
   # 在 backend/.env 文件中
   OPENAI_API_KEY=your_modelscope_api_key_here
   OPENAI_BASE_URL=https://api-inference.modelscope.cn/v1
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

4. 安装依赖:
   ```bash
   cd backend
   pip install .
   ```

## 新功能

1. **真实的网络搜索**: 使用 Tavily API 进行实时网络搜索，获取最新信息。
2. **准确的引用**: 提供真实的网络链接和引用。
3. **高级搜索**: 支持搜索深度控制、域名过滤等高级功能。
4. **容错机制**: 如果网络搜索失败，会回退到基于知识库的回答。

## 配置选项

Tavily 搜索可以通过以下参数进行配置：
- `search_depth`: "basic" 或 "advanced"
- `max_results`: 最大搜索结果数量
- `include_domains`: 包含的域名列表
- `exclude_domains`: 排除的域名列表
- `include_answer`: 是否包含 Tavily 的直接回答

## 性能优化

1. **并发搜索**: 多个搜索查询可以并行执行
2. **结果缓存**: 搜索结果可以缓存以减少 API 调用
3. **智能分析**: 使用 Qwen 模型对搜索结果进行智能分析和总结
