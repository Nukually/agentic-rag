# agentic-rag

本项目当前是一个可运行的 **本地 Agentic RAG**：
- 支持 `.txt/.md/.pdf` 入库
- 向量检索 + reranker 重排
- 命令行对话问答（带引用）
- Agentic 工具链（`retrieve -> calculate`）与轨迹输出
- 多轮 memory 追问（第二轮可复用第一轮工具结果）
- 工具注册机制（可扩展 `web_search/sql/file`）

## 当前已实现

- 文档解析：`pymupdf` 解析 PDF，文本统一分块
- 向量检索：Embedding API + Milvus（默认 `milvus-lite` 本地文件）
- 重排：OpenAI-style `/rerank` 接口（失败自动降级到纯向量结果）
- Agentic 执行：`plan -> act -> answer`
- 已接工具：
  - `retrieve`：检索知识库
  - `calculate`：基于检索到的变量做安全表达式计算
- 记忆能力：
  - 记住上轮计算结果（`LAST_RESULT`）
  - 记住已提取变量（如 `Q1_PROFIT`）
  - 追问时可直接复用 memory，避免重复检索
- 工具机制：
  - 通过 `ToolRegistry` 注册工具
  - Agent 按规划步骤动态调用工具

## 快速开始

### 1. 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

### 2. 配置 `.env`

必填：
- `LLM_API_URL` `LLM_API_KEY` `LLM_MODEL`
- `EMBEDDING_API_URL` `EMBEDDING_API_KEY` `EMBEDDING_MODEL`

建议同时配置（已接入）：
- `RERANKER_API_URL` `RERANKER_API_KEY` `RERANKER_MODEL`

常用可选：
- `MILVUS_URI`（默认 `./data/index/milvus.db`）
- `MILVUS_COLLECTION`（默认 `rag_chunks`）
- `RAW_DATA_DIR`（默认 `./knowledge`）
- `RETRIEVAL_TOP_K`（默认 `4`）
- `RETRIEVAL_CANDIDATE_K`（默认 `12`）

### 3. 构建索引

```bash
python3 scripts/ingest_once.py
```

## 运行方式

### A. 普通 RAG 单次查看（最直观）

```bash
python3 scripts/query_once.py --question "这份文件主要讲什么？"
```

输出会显示：
- 向量召回结果
- 重排结果
- 最终回答
- 引用信息

### B. Agentic 单次演示（推荐看工具链）

```bash
python3 scripts/agentic_query_once.py --rebuild-index
```

默认会命中 `knowledge/agentic_test_case.md`，并触发：
- `retrieve`
- `calculate`

### C. 进入交互式对话

```bash
python3 -m src.app.cli_chat
```

首次可直接：

```bash
python3 -m src.app.cli_chat --rebuild-index
```

对话命令：
- `/rebuild`
- `/reset`
- `/tools`（查看已注册工具）
- `/memory`（查看当前 memory 摘要）
- `/exit`

### D. Streamlit 网页端（简洁 UI）

```bash
streamlit run src/app/streamlit_chat.py
```

网页端能力：
- 多轮聊天（复用同一个 Agent memory）
- 一键重建索引
- 会话清空 / memory 重置
- 每轮可展开查看工具轨迹、引用、memory 摘要

## 怎么触发 Agentic

最稳定的提问方式是“先检索再计算”的问题，例如：

```text
请根据 AGENTIC-CASE-ALPHA-OPS-2049 文档，计算 Q1_PROFIT + Q2_PROFIT - RD_COST 的值，并说明依据。
```

建议表达式写法：
- 变量大写：`Q1_PROFIT`
- 运算符两侧有空格：`Q1_PROFIT + Q2_PROFIT - RD_COST`

触发成功时会看到：
- `tool=retrieve`
- `tool=calculate`

## 多轮追问示例（memory 驱动）

第一轮：

```text
请根据 AGENTIC-CASE-ALPHA-OPS-2049 文档，计算 Q1_PROFIT + Q2_PROFIT - RD_COST
```

第二轮追问：

```text
把刚才结果再加 10
```

你会看到第二轮轨迹通常是：
- `tool=calculate`
- 输入表达式变成 `LAST_RESULT + 10`

说明 Agent 已复用上一轮结果，不需要再次检索。

## 工具注册机制（扩展入口）

核心结构：
- `src/agent/tools/registry.py`
- `ToolRegistry.register(tool)`
- 工具实现约定：提供 `name` 和 `run(tool_input, context) -> ToolOutput`

当前默认注册工具在：
- `src/agent/graph.py` 中 `AgentExecutor.__init__`

已实现工具：
- `src/agent/tools/retrieve_tool.py`
- `src/agent/tools/calculate_tool.py`

后续新增 `web_search/sql/file` 时，直接新增工具类并在 registry 注册即可。

## 测试

单元测试（包含 agentic 流程测试）：

```bash
python3 -m unittest tests/test_agentic_executor.py -v
```

## 关键文件

- Agent 执行器：`src/agent/graph.py`
- Memory：`src/agent/memory.py`
- Planner：`src/agent/planner.py`
- Calculator 工具：`src/agent/tools/calculator.py`
- Tool Registry：`src/agent/tools/registry.py`
- 检索工具：`src/agent/tools/rag_retrieve.py`
- Retrieve Tool：`src/agent/tools/retrieve_tool.py`
- Calculate Tool：`src/agent/tools/calculate_tool.py`
- CLI：`src/app/cli_chat.py`
- Agentic 演示脚本：`scripts/agentic_query_once.py`
- 测试样例文档：`knowledge/agentic_test_case.md`

## 规划文档

- `docs/multimodal_agentic_rag_plan.md`
