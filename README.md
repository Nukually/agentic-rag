# agentic-rag

本项目当前是一个可运行的 **本地 Agentic RAG**：
- 支持 `.txt/.md/.pdf` 入库
- 向量检索 + reranker 重排
- 命令行对话问答（带引用）
- Agentic 工具链（`retrieve -> calculate`）与轨迹输出

## 当前已实现

- 文档解析：`pymupdf` 解析 PDF，文本统一分块
- 向量检索：Embedding API + Milvus（默认 `milvus-lite` 本地文件）
- 重排：OpenAI-style `/rerank` 接口（失败自动降级到纯向量结果）
- Agentic 执行：`plan -> act -> answer`
- 已接工具：
  - `retrieve`：检索知识库
  - `calculate`：基于检索到的变量做安全表达式计算

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
- `/exit`

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

## 测试

单元测试（包含 agentic 流程测试）：

```bash
python3 -m unittest tests/test_agentic_executor.py -v
```

## 关键文件

- Agent 执行器：`src/agent/graph.py`
- Planner：`src/agent/planner.py`
- Calculator 工具：`src/agent/tools/calculator.py`
- 检索工具：`src/agent/tools/rag_retrieve.py`
- CLI：`src/app/cli_chat.py`
- Agentic 演示脚本：`scripts/agentic_query_once.py`
- 测试样例文档：`knowledge/agentic_test_case.md`

## 规划文档

- `docs/multimodal_agentic_rag_plan.md`
