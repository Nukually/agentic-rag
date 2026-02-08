# agentic-rag

本项目是一个可运行的 **本地 Agentic RAG**（当前仅保留 CLI）：
- 支持 `.txt/.md/.pdf` 入库与向量化检索
- 向量检索 + 可选 reranker 重排
- Agentic 工具链（`retrieve -> calculate / budget_analyst`）与轨迹输出
- 多轮 memory 追问（复用上一轮变量与结果）
- Router Chain：先分类再决定是否进入检索链路
- 工具注册机制（可扩展 `web_search/sql/file`）
- 预算分析师工具：根据年度预算做股价评级（`budget_analyst`）

---

## 1. 系统完整运行逻辑（从输入到输出）

简化流程如下：

```
用户问题
  -> Router 分类（闲聊 / 需要查询知识库 / 其他）
    -> Planner 生成工具计划（0~4 步）
      -> Tool 执行（retrieve / calculate）
        -> 汇总引用 + 轨迹
          -> 最终回答
```

关键细节：
- **Router Chain**：优先判断问题类型，闲聊不进检索链路，避免性能浪费。
- **Planner**：根据问题/记忆决定是否需要检索或计算，最多 4 步。
- **Tool 执行**：
  - `retrieve`：向量检索 + 可选 rerank，写入 `memory`。
  - `calculate`：从检索文本中提取变量 + 安全表达式计算。
  - `budget_analyst`：从年度预算与股价中生成增速、评分、评级。
- **Memory**：保存上轮变量、计算结果、引用，支持追问复用。
- **Answer**：根据轨迹与上下文生成最终答案（无上下文时不会乱编）。

---

## 2. 安装与配置

### 2.1 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

### 2.2 配置 `.env`

必填：
- `LLM_API_URL` `LLM_API_KEY` `LLM_MODEL`
- `EMBEDDING_API_URL` `EMBEDDING_API_KEY` `EMBEDDING_MODEL`

建议配置（已接入 rerank）：
- `RERANKER_API_URL` `RERANKER_API_KEY` `RERANKER_MODEL`

常用可选：
- `MILVUS_URI`（默认 `./data/index/milvus.db`）
- `MILVUS_COLLECTION`（默认 `rag_chunks`）
- `RAW_DATA_DIR`（默认 `./knowledge`）
- `RETRIEVAL_TOP_K`（默认 `4`）
- `RETRIEVAL_CANDIDATE_K`（默认 `12`）

---

## 3. 用户使用指南（逐步）

### 步骤 1：准备文档
把待检索文件放入 `knowledge/`，支持 `.txt/.md/.pdf`。

示例：
```bash
cp ~/docs/report.pdf knowledge/
```

### 步骤 2：构建索引
```bash
python3 scripts/ingest_once.py
```

### 步骤 3：进入 CLI 对话
方式一（推荐，自动走 conda 环境）：
```bash
scripts/run_cli.sh --rebuild-index
```

方式二（手动激活 conda 环境后运行）：
```bash
source /home/nuku/miniconda3/bin/activate agentic-rag
python3 -m src.app.cli_chat --rebuild-index
```

方式三（不激活环境，直接用 conda run）：
```bash
/home/nuku/miniconda3/bin/conda run -n agentic-rag python -m src.app.cli_chat --rebuild-index
```

### 步骤 4：提问
输入问题即可。系统会根据 Router 分类决定是否检索。

示例：
```
你能做什么？
```
（不会触发检索）

```
请根据 688230_20260203_JJZK.pdf 的内容总结经营亮点
```
（会触发检索）

---

## 4. 功能说明（含样例）

### 4.1 Router Chain（分类分流）
**功能**：将问题分为“闲聊 / 需要查询知识库 / 其他”，决定是否进入 RAG。

示例：
- 输入：`你好`
  - 分类：闲聊 → **不检索**
- 输入：`请根据文档说明营业收入变化原因`
  - 分类：需要查询知识库 → **触发检索**

实现方式（怎么实现的）：
- 系统先把你的问题原样交给“路由提示词”，让模型只输出一个标签：**闲聊 / 需要查询知识库 / 其他**。
- 如果结果是“闲聊”，就**直接走聊天回答**，不会浪费时间检索文档。
- 如果结果是“需要查询知识库”，才会进入“规划器 + 工具链”流程。
- 如果结果是“其他”，会走简化的通用回答，不强制检索。
- 这一步只是**分流**，不做任何计算或检索；它的价值是节省成本、避免无关检索。

代码位置（可选）：
`src/agent/planner.py`（路由逻辑）、`src/llm/prompts.py`（路由提示词）、`src/agent/graph.py`（分流后选择路径）

### 4.2 检索与重排（retrieve）
**功能**：向量检索文档片段，必要时使用 rerank 重排。

示例问题：
```
请从 688230_20260203_JJZK.pdf 找到“净利润”相关描述
```
输出包含：
- 检索片段
- 引用来源与页码
- rerank 信息（若启用）

实现方式（怎么实现的）：
- 系统先把“用户问题”或“指定检索关键词”转成向量（embedding），这一步是把文本变成可比较的数字。
- 用这个向量去向量库里找“最相近”的文档片段（相似度高的先返回）。
- 如果配置了 reranker，就把初步结果再交给重排模型，按“更相关”的顺序重新排序。
- 最终结果会被写进 memory（包括检索到的原文片段、来源和页码），后续问题可直接复用。
- 工具输出会生成一段“检索轨迹”，最终答案会引用这些片段，避免胡编。

代码位置（可选）：
`src/agent/tools/retrieve_tool.py`、`src/agent/tools/rag_retrieve.py`

### 4.3 计算工具（calculate）
**功能**：对表达式进行安全计算，变量来自检索文本或 memory。

示例问题：
```
请根据文档计算 Q1_PROFIT + Q2_PROFIT - RD_COST
```

当变量已存在时：
```
把刚才结果再乘以 0.1
```
会使用 `LAST_RESULT` 直接计算。

实现方式（怎么实现的）：
- 系统会先拿到你写的表达式（比如 `A + B - C`），然后把它“清洗”成标准格式。
- 变量从两个地方来：  
  1) **检索到的文本**（比如文档里出现的 `A=100`）  
  2) **上一次计算结果**（自动放在 `LAST_RESULT`）
- 如果表达式里出现的变量都能找到，就进入安全计算器（只允许加减乘除和括号）。
- 计算得到的结果会存进 memory，下一轮可以直接继续“再乘以 0.1”这类追问。

代码位置（可选）：
`src/agent/tools/calculate_tool.py`、`src/agent/tools/calculator.py`

### 4.4 预算分析师工具（budget_analyst）
**功能**：根据年度预算与股价信息生成“预算增速 + 评分 + 评级”。

示例问题：
```
请根据年度预算分析股价，给出分析评级
```

实现方式（怎么实现的）：
- **先找数据**：  
  - 如果你传的是 JSON（结构化输入），就直接读取年度预算和股价。  
  - 如果是自然语言，系统会从“问题 + 检索文本”里用规则抓取“预算金额”和“年份”。
- **再算增速**：找到“最新年度预算”和“上一年预算”，计算同比增速（涨了多少百分比）。
- **再给评分**：  
  - 增速高 -> 加分；增速低或下滑 -> 扣分  
  - 文本里如果出现“上调预算/下调预算”的语气，也会加/扣分
- **最后映射成评级**：根据总分输出 **买入/增持/中性/减持/卖出**。
- 结果会写入 memory，后续可以继续问“用这个评分做计算”等。

代码位置（可选）：
`src/agent/tools/budget_analyst_tool.py`

### 4.5 多轮 Memory
**功能**：自动保存变量和上轮计算结果，用于追问。

示例：
1) `请计算 Q1_PROFIT + Q2_PROFIT`
2) `把上一步结果减去 1000000`

第二轮通常只触发 `calculate`，不会重复检索。

实现方式（怎么实现的）：
- memory 就像一个小记事本，专门记住：上轮计算结果、提取的变量、检索到的片段。
- 每个工具执行完都会返回一份“增量更新”，系统把这些更新合并进记事本。
- 当你说“把上一步结果再乘以 0.1”，规划器会识别这是追问，就直接调用计算工具。

代码位置（可选）：
`src/agent/memory.py`、`src/agent/planner.py`

### 4.6 CLI 命令
在对话中可使用：
- `/rebuild`：重建索引
- `/reset`：清空会话 + memory
- `/tools`：查看已注册工具
- `/memory`：查看当前 memory 摘要
- `/exit`：退出

实现方式（怎么实现的）：
- CLI 里以 `/` 开头的输入会被当作命令处理，比如 `/tools`、`/memory`。
- `/tools` 会列出当前注册的工具，让你确认系统能做哪些动作。
- `/memory` 会打印当前记事本摘要，方便你知道系统“记住了什么”。

代码位置（可选）：
`src/app/cli_chat.py`

---

## 5. 脚本入口说明

- `scripts/ingest_once.py`：构建索引
- `scripts/query_once.py`：单次 RAG 查询（含引用）
- `scripts/agentic_query_once.py`：单次 Agentic 演示（含轨迹）
- `scripts/rebuild_index.py`：仅重建索引
- `scripts/run_cli.sh`：推荐的 CLI 启动脚本

使用样例：
```bash
python3 scripts/query_once.py --question "这份文件主要讲什么？"
```

---

## 6. 关键模块说明

- Agent 执行器：`src/agent/graph.py`
- Planner + Router：`src/agent/planner.py`
- Memory：`src/agent/memory.py`
- Tool Registry：`src/agent/tools/registry.py`
- Retrieve Tool：`src/agent/tools/retrieve_tool.py`
- Calculate Tool：`src/agent/tools/calculate_tool.py`
- Budget Analyst Tool：`src/agent/tools/budget_analyst_tool.py`
- CLI：`src/app/cli_chat.py`
- Prompt 模板：`src/llm/prompts.py`

---

## 7. 规划文档

- `docs/multimodal_agentic_rag_plan.md`
