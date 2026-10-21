# agentic-rag

## TODO（当前未完成）

- 语义去重：目前只做基于 chunk 键的去重，未做语义级去重
- 长期 Memory：目前只有进程内 / 会话内 memory，未做持久化长期记忆
- 多模态：目前以文本和 PDF 文本/表格抽取为主，未做图像/图表多模态理解
- 效果评估：已支持离线文档检索指标；仍需真实语料标注集、answer / tool-use 评估及检索消融实验
- 结构化 Chunk：已支持段落、标题、表格行和代码围栏；尚未实现跨块表头复制、完整 Markdown AST 或父子块检索
- 知识图谱：目前缺少跨文档实体关系建模，难以稳定检索文档之间的关系

本项目是一个可运行的 **本地 Agentic RAG**（当前仅保留 CLI）：
- 支持 `.txt/.md/.pdf` 入库与向量化检索/关键词索引
- PDF 在文本提取之外，额外抽取表格并入库（可检索利润表/现金流量表等）
- 查询改写（Query Rewrite）+ 多查询召回（Multi-Query Retrieval）
- 混合检索（向量 + 关键词，使用 weighted RRF 融合）+ 默认启用 reranker 重排
- 支持“遗漏补全”追问（如“你漏掉了一些公司”会触发重新检索）
- Agentic 工具链（`retrieve -> calculate / budget_analyst`）与轨迹输出
- 内置反思循环：`reflect -> replan -> retry`（默认最多重试 1 次）
- 执行阶段耗时反馈（route / plan / reflect / replan / tool / answer / total）
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
    -> Planner 生成工具计划（默认 0~8 步，可配置）
      -> Tool 执行（retrieve / calculate / budget_analyst）
        -> Reflect 质量检查（是否需要重试）
          -> Replan + Retry（最多 1 次，默认）
            -> 汇总引用 + 轨迹
              -> 最终回答
```

关键细节：
- **Router Chain**：优先判断问题类型，闲聊不进检索链路，避免性能浪费。
- **Planner**：根据问题/记忆决定是否需要检索或计算，默认最多 8 步（可配置）。
- **Reflect / Replan / Retry**：当检索无命中、计算变量缺失、工具运行异常等情况出现时，会自动反思并重规划。
- **Tool 执行**：
  - `retrieve`：混合检索（向量 + 关键词）+ 默认启用 rerank，写入 `memory`。
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

建议配置（rerank 默认启用）：
- `RERANKER_API_URL` `RERANKER_API_KEY` `RERANKER_MODEL`

常用可选：
- `MILVUS_URI`（默认 `./data/index/milvus.db`）
- `MILVUS_COLLECTION`（默认 `rag_chunks`）
- `RAW_DATA_DIR`（默认 `./knowledge`）
- `CHUNK_SIZE`（默认 `1200`）
- `CHUNK_OVERLAP`（默认 `180`）
- `RETRIEVAL_TOP_K`（默认 `8`）
- `RETRIEVAL_CANDIDATE_K`（默认 `64`）
- `HYBRID_VECTOR_WEIGHT`（默认 `0.6`）
- `HYBRID_KEYWORD_WEIGHT`（默认 `0.4`）
- `QUERY_REWRITE_ENABLED`（默认 `true`）
- `MULTI_QUERY_ENABLED`（默认 `true`）
- `MULTI_QUERY_COUNT`（默认 `3`）
- `EMBEDDING_BATCH_SIZE`（默认 `64`）
- `CHAT_HISTORY_MAX_MESSAGES`（默认 `80`）
- `PLANNER_MAX_STEPS`（默认 `8`）
- `PLANNER_RECENT_HISTORY_MESSAGES`（默认 `20`）
- `ANSWER_MAX_CONTEXTS`（默认 `16`）
- `ANSWER_MAX_TRACES`（默认 `24`）

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
这一步会同时生成向量索引和 `data/processed/chunks.jsonl`，供关键词检索使用。

如只新增/更新单个文件，可增量 upsert：
```bash
python3 scripts/ingest_once.py --file knowledge/xxx.pdf
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

### 4.2 查询改写 + 多查询混合检索与重排（retrieve）
**功能**：先进行 Query Rewrite，再做 Multi-Query 召回（每条 query 走向量检索 + 关键词检索），最后合并候选并启用 rerank 重排。

示例问题：
```
请从 688230_20260203_JJZK.pdf 找到“净利润”相关描述
```
输出包含：
- 向量召回片段
- 关键词召回片段
- 引用来源与页码
- rerank 信息（默认启用，未配置会自动降级）

实现方式（怎么实现的）：
- **Query Rewrite**：先将用户问题改写成更适合检索的一条查询，保留实体、时间、指标、变量名。
- **Multi-Query**：基于改写后的查询生成多个语义等价变体（默认 3 条），扩大召回覆盖面。
- **向量召回**：每条 query 都转向量，在向量库里找最相近片段，并跨 query 聚合。
- **关键词召回**：每条 query 都走 BM25，在本地 chunks 里找关键词匹配片段，并跨 query 聚合。
- **Metadata 增强**：chunk 持久化 `doc_id/file_name/company_code/company_name/report_year/is_table` 等元数据。
- **覆盖优先**：全局/补全类问题会按公司 metadata 做覆盖优先的结果选择，降低“只命中部分公司”概率。
- **关键词数据来源**：chunks 来自 `scripts/ingest_once.py` 生成的 `data/processed/chunks.jsonl`。
- **合并候选**：对每条查询分别排名，再按每种检索器的最佳名次进行 weighted RRF 融合（权重默认 0.6 向量 / 0.4 关键词，常数 60）；同一 chunk 在每种检索器中只贡献一次，避免 BM25 与余弦分数尺度影响排序。
- **默认启用 rerank**：候选池交给重排模型做最终排序（配置好 `RERANKER_*` 就会生效）。
- **写入 memory**：最终结果写入 memory，后续追问可以直接复用。

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
- 当你说“你漏掉了一些公司”这类补全追问，规划器会基于上一轮问题自动触发一次“全覆盖”检索。

代码位置（可选）：
`src/agent/memory.py`、`src/agent/planner.py`

### 4.6 Reflect / Replan / Retry（执行期纠错）
**功能**：执行工具链后自动做一轮反思；若判断当前结果质量不足，则重规划并重试一次（默认）。

常见触发场景：
- `retrieve` 返回 `no hits`
- `calculate` 出现 `unknown variable`
- 工具运行时抛异常（会被记录为 `tool_failed: ...`，不直接中断主流程）
- 路由为“需要查询知识库”但计划为空或无引用

实现方式（怎么实现的）：
- 每轮工具执行后进入反思判定，输出 `should_retry / reason / replan_feedback`。
- 需要重试时，规划器会接收“上一次计划 + 上一次观察 + replan_feedback”重新生成步骤。
- 默认最多重试 1 次（`AgentExecutor(max_replan_retries=1)`）。

代码位置（可选）：
`src/agent/graph.py`、`src/agent/planner.py`、`src/llm/prompts.py`

### 4.7 CLI 命令
在对话中可使用：
- `/rebuild`：重建索引
- `/upsert <file>`：增量更新单个文件到索引（不全量重建）
- `/reset`：清空会话 + memory
- `/tools`：查看已注册工具
- `/memory`：查看当前 memory 摘要
- `/exit`：退出

实现方式（怎么实现的）：
- CLI 里以 `/` 开头的输入会被当作命令处理，比如 `/tools`、`/memory`。
- `/tools` 会列出当前注册的工具，让你确认系统能做哪些动作。
- `/memory` 会打印当前记事本摘要，方便你知道系统“记住了什么”。
- 若触发反思重试，CLI 的 Timing 会出现 `reflect_*` / `replan_*` 阶段。

代码位置（可选）：
`src/app/cli_chat.py`

---

## 5. 脚本入口说明

- `scripts/ingest_once.py`：构建索引（支持 `--file` 增量 upsert）
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

---

## 8. 近期更新（同步代码实现）

- 按 WeKnora 的文档身份、结构切块和排名融合思路完成首轮改造，参考版本见下文。
- 文档 ID 按源根目录下的规范化相对路径生成；内容 hash 与 chunk ID 单独存储，修复同名文件互相覆盖及 PDF 多个解析单元重复 chunk 序号的问题。
- 检索失败或无命中后不再回退上一轮证据；新检索变量优先，保留显式 `LAST_RESULT` 计算追问。
- 修复千分位数字解析、连续 `LAST_RESULT` 追问，以及 `finish` 跳过失败检查的问题。
- 新增离线检索评估 CLI、回归测试和 GitHub Actions；合成样例只验证算分流程。
- 新增执行期反思循环：`reflect -> replan -> retry`。
- 修复高风险问题：工具异常不再直接中断整次运行，改为观测并交给反思重试。
- 修复中风险问题：`finish` 作为控制信号处理，不再误报 `tool_not_registered`。
- 修复中风险问题：Router 分类解析增加规范化和否定词规避（如“这不是闲聊”“不需要查询知识库”）。
- 修复中风险问题：知识库路由下空计划会触发补救重规划，避免直接无依据回答。

## 9. 本轮改造与验证

### 索引升级

旧索引缺少新版文档与分块身份，不能和新数据混写。升级后先全量重建：

```bash
python3 scripts/ingest_once.py
```

该命令会替换配置中的 Milvus collection 和 `chunks.jsonl`，需要原始文档及 embedding 服务配置。增量更新继续使用：

```bash
python3 scripts/ingest_once.py --file knowledge/a.pdf
```

直接调用 `RAGIndexer.upsert_files` 时必须传 `raw_data_dir`，并与全量重建使用同一逻辑源根目录。ID 不包含文件内容：修改同一路径的文件会更新同一个文档；移动整个源目录但保持相对路径不会改变 ID；重命名文件视为新文档，旧条目需要全量重建清理。文件内容变化或切块文本/位置变化会改变 chunk ID。

每个 chunk 保存 `doc_id`、`chunk_id`、`content_hash`、页码和 `start_offset/end_offset`。offset 相对当前解析单元的文本（PDF 页面文本或抽取表格），不是 PDF 文件字节位置。CLI 引用显示文档及 chunk ID。

切块优先保持段落、标题、表格行和代码围栏完整；超长单元回退字符窗口。overlap 是上限，结构边界只复用能完整放下的尾部单元，实际重叠可能更少。标题处不带上一节内容。

embedding 批次数量、维度及有限值会在替换数据前检查；向量维度变化要求重建；删除失败直接报错。关键词 JSONL 使用临时文件加原子替换。Milvus 与 JSONL 没有跨存储事务，本地索引按单写入者使用，外部写入故障后应全量重建，不能把这些检查视为双索引原子更新。

### 检索与多轮上下文

每份查询结果列表单独计算名次，每种检索器对同一 chunk 使用最佳名次。融合分数为 `sum(weight / (60 + rank))`，权重先归一化；`rrf_score` 不是相似度或相关性概率。多查询的完成顺序和 BM25 分数单位不会改变融合排名，reranker 仍在融合后执行。

每次尝试检索先清空旧检索文本、旧引用和普通变量；失败或空结果不会重新启用它们。同一轮内多次检索仍会合并本轮成功获取的证据。无检索的计算追问可继续使用历史上下文，显式 `LAST_RESULT` 保存前次计算值。新文本中的同名普通变量优先于旧变量。

### 自动化测试

推荐 Python 3.11。在已安装项目运行依赖的环境中：

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest -q
```

测试用合成文档、fake embedding、fake 向量库和 fake planner 验证真实切块、入库、检索融合及执行器，不调用模型 API 或数据库。GitHub Actions 运行相同测试。真实 embedding、Milvus 和 reranker 的集成效果需另外验证。

### 离线检索评估

```bash
python3 scripts/evaluate_retrieval.py \
  --cases eval/examples/cases.jsonl \
  --results eval/examples/results.jsonl \
  --k 2
```

`cases` 每行含 `id` 和 `relevant_doc_ids`，`results` 每行含同一 `id` 和按排名排列的 `retrieved_doc_ids`。用 CLI 引用中的实际 `doc_id` 标注证据，并从真实检索输出记录排名；不要直接把标准答案 ID 填成检索结果。失败检索显式记录空列表，缺失或多余 case、重复 ID 会报错。

脚本按文档去重后取前 K 个，计算有正例问题的宏平均 Recall@K、MRR@K、nDCG@K。无答案问题单独统计，不混入正例均值。仓库中的三条合成案例仅演示输入格式和算分；其分数不代表当前模型或检索配置的效果。脚本不运行检索，也不评估最终答案事实性。

### WeKnora 参考

参考 [Tencent/WeKnora](https://github.com/Tencent/WeKnora/tree/e0c28bef3562de7d2d78d6c1fa480a045468a40f)，固定 commit 为 `e0c28bef3562de7d2d78d6c1fa480a045468a40f`。本轮保留 Python CLI，按以下设计重新实现，没有移植 Go 平台代码：

- [文档创建](https://github.com/Tencent/WeKnora/blob/e0c28bef3562de7d2d78d6c1fa480a045468a40f/internal/application/service/knowledge_create.go)与[替换](https://github.com/Tencent/WeKnora/blob/e0c28bef3562de7d2d78d6c1fa480a045468a40f/internal/application/service/knowledge_replace.go)：源身份、内容版本及展示文件名分开。
- [检索融合](https://github.com/Tencent/WeKnora/blob/e0c28bef3562de7d2d78d6c1fa480a045468a40f/internal/application/service/knowledgebase_search_fusion.go)：按独立结果列表的名次融合。
- [标题切块](https://github.com/Tencent/WeKnora/blob/e0c28bef3562de7d2d78d6c1fa480a045468a40f/internal/infrastructure/chunker/heading_splitter.go)：结构分块并保留原文坐标。
- [引用约束](https://github.com/Tencent/WeKnora/blob/e0c28bef3562de7d2d78d6c1fa480a045468a40f/internal/modelcontext/citations.go)与[评估类型](https://github.com/Tencent/WeKnora/blob/e0c28bef3562de7d2d78d6c1fa480a045468a40f/internal/types/evaluation.go)：区分当前证据与历史上下文，将检索指标和生成指标分开。

本轮未加入 WeKnora 的 Web 管理端、Wiki、知识图谱、多租户、父子块/邻块扩展阅读或异步文档任务。
