# 多模态 Agentic RAG 项目规划（V1）

## 1. 目标范围
- 输入模态：.txt文本、.md文档、PDF文档、图片、音频（后续扩展视频）。
- 输出能力：多轮对话问答、可追溯引用、工具调用、任务分解与执行。
- 工程目标：先把本地核心流程跑通（不考虑部署），再逐步增强能力与性能。

## 2. 建议项目结构
```text
agentic-rag/
├── .env.example
├── README.md
├── docs/
│   └── multimodal_agentic_rag_plan.md
├── src/
│   ├── app/
│   │   └── cli_chat.py              # 命令行多轮对话入口（本地优先）
│   ├── llm/
│   │   ├── client.py                # OpenAI 兼容客户端封装
│   │   ├── prompts.py               # 系统提示词、模板
│   │   └── schema.py                # 结构化输出 schema（Pydantic）
│   ├── agent/
│   │   ├── graph.py                 # Agent 流程图（plan -> act -> reflect）
│   │   ├── planner.py               # 任务规划
│   │   ├── memory.py                # 会话记忆
│   │   └── tools/
│   │       ├── rag_retrieve.py      # 检索工具
│   │       ├── web_search.py        # 外部检索（可选）
│   │       └── multimodal_parse.py  # 多模态解析入口
│   ├── ingest/
│   │   ├── pipeline.py              # 数据入库主流程
│   │   ├── parsers/
│   │   │   ├── text_pdf.py          # 文本/PDF解析
│   │   │   ├── image_ocr.py         # 图片 OCR
│   │   │   ├── audio_asr.py         # 音频转写
│   │   │   └── video_extract.py     # 视频抽帧+音频（后续）
│   │   ├── chunking.py              # 分块策略
│   │   └── metadata.py              # 元数据标准化
│   ├── retrieval/
│   │   ├── embeddings.py            # 向量化
│   │   ├── vector_store.py          # 向量库读写
│   │   ├── hybrid_search.py         # 混合检索（向量+关键词）
│   │   └── reranker.py              # 重排
│   ├── eval/
│   │   ├── smoke_test.py            # 冒烟测试
│   │   └── rag_eval.py              # 离线评测
│   └── utils/
│       ├── config.py                # 配置加载（.env）
│       └── logger.py                # 日志
├── data/
│   ├── raw/                         # 原始数据
│   ├── processed/                   # 清洗后的中间数据
│   └── index/                       # 本地索引/向量库持久化
├── scripts/
│   ├── ingest_once.py               # 单次入库
│   ├── rebuild_index.py             # 重建索引
│   └── run_eval.py                  # 评测脚本
├── tests/
│   ├── test_ingest.py
│   ├── test_retrieval.py
│   └── test_agent.py
├── environment.yml
└── test_llm.py
```

## 3. 工具栈（推荐默认）
| 模块 | 首选工具 | 说明 |
|---|---|---|
| LLM 接入 | `openai` SDK（兼容接口） | 已在项目中验证，继续沿用 |
| Agent 编排 | `langgraph` | 适合多步决策、工具调用、可回溯状态 |
| 配置管理 | `python-dotenv` + `pydantic-settings` | `.env` 管理与类型化配置 |
| 文档解析 | `pymupdf` | PDF 主解析器（当前固定方案） |
| 图片 OCR | 暂时不需要支持 |
| 音频转写 | `faster-whisper` | 成本低、可离线推理 |
| Embedding 模型 | 使用已有配置调用API |
| 向量库 | milvus |
| Reranker 模型 | 使用已有配调用API |

## 4. 环境变量规划（建议）
```env
LLM_API_URL=
LLM_API_KEY=
LLM_MODEL=
LLM_TIMEOUT=30
LLM_TEMPERATURE=0.2

EMBEDDING_API_URL=
EMBEDDING_API_KEY=
EMBEDDING_MODEL=
EMBEDDING_TIMEOUT=30

RERANKER_API_URL=
RERANKER_API_KEY=
RERANKER_MODEL=
RERANKER_TIMEOUT=30

MILVUS_URI=./data/index/milvus.db
MILVUS_COLLECTION=rag_chunks
RAW_DATA_DIR=./data/raw
PROCESSED_DATA_DIR=./data/processed
```

## 5. 实施顺序（建议）
1. 先补 `src/llm`、`src/app/cli_chat.py`，把现有 `test_llm.py` 能力模块化。
2. 再做 `ingest + retrieval` 的最小链路（文本/PDF -> 向量库 -> 检索问答）。
3. 接入 `agent` 层，加入工具调用与简单规划。
4. 最后加图片 OCR、音频 ASR，并补 `eval` 与 `tests`。
