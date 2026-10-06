# GraphNexus：融合知识抽取、图谱问答与联网检索的增强系统

## 📖 项目简介

GraphNexus 基于开源项目 OneKE 二次开发，构建**离线知识加工 + 在线知识问答**双链路闭环。

- **离线链路**：解析PDF/DOCX/HTML/JSON/TXT多格式文档，通过多智能体完成实体、关系、事件、三元组的结构化抽取；将抽取得到的事实以 `Document‑Fact‑Entity` 来源化模型存入Neo4j图数据库，每条事实可溯源原始文档。
- **在线链路**：基于 LangGraph 实现 **local‑first 结构化RAG**。用户提问优先检索Neo4j本地图谱事实；当本地证据得分低于阈值且允许联网时，通过 Tavily Remote MCP 调用网页搜索补充外部信息；最终输出带证据编号引用的回答，同时具备缓存、重试、熔断降级、可观测工具执行轨迹。

## ✨ 核心特性

### 📄 离线知识抽取

-  多格式文档解析：TXT / DOCX / HTML / JSON / PDF；PDF支持普通pdfplumber与复杂Docling双路由
-  支持NER实体识别、RE关系抽取、EE事件抽取、Triple三元组抽取、Base开放域抽取
-  Dynamic Schema：开放任务可由LLM动态推导Pydantic输出结构；标准任务使用预定义Schema约束输出格式
-  案例库增强抽取、三次采样自一致性、反思修正，降低大模型幻觉与非法JSON输出
-  来源化入库：`Document‑Fact‑Entity` 图模型，参数化Cypher写入，每条事实可追溯来源文档，兼容旧版三元组数据

### 🤖 在线知识问答

-  LangGraph 有状态工作流编排结构化RAG完整链路
-  Neo4j结构化事实检索，生成本地证据标记 `[L1][L2]`
-  证据充分度启发式评分，**仅本地证据不足且开启开关才触发联网搜索**
-  Tavily Remote MCP 标准协议调用外部搜索，网页证据标记 `[W1][W2]`
-  引用式生成 + 引用编号校验，校验失败执行单次修订
-  工程稳定性：SQLite TTL缓存、超时、指数退避重试、故障熔断、本地降级，外部服务异常不中断本地问答
-  多接入方式：Streamlit问答前端、FastAPI HTTP接口、命令行CLI
-  可观测：完整Tool Trace工具调用轨迹、端到端评测脚本、自动化单元测试

## 🏗️ 系统架构

整体分为**离线知识加工链路**与**在线知识问答链路**。

```
┌─────────────────────────────────────────────────────────────────┐
│  接入层：Streamlit前端 / FastAPI RESTful接口 / CLI命令行        │
│  文档上传、任务配置、用户问答；返回抽取结果、带引用答案、工具轨迹 │
├─────────────────────────────────────────────────────────────────┤
│  编排调度层：流水线Pipeline / LangGraph状态机调度器             │
│  离线：文档处理→多智能体流水线调度；在线：节点分发、条件分支、回退 │
├─────────────────────────────────────────────────────────────────┤
│  智能体业务层：离线抽取智能体组 + 在线问答工作流节点            │
│  Schema智能体 / 抽取智能体 / 反思智能体                         │
│  本地检索 / 证据评分 / MCP搜索 / 生成回答 / 引用校验与修订     │
├─────────────────────────────────────────────────────────────────┤
│  工具能力层：公共能力组件                                       │
│  多格式文档解析工具 / Pydantic‑Schema工具 / Case案例库工具     │
│  LLM调用工具 / Neo4j结构化检索 / Tavily‑MCP工具网关            │
│  SQLite‑TTL缓存 / JSON解析清洗 / Token统计 / 评测测试工具       │
├─────────────────────────────────────────────────────────────────┤
│  数据存储层：多介质存储体系                                      │
│  Neo4j（Document‑Fact‑Entity来源化知识图谱）                     │
│  JSON案例库 / SQLite搜索缓存 / 本地文件（配置、样例、评测集）    │
└─────────────────────────────────────────────────────────────────┘
```

> 说明：离线侧三个智能体为**固定流水线职责模块，非自主规划Agent**；在线侧使用LangGraph状态+条件边实现动态路由。

## 🧩 智能体角色分工

### 离线链路智能体

1. **Schema智能体（结构定义）** 负责确定抽取输出结构：标准任务加载预定义Pydantic Schema；Base开放任务支持LLM动态推导Schema，生成格式约束交给后续抽取。
2. **抽取智能体** 组装任务指令、约束条件、Few‑shot案例、Schema格式说明，调用大模型执行分块抽取，做JSON清洗解析，输出结构化结果。支持普通抽取 / 案例库增强抽取两种模式。
3. **反思智能体** 对同一份文本做多温度采样，做结果规范化后执行多数投票；全部结果不一致时召回历史错误案例，对异常结果进行反思修正，降低输出不稳定。

### 在线链路工作流节点（LangGraph）

1. `retrieve_local`：从Neo4j检索本地结构化事实，输出带L编号本地证据
2. `assess_evidence`：计算本地证据充分分数，判断是否需要调用外部搜索
3. `external_search`：MCP调用Tavily搜索网页，输出带W编号网页证据，内置缓存/重试/熔断
4. `synthesize`：融合本地+网页证据，LLM生成带引用编号的回答
5. `reflect`：正则校验回答内引用编号是否存在于证据集合
6. `revise`：引用无效时执行一次回答修订，修订后直接输出，不循环校验

## 🛠️ 技术栈详解

| 分类        | 技术组件                                           | 用途                                                         |
| ----------- | -------------------------------------------------- | ------------------------------------------------------------ |
| 文档解析    | pdfplumber、Docling、docx2txt、BeautifulSoup4      | 多格式文档加载；PDF双路由，兼顾速度与复杂版面解析            |
| 抽取&Schema | Pydantic、JsonOutputParser                         | 定义、序列化结构化输出约束，约束LLM输出JSON                  |
| 案例库      | Sentence‑Transformers、RapidFuzz、本地JSON         | 语义+字符串混合相似度召回好坏案例，案例持久化JSON，向量内存缓存 |
| Agent编排   | LangGraph、LangChain                               | 在线问答有状态工作流、条件分支、状态管理                     |
| 大模型      | 智谱/OpenAI兼容API、vLLM、DeepSeek/Qwen系列        | 支持远程API与本地开源模型推理；问答模块独立轻量LLM客户端     |
| 图数据库    | Neo4j、参数化Cypher                                | 存储Document‑Fact‑Entity来源化图谱，结构化事实检索，避免Cypher注入 |
| 外部工具    | MCP协议、langchain‑mcp‑adapters、Tavily Remote MCP | 标准协议调用网页搜索，不直接封装Tavily SDK                   |
| 稳定性      | SQLite(TTL缓存)、asyncio超时、指数退避重试、熔断器 | 搜索结果缓存、外部服务故障隔离降级                           |
| 服务与交互  | Streamlit、FastAPI、Uvicorn                        | Web前端可视化、HTTP接口服务、命令行交互                      |
| 测试评测    | Pytest、JSONL评测数据集                            | 单元测试、端到端评测，统计引用率、路由准确率、延迟等指标     |
| 配置管理    | python‑dotenv                                      | 密钥、环境变量管理，密钥不提交代码仓库                       |

## 📂 项目目录结构

```
OneKE‑Nexus
├── figs/                     # 图片资源logo、截图
├── frontend/                 # Streamlit前端
│   ├── app.py                # 前端主入口
│   └── components/knowledge_agent.py  # 问答页面组件
├── src/
│   ├── run.py                # 离线知识抽取CLI入口
│   ├── pipeline.py           # 离线完整流水线
│   ├── construct/convert.py  # 来源化Neo4j入库逻辑
│   ├── modules/              # 离线三大智能体、案例库、schema库
│   │   ├── schema_agent.py
│   │   ├── extraction_agent.py
│   │   ├── reflection_agent.py
│   │   └── knowledge_base/
│   ├── utils/                # 文件解析、分块、JSON清洗工具
│   ├── knowledge_agent/      # ✨新增在线问答智能体核心
│   │   ├── state.py          # LangGraph状态定义
│   │   ├── workflow.py       # 工作流编排
│   │   ├── retriever.py      # Neo4j检索器
│   │   ├── mcp_gateway.py    # Tavily MCP网关
│   │   ├── cache.py          # SQLite缓存
│   │   └── factory.py        # 轻量LLM客户端
│   ├── knowledge_agent_cli.py   # 问答命令行入口
│   └── knowledge_agent_api.py   # FastAPI服务入口
├── examples/config/          # 抽取任务yaml配置样例
├── scripts/                  # 脚本：MCP检查、评测脚本
├── tests/                    # 单元测试
├── data/eval/                # 评测数据集
├── .env.example              # 环境变量模板
├── requirements.txt
└── README.md
```

## 🚀 快速开始

### 1. 环境准备

```
# 创建conda虚拟环境
conda create -n GraphNexus python=3.9
conda activate GraphNexus

# 安装依赖
pip install -r requirements.txt
```

复制环境变量模板，填写模型、Neo4j、Tavily密钥：

```
cp .env.example .env
# 修改 .env 文件填入 LLM、Neo4j、TAVILY_API_KEY 配置
```

> 前置依赖：Neo4j服务本地/远程部署；GPU可选，本地模型需要CUDA环境。

### 2. 离线：文档抽取并构建知识图谱

修改 `examples/config/Triple2KG.yaml`，配置Neo4j连接信息。

```
python src/run.py --config examples/config/Triple2KG.yaml
```

执行后完成文档解析、抽取，以来源化模型写入Neo4j数据库。

### 3. 在线知识问答（三种方式）

#### 方式① Streamlit Web前端（推荐演示）

```
cd frontend
python -m streamlit run app.py
```

浏览器访问 `[http://127.0.0.1:8501](http://127.0.0.1:8501)`，切换页面到 `Knowledge Agent 问答`，可输入问题，控制是否允许联网搜索。

#### 方式② 命令行CLI

```
# 仅查询本地图谱，不联网
python src/knowledge_agent_cli.py "你的问题" --no-web
# 允许本地不足时调用MCP联网搜索
python src/knowledge_agent_cli.py "你的问题"
```

#### 方式③ FastAPI HTTP接口

```
uvicorn knowledge_agent_api:app --app-dir src --host 0.0.0.0 --port 8080
```

接口示例请求

```
POST [http://127.0.0.1:8080/v1/query](http://127.0.0.1:8080/v1/query)
Content‑Type: application/json
{
  "question": "你的问题",
  "allow_web": true
}
```

访问健康检查：`GET [http://127.0.0.1:8080/health](http://127.0.0.1:8080/health)`

### 4. 评测与MCP连通检查

```
# 检查Tavily MCP连通性
python scripts/check_tavily_mcp.py

# 执行端到端评测
python scripts/evaluate_knowledge_agent.py data/eval/xxx.jsonl --output result.json

# 运行单元测试
python -m pytest tests/ -q
```

------

## 📌 致谢 & 引用

本项目基于 **ZJUNLP OneKE** 进行二次开发。

```
@inproceedings{luo2025oneke,
  title={OneKE: A Dockerized Schema‑Guided LLM Agent‑based Knowledge Extraction System},
  author={Luo, Yujie and Ru, Xiangyuan and Liu, Kangwei and others},
  booktitle={Companion Proceedings of the ACM on Web Conference 2025},
  pages={2871--2874},
  year={2025}
}
```

