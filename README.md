# 法律咨询聊天机器人

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> 基于大模型的智能法律咨询服务系统，提供劳动纠纷、婚姻家庭、借贷、交通事故、侵权责任等高频法律场景的咨询服务。

---

## 📋 目录

- [项目简介](#-项目简介)
- [功能特性](#-功能特性)
- [系统架构](#-系统架构)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [核心功能实现](#-核心功能实现)
- [性能测试](#-性能测试)
- [未来改进](#-未来改进)
- [致谢](#-致谢)

---

## 🎯 项目简介

随着人工智能技术快速发展，大模型在垂直领域的应用越来越广泛。在法律服务场景中，普通人面临律师咨询成本高、法律知识不足、维权流程复杂等问题。

本项目依托大模型接口、前后端分离架构、数据库技术与自然语言处理技术，实现可对话、可分类、可记忆、可检索的法律咨询机器人，具有较强的现实意义与实用价值。

### 技术栈

| 类别 | 技术 |
|------|------|
| 后端框架 | FastAPI |
| 数据库 | MySQL |
| 向量库 | Chroma |
| 大模型 | 智谱 GLM-4-Flash / 通义千问 / 本地 Ollama |
| 前端 | HTML + CSS + JavaScript |
| 部署 | Uvicorn |

---

## ✨ 功能特性

### 核心功能

- 🤖 **智能法律咨询** - 对接在线大模型 API，提供合规法律问答
- 📝 **法律文本处理** - 数据清洗、去重、格式标准化
- 🏷️ **意图分类** - 自动识别劳动争议、婚姻家庭、借贷纠纷等场景
- 🔍 **实体识别** - 提取法律文件、当事主体、行为动作等关键信息
- 💾 **多轮对话** - 支持上下文理解与连续交流
- 🔎 **RAG 检索** - 本地向量库增强，减少模型幻觉
- 📊 **性能监控** - 压力测试与响应时间统计

### 支持的法律场景

- 劳动争议（工资、劳动合同、试用期、辞退、加班、社保、工伤）
- 婚姻家庭（离婚、财产分割、抚养权、赡养、继承、遗嘱）
- 借贷纠纷（借条、欠款、利息、还款、债务、担保）
- 合同纠纷（合同、协议、违约、条款、解除、履行）
- 侵权责任（赔偿、侵权、人身损害、名誉权、肖像权、交通事故）
- 知识产权（专利、商标、著作权、抄袭、盗版）

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        应用层 (Web UI)                       │
│              HTML + CSS + JavaScript 交互界面                │
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP/REST API
┌─────────────────────────▼───────────────────────────────────┐
│                        服务层 (FastAPI)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  意图识别    │  │  RAG 检索   │  │    大模型调用        │  │
│  │  实体提取    │  │  向量匹配   │  │  GLM-4-Flash/Ollama │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   MySQL       │ │    Chroma     │ │   外部 API    │
│  对话历史      │ │   向量库      │ │  智谱/通义    │
└───────────────┘ └───────────────┘ └───────────────┘

---

## 💻 核心功能实现

### 1. 法律数据预处理

```python
import re

# 读取原始法律文本
with open("laws.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 清洗每一行
cleaned = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    # 去掉乱码、控制字符
    line = re.sub(r'[\x00-\x1F\x7F]', '', line)
    # 只保留中文、数字、常用法律标点
    line = re.sub(r'[^\u4e00-\u9fa50-9，。！？；：、（）【】《》------]', '', line)
    if len(line) < 5:
        continue
    cleaned.append(line)

# 去重并保存
unique_lines = list(dict.fromkeys(cleaned))
```

### 2. 意图分类与实体识别

```python
intent_rules = {
    "合同纠纷": ["合同", "协议", "违约", "条款", "解除", "履行", "无效"],
    "劳动争议": ["工资", "劳动合同", "试用期", "辞退", "加班", "社保", "工伤"],
    "婚姻家庭": ["离婚", "财产分割", "抚养权", "赡养", "继承", "遗嘱"],
    "借贷纠纷": ["借条", "欠款", "利息", "还款", "债务", "担保"],
    "侵权责任": ["赔偿", "侵权", "人身损害", "名誉权", "肖像权", "交通事故"],
    "知识产权": ["专利", "商标", "著作权", "抄袭", "侵权", "盗版"],
}

def classify_intent(text):
    for intent, keywords in intent_rules.items():
        for kw in keywords:
            if kw in text:
                return intent
    return "其他"
```

### 3. 大模型对话

```python
system_prompt = """
你是专业法律AI咨询助手，基于中国现行法律法规作答。

约束：
1. 仅依据中国大陆现行有效法律；
2. 不预测判决、不代写诉状、不提供诉讼代理；
3. 拒绝违法/灰色/侵权咨询；
4. 回答简洁通俗、条理清晰、分点说明；
5. 结合上下文连贯回答。
"""

def chat_with_glm(user_question, history=None):
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_question})
    
    response = client.chat.completions.create(
        model="GLM-4-Flash",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content
```

### 4. RAG 检索增强

```python
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# 初始化向量库
embeddings = OllamaEmbeddings(model="qwen2.5:1.5b")
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./vector_db"
)

# 检索相关法条
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
docs = retriever.invoke(question)
context = "\n".join([d.page_content for d in docs])
```

---

## 📊 性能测试

### 测试结果

| 测试项目 | 单并发 | 10并发 |
|---------|--------|--------|
| 平均响应时间 | 1.5s | 87.21s |
| 接口成功率 | 100% | 100% |

### 运行压力测试

```bash
python stress_test.py
```

---

## 🔮 未来改进

- [ ] 引入 Redis 缓存提升响应速度
- [ ] 采用机器学习提升意图分类精度
- [ ] 扩展案例检索、法律文书生成功能
- [ ] 优化移动端适配
- [ ] 支持语音输入与输出
- [ ] 接入更多大模型 API

---

## 🙏 致谢

- 智谱 AI - 提供 GLM-4-Flash 大模型接口
- 阿里巴巴 - 通义千问大模型
- Ollama - 本地大模型部署方案
- FastAPI - 高性能 Web 框架

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

<p align="center">
  Made with ❤️ for Legal AI
</p>
