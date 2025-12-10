# Agent Workflow Memory 论文阅读报告

> **论文信息**  
> - 标题：Agent Workflow Memory  
> - 作者：Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, Graham Neubig  
> - 机构：Carnegie Mellon University, Massachusetts Institute of Technology  
> - 链接：https://github.com/zorazrw/agent-workflow-memory

---

## 目录

1. [Introduction](#1-introduction)
2. [Agent Workflow Memory](#2-agent-workflow-memory)
3. [Experiments](#3-experiments)
4. [Exploring Optimal Workflow Representations](#4-exploring-optimal-workflow-representations)
5. [Exploring Workflow Utilization in Context and in Action](#5-exploring-workflow-utilization-in-context-and-in-action)
6. [Related Work](#6-related-work)
7. [Conclusion](#7-conclusion)
8. [Appendix分析](#appendix分析)
9. [总体评价](#总体评价)

---

## 1. Introduction

### 1.1 研究背景与问题

论文开篇指出当前LLM-based agent在数字任务（如网页导航、移动应用操作）上快速发展，但存在一个核心缺陷：现有agent主要通过training或in-context learning整合固定的示例集合，导致它们在面对与训练示例不同的任务上下文或环境时缺乏鲁棒性。

论文提出的核心洞察是：**人类能够从过去经验中抽象出可复用的任务流程（task workflows），并用这些知识指导未来活动**。现有agent缺乏这种能力，它们单独处理每个任务，无法从成功和失败中学习，因而无法随时间适应。

### 1.2 方法概述

论文提出Agent Workflow Memory (AWM)，核心机制包括两步：

1. **Workflow Induction**：从agent轨迹中归纳workflow，提取可复用的子程序
2. **Workflow Integration**：将这些workflow整合进agent的memory中，用于指导后续任务的求解

每个workflow代表一个目标及其对应的通用例程，能有效捕捉agent需要习得的最核心、最可复用的技能。论文强调AWM能够构建越来越复杂的workflow：例如先学会"按名称查找地点"，之后基于这个workflow构建更复杂的"获取某地点的邮编"。

### 1.3 应用场景

AWM支持两种场景：

| 场景 | 描述 | 适用情况 |
|------|------|----------|
| **Offline** | 预先从标注样本中归纳workflow，测试时使用 | 有高质量训练数据 |
| **Online** | 在测试过程中动态地从成功预测中归纳workflow | 无额外标注数据 |

### 1.4 主要贡献

论文在WebArena和Mind2Web两个benchmark上进行实验，覆盖1000+任务、200+域。主要结果包括：

- WebArena上相对成功率提升**51.1%**，同时减少了完成任务所需的步数
- Mind2Web上相对成功率提升**24.6%**
- 在cross-task、cross-website、cross-domain评估中，online AWM展现出强大的泛化能力，随着train-test分布差距增大，相对baseline的优势从8.9增长到14.0个绝对百分点

### 1.5 批判性分析

Introduction部分的motivation非常清晰，但有几点值得注意：

> **问题1**：Figure 1展示的累积成功率曲线看起来很impressive，但它是基于特定的样本顺序得到的。论文没有讨论样本顺序对结果的影响，这在online setting下是一个重要的confounding factor。

> **问题2**：论文声称AWM不需要human supervision，但实际上online setting依赖一个evaluator来判断trajectory是否成功，这个evaluator本身的准确性会直接影响workflow的质量。

---

## 2. Agent Workflow Memory

### 2.1 Problem Statement

论文定义了web navigation任务的形式化框架：

**Agent组成**：
- 语言模型backbone $L$
- 文本形式的memory $M$（基础memory包含内置action如CLICK、TYPE的文档）

**任务求解过程**：
1. 给定自然语言指令 $q$，agent在环境中行动
2. 每个时间步 $t_i$，环境状态 $s_i$ 产生观察 $o_i$
3. 模型据此生成action：$L(q, M, o_i) \rightarrow a_i$
4. Action在环境中执行并改变状态：$T(s_i, a_i) \rightarrow s_{i+1}$
5. 循环持续直到模型预测STOP或达到终止条件

**Experience定义**：
- 每个完成的任务形成一个experience $e$
- 包含指令 $q$ 和尝试解决任务的轨迹（每步包含observation $o$ 和action $a$）

**目标**：从experience集合 $E$ 中归纳有用的workflow $W$，并加入agent memory $M$ 作为后续任务求解的指导。

**批判性分析**：这个形式化比较标准，但有一个隐含假设：observation是以文本形式呈现的（如accessibility tree）。这限制了方法对需要视觉理解的任务的适用性。

### 2.2 Workflow Representation

每个workflow包含两个组成部分：

#### Workflow Description ($d$)

自然语言描述workflow的高层目标，本质上是该workflow功能的总结。描述的获取方式：
- 启发式地从experience指令中提取
- 用LM进行总结

#### Workflow Trajectory

一系列步骤 $(p_1, p_2, \ldots)$ 来完成 $d$ 描述的过程。每个步骤 $p$ 包含三部分：

| 组成部分 | 描述 | 示例 |
|----------|------|------|
| 环境状态描述 | 当前状态的NL描述 | "Order {id} is shown" |
| 推理过程 | Agent决定下一步的reasoning | "Order {id} is found, I will now terminate" |
| 可执行action | 程序形式的action | `stop()` |

**批判性分析**：

这种表示设计有几个亮点：
1. 使用NL描述环境状态而非raw HTML，保持了抽象性
2. 包含reasoning过程是chain-of-thought的思路

潜在问题：workflow中的element id（如'145'、'147'）可能是example-specific的，在新环境中这些id会变化。论文在LM-based induction中用placeholder替换这些值，但具体如何替换、替换的粒度如何控制，没有详细说明。

### 2.3 Inducing and Using Workflows

#### LM-based Workflow Induction

核心思想是prompting LM从一个或多个input experience中提取common sub-routines。

**两个设计原则**：

1. **归纳更细粒度的sub-task而非整个task**
   - 例如：从"在Amazon买猫粮并送到我家地址"中提取"在Amazon搜索商品"

2. **用变量名替换example-specific值**
   - 例如：将"dry cat food"替换为`{product-name}`

归纳出的workflow基于双空行分割并分别存储。

#### Workflow的使用

归纳完成后，workflow被整合进agent memory：

$$M + W \rightarrow M_w$$

求解新任务时：

$$L(q, M_w, o) = L(q, M + W, o) \rightarrow a$$

#### Offline Scenario

当有额外的canonical experience时采用，分两阶段：

```
阶段1 (Training): I(E_train) → W_offline
阶段2 (Inference): L(q, M + W_offline, o_test) → a_test
```

特点：每个测试样本使用相同的workflow memory。

#### Online Scenario

当没有额外标注数据时采用，只需要测试queries。

**具体流程**：

```
1. Agent从默认memory M开始
2. 给定第t个测试指令q_t，生成action trajectory
3. 使用LM-based evaluator判断是否成功
4. 如果成功，将trajectory转换为workflow: I(e_t) → {w_t}
5. 将新workflow加入memory: M_t + {w_t} → M_{t+1}
6. 用更新后的memory处理下一个指令
```

**批判性分析**：

这一节有几个关键问题论文没有充分讨论：

| 问题 | 描述 |
|------|------|
| Scalability | 将"所有训练样本拼接成单个prompt"可能遇到context length限制 |
| 顺序依赖 | Online setting下样本顺序会显著影响结果，论文没有报告variance |
| Evaluator准确性 | False positive会导致错误workflow污染memory |
| Workflow selection | 没有relevance-based retrieval，所有workflow直接塞进prompt |

---

## 3. Experiments

论文在WebArena和Mind2Web两个benchmark上进行实验。对于两个benchmark，AWM都按website分组进行。

### 3.1 WebArena

**Benchmark介绍**：
- 812个web navigation任务
- 5个website，4个应用域：电商、社交论坛、协作软件开发、内容管理
- 支持execution-based evaluation

**Baseline方法**：
- BrowserGym：当前SOTA的autonomous方法
- BrowserGym_{ax-tree}：只使用accessibility tree表示的版本
- SteP：使用14个人工编写的expert workflow

**实验设置**：GPT-4 (gpt-4-0613)，temperature=0，只进行online setting。

#### 3.1.1 Main Results

| Method | Total SR | Shopping | CMS | Reddit | GitLab | Maps | # Steps |
|--------|----------|----------|-----|--------|--------|------|---------|
| SteP* | 33.0 | 37.0 | 24.0 | 59.0 | 32.0 | 30.0 | - |
| WebArena | 14.9 | 14.0 | 11.0 | 6.0 | 15.0 | 16.0 | - |
| AutoEval | 20.2 | 25.5 | 18.1 | 25.4 | 28.6 | 31.9 | 46.7 |
| BrowserGym | 23.5 | - | - | - | - | - | - |
| BrowserGym_{ax-tree} | 15.0 | 17.2 | 14.8 | 20.2 | 19.0 | 25.5 | 7.9 |
| **AWM** | **35.5** | 30.8 | 29.1 | 50.9 | 31.8 | 43.3 | 5.9 |

**关键发现**：
- AWM比BrowserGym baseline提高**12.0个绝对百分点**（51.1%相对提升）
- 超过使用人工编写workflow的SteP方法，高出7.6%相对提升
- 在5个website上都有11.8-30.7个绝对百分点的提升
- 平均步数比BrowserGym少约2步

**批判性分析**：
- 与BrowserGym的比较需注意实验设置对齐
- SteP在Reddit上达到59.0，而AWM只有50.9，说明某些特定domain人工设计的workflow仍有优势

#### 3.1.2 Efficient Learning from Small Amounts of Data

Figure 5展示了online AWM的学习曲线特征：

- **Rapid learning phase** (0-40 examples)：快速获取essential workflow
- **Stable inference phase**：学习更高级workflow，成功率趋于稳定

**批判性分析**：
- 曲线基于特定样本顺序，应报告多次随机shuffle的mean和std
- 图中成功率在40样本后略有下降，可能暗示后期归纳的workflow有噪声

#### 3.1.3 Cross-Template Workflow Generalization

**动机**：验证AWM不只是学会了template内的pattern，而是真正的跨任务泛化。

**实验设计**：对每个template只随机选择一个样本，确保所有样本来自不同template。

| Method | Total SR | Shopping | CMS | Reddit | GitLab | Maps |
|--------|----------|----------|-----|--------|--------|------|
| SteP* | 32.1 | 26.5 | 29.3 | 52.2 | 27.3 | 36.4 |
| AutoEval | 23.2 | 12.2 | 17.1 | 21.7 | 31.8 | 36.4 |
| BrowserGym_{ax-tree} | 20.5 | 10.4 | 17.8 | 23.1 | 27.3 | 28.6 |
| **AWM** | **33.2** | 24.5 | 29.3 | 52.2 | 31.8 | 39.4 |

**Case Study (Figure 6)**：

```
[Workflow 1: Find a place by its name]
    ↓ 复用前几步
[Workflow 2: Get the zip code of a place]
    = Workflow 1的步骤 + 获取邮编的新步骤
```

这展示了hierarchical skill building的思想。

**批判性分析**：实验仍在同一streaming过程中进行，更严格的做法应该只用cross-template样本运行online AWM。

### 3.2 Mind2Web

**Benchmark介绍**：强调跨task、website、domain的泛化能力。

**评估指标**：
1. Element Accuracy：是否选择了正确的页面元素
2. Action F1：对元素采取的action是否正确
3. Step Success Rate：结合(1)和(2)
4. Task-level Success Rate：所有步骤是否都成功

**Baseline方法**：
- MindAct：webpage element filtering + multi-choice格式
- Synapse：trajectory风格 + retrieved relevant examples

#### 3.2.1 Main Results

| Method | Elem Acc | Action F1 | Step SR | SR |
|--------|----------|-----------|---------|-----|
| MindAct (3.5) | 20.3 | 56.6 | 17.4 | 0.8 |
| CogAgent (3.5) | - | - | 18.6 | - |
| Synapse (3.5) | 34.0 | - | 30.6 | 2.4 |
| **AWM (3.5)** | 39.0 | 52.8 | 34.6 | 2.8 |
| MindAct (4) | 41.6 | 60.6 | 36.2 | 2.0 |
| **AWM (4)** | **50.6** | 57.3 | **45.1** | **4.8** |

**关键发现**：

1. AWM在step-level和task-level success rate上都最高
2. 提升主要来自element selection（element accuracy提高5.0-9.0点）
3. **与Synapse的比较**：Synapse retrieves具体examples，AWM使用抽象sub-routine，AWM效果更好
4. Action F1略低于MindAct：agent在识别何时该偏离workflow guidance方面仍有挑战

#### 3.2.2 Online AWM Enables Generalization

| Method | Cross-Task Step SR | Cross-Website Step SR | Cross-Domain Step SR |
|--------|-------------------|----------------------|---------------------|
| MindAct* | 36.2 | 30.1 | 18.6 |
| AWM_{offline} | 45.1 | 33.7 | 32.6 |
| AWM_{online} | 43.6 | 33.9 | 35.5 |

**关键发现**：

1. **In-domain (cross-task)**：两种AWM表现接近
2. **Cross-website和cross-domain**：随着domain gap增大，AWM_{online}优势逐渐显现
3. 两种AWM都大幅超过MindAct baseline

**批判性分析**：
- AWM_{offline}在cross-domain上用"随机选择所有domain的workflow"是weak baseline
- AWM_{online}实际上是在test set上"训练"的，可能存在implicit adaptation

---

## 4. Exploring Optimal Workflow Representations

### 4.1 How Much Does the Sub-routine, Abstract Format Contribute?

**实验动机**：验证LM-based induction中context abstraction和sub-routine extraction的价值。

**Rule-based Induction方法**：
1. 提取action sequence（如CLICK→CLICK→TYPE）
2. 按action sequence去重
3. 移除invalid action步骤

**WebArena结果**：

| Method | Total SR | # Steps |
|--------|----------|---------|
| AWM_{rule} | 35.6 | 6.3 |
| AWM_{lm} | 35.5 | 5.9 |

成功率几乎相同，但LM方法步数更少。

**Mind2Web结果**：

| Method | Elem Acc | Action F1 | Step SR | SR |
|--------|----------|-----------|---------|-----|
| AWM_{rule} | 49.5 | 57.0 | 43.4 | 2.0 |
| AWM_{lm} | 50.6 | 57.3 | 45.1 | 4.8 |

LM方法明显更好。

**分析**：对于结构化程度高的domain，rule-based可能足够；对于需要更强泛化的场景，LM-based abstraction更重要。

### 4.2 Workflows in Descriptive Texts

**实验动机**：比较程序格式 vs 文本格式的workflow。

**方法**：用GPT-3.5-turbo将action程序verbalize为自然语言。

| Method | Elem Acc | Action F1 | Step SR | SR |
|--------|----------|-----------|---------|-----|
| AWM | 50.6 | 57.3 | 45.1 | 4.8 |
| AWM_{text} | 51.2 | 57.4 | 45.4 | 3.6 |

**结论**：Text和code格式没有显著差异，都能有效augment agent memory。

### 4.3 Environment Abstraction in Workflows

**实验动机**：NL描述 vs HTML作为环境状态表示。

| Desc. | HTML | Elem Acc | Act F1 | Step SR | SR |
|-------|------|----------|--------|---------|-----|
| ✓ | ✗ | 39.0 | 52.8 | 34.6 | 2.8 |
| ✗ | ✓ | 38.1 | 54.0 | 33.8 | 2.8 |
| ✓ | ✓ | 37.1 | 51.3 | 32.9 | 2.0 |

**关键发现**：
- NL描述略优于filtered HTML
- 两者结合反而最差（context过长 + 信息可能冲突）
- Filtered HTML有47%的情况漏掉了正确元素

---

## 5. Exploring Workflow Utilization in Context and in Action

**AWM_{AS}方法**：将workflow wrap成可调用的high-level function。

例如，调用 `login(username, password)` 会依次执行：
```
click(box1-id) → type(box1-id, username) → click(box2-id) 
→ type(box2-id, password) → click(submit-id)
```

**结果**：

| Method | Elem Acc | Action F1 | Step SR | SR |
|--------|----------|-----------|---------|-----|
| MindAct | 41.6 | 60.6 | 36.2 | 2.0 |
| AWM | 50.6 | 57.3 | 45.1 | 4.8 |
| AWM_{AS} | 51.8 | 56.7 | 46.4 | 3.6 |

Agent只在18.5%的任务中调用workflow action。

**问题分析 (Figure 7)**：

预定义的action sequence无法适应dynamic环境变化（如pop-up选项）。

**结论**：作为action的workflow可以reinforce memory中的workflow，带来small extra gain，但不够flexible。未来需要real-time state access或dynamic execution loops。

---

## 6. Related Work

### 6.1 Web Agent Benchmarks

| Benchmark | 年份 | 特点 |
|-----------|------|------|
| MiniWob | 2017 | 最早的现代web agent benchmark |
| MiniWob++ | 2018 | 增加额外挑战 |
| WebShop | 2022 | 模拟电商网站 |
| WebArena | 2024 | 5个website，execution-based evaluation |
| VisualWebArena | 2024 | 扩展视觉输入任务 |
| Mind2Web | 2023 | 强调跨website/domain泛化 |

### 6.2 Enhancing Agents for Complex Tasks

**修改action space的工作**：
- 约束action search space
- LM self-feedback refine action
- 人工设计task-specific action

**Augment memory的工作**：
- In-context demonstrations

AWM的优势：不需要high-quality examples，可以在只有test queries时工作。

### 6.3 Learning Common Procedures from Experiences

**使用full examples的问题**：
- 与example-specific context纠缠
- 难以外推到其他task/domain

**提取reusable sub-routines的方法**：
- Rule-based方法
- LM-based方法
- 作为auxiliary skills

AWM的定位：探索了rule和LM两种方法，使用workflow作为context guidance避免environment grounding问题。

---

## 7. Conclusion

**主要贡献总结**：
- 提出AWM，可以offline从已有examples或online在inference时归纳、augment、使用workflow
- 在WebArena和Mind2Web上分别取得51.1%和24.6%的相对成功率提升
- 展示了跨task、website、domain的superior泛化能力

**未来方向**（implied）：
- Dynamic memory building
- Agent adaptation on varied digital tasks

---

## Appendix分析

### A. LM-based Workflow Induction

**A.1 Model Prompt**：

```
Given a list of web navigation tasks, your task is to extract the common workflows.
Each given task contains a natural language instruction, and a series of actions to solve the task.
You need to find the repetitive subset of actions across multiple tasks, and extract each of them out as a workflow.
Each workflow should be a commonly reused sub-routine of the tasks. Do not generate similar or overlapping workflows. Each workflow should have at least two steps. Represent the non-fixed elements (input text, button strings) with descriptive variable names as shown in the example.
```

**A.2 Example Workflows**：给出了各个website/domain的具体workflow示例。

**A.3 Workflow Quality Analysis**：

| Metric | WebArena | Mind2Web |
|--------|----------|----------|
| # Workflows | 7.4 | 7.3 |
| Coverage | - | 0.40 |
| Function overlap | 0.08 | 0.20 |
| Utility rate | 0.94 | 0.91 |

Coverage 0.40意味着60%的test步骤不被workflow覆盖，这可能是performance ceiling的一个来源。

### B. Rule-based Workflow Induction

两步process：
1. Experience deduplication
2. Invalid action filtering

### C. Integrating AWM Offline and Online

AWM_{off+on}的结果介于offline和online之间，没有additive effect。说明简单combine两种workflow不够，需要更sophisticated的integration策略。

---

## 总体评价

### 优点

| 方面 | 描述 |
|------|------|
| **Clear motivation** | 从人类学习workflow的认知科学角度motivate，intuitive且well-grounded |
| **Comprehensive experiments** | 在两个major benchmark上评估，覆盖offline/online settings，多个ablation studies |
| **Strong results** | 相对baseline有显著提升，且在challenging的cross-domain setting表现更突出 |
| **Ablation studies有价值** | 4.1-4.3节的实验提供了关于workflow representation的insight |

### 缺点

| 问题 | 描述 |
|------|------|
| **Sample order sensitivity** | Online setting下样本顺序对结果的影响未评估 |
| **Workflow selection机制缺失** | 全部塞进prompt的做法不scalable |
| **Evaluator dependency** | Online setting依赖external evaluator，其准确性直接影响workflow质量 |
| **Reproducibility concerns** | 缺少action generation的prompt template |
| **Limited domain** | 只在web navigation评估 |

### 对未来工作的启示

1. **更robust的online learning**：需要处理样本顺序sensitivity，可能需要某种curriculum或batch-based更新

2. **Relevance-based workflow retrieval**：随着workflow library增长，需要selective retrieval机制

3. **Self-correction**：需要机制来识别和修正错误workflow

4. **Hierarchical execution**：需要更flexible的workflow execution机制

---

## 核心技术点质疑

### 关于"running on WebArena"的问题

Online setting下存在潜在的数据泄露问题：agent在测试集上streaming地处理样本，先做第t个样本，成功了就提取workflow，然后用这个workflow帮助做第t+1个样本。

**问题1**：如果样本顺序不同，结果可能差异很大。论文没有报告不同random order下的variance。

**问题2**：虽然做了cross-template实验，但仍然是在同一个streaming过程中评估的，没有真正解决顺序依赖问题。

### 关于integrate机制的问题

从论文描述来看，就是**直接append**：

> "we add {w^t} into the agent memory M^t + {w^t} → M^{t+1}"

没有提到任何rerank或retrieval机制。只说按website分组来维护workflow集合。

### 关于workflow调取方式

就是**直接塞进prompt**。Section 2.1脚注：

> "Memory is usually implemented as a system prompt or auxiliary information in the main prompt context."

论文没有交代：
- 放在system prompt还是user prompt
- 是否有选择性地挑选相关workflow
- Action generation时的具体prompt template

这种"全部塞进去"的做法不scalable。

---

*报告完成*
