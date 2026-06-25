# Stage C: L2 判断隔离评测 — 完成报告

**日期**: 2026-06-26  
**状态**: ✅ 诊断完成,修复已实施,验证进行中

---

## 一、目标

隔离测试 L2 VQA 判断质量,定位假阳/假阴率,避免 L3 污染归因。

---

## 二、诊断过程

### C1: 复用 baseline run 数据

使用 `real-flux-typed-router36-512-30-gpu0-r2`(24 cases)的现有数据,避免重新烧 GPU。

### C2: 硬 VQA vs 软评估对比

提取每个 case 的 `constraint_passed`(硬 VQA)、`evaluation_passed`(软评估)、`completion_passed`,定位脱节 case。

**初步发现**: 6 个 case 硬 VQA 过了但 completion 没过,怀疑是软评估错误卡住。

| case | category | hard VQA | eval | completion | 初步归因 |
|---|---|---|---|---|---|
| all10_negation_004 | negation_absence | pass(1.0) | fail(0.3) | fail | 软评估卡住? |
| all10_negation_006 | negation_absence | pass(1.0) | fail(0.5) | fail | 软评估卡住? |
| all10_negation_010 | negation_absence | pass(1.0) | fail(0.3) | fail | 软评估卡住? |
| all10_interaction_005 | interaction_relation | pass(0.888) | fail(0.75) | fail | 软评估卡住? |
| all10_count_006 | count_quantity | **fail(0.755)** | fail(0.85) | fail | ✓ 归因正确 |
| all10_color_001 | color_binding | **fail(0.755)** | fail(0.7) | fail | ✓ 归因正确 |

### C3: 深挖 negation case 的 eval errors

**关键发现反转假设**:

查看 negation_004/006/010 的 `evaluation_round_*.json` 中的 `errors` 字段:

```json
// negation_006: "no bowl and no spoon nearby"
{
  "evidence": "The image contains a bowl and a cracker-like object, which violates the user's instruction...",
  "type": "missing_object",
  "severity": "major"
}

// negation_010: "no window and no sign"
{
  "evidence": "The door has a round window...",
  "type": "wrong_attribute"
}

// negation_004: "no visible zipper pull and no side pocket"
{
  "evidence": "The backpack has visible zipper pulls on the side pockets...",
  "type": "wrong_attribute"
}
```

**结论反转**: 不是"软评估错误卡住",而是:
- 图里**真的有禁止物**(bowl、window、zipper pull 都出现了)
- 软评估(factuality)**抓对了**真实违规
- **硬 VQA 才是误判的一方**:它给了 pass=1.0,但根本没检查禁止物

### C4: 检查硬 VQA 生成的问题

提取 `constraint_check` 的 `checks` 字段:

```python
# negation_006: "no bowl and no spoon"
硬 VQA 问题:
  - entity_existence | cereal box | expected=yes observed=yes
  - entity_existence | counter | expected=yes observed=yes
  - count | cereal box | expected=1 observed=1
  - color_binding | cereal box | expected=yellow observed=yellow
  # ❌ 完全没有 "bowl absent?" "spoon absent?" 的检查

# negation_010: "no window and no sign"
硬 VQA 问题:
  - entity_existence | door | expected=yes observed=yes
  - count | door | expected=1 observed=1
  - color_binding | door | expected=blue observed=blue
  # ❌ 完全没有 "window absent?" "sign absent?" 的检查
```

**根因坐实**: 硬 VQA 对 negation 约束**系统性盲区**,对禁止物 0 检查 → 全判 pass(假阳性)。

---

## 三、根因分析

### 问题定位: `constraint_questions.py` 缺对象存在否定处理

**现有代码只处理两类 negation**:

1. `_negative_relation_constraints` (Line 1357-1401):
   - 处理关系否定:"X is not touching/attached to Y"
   - 正则: `r"\b(?P<subject>...)\s+(?:is|are)?\s*not\s+(?P<relation>attached|touching)..."`
   - **匹配不到**: "no bowl", "no window"

2. `_negative_symbol_text_constraints` (Line 1404-1442):
   - 处理符号文字否定:"no text/symbol on X"
   - 正则: `r"\b(no|without)\b"` 且要求含 `symbol|text|logo|mark` 关键词
   - **匹配不到**: "no bowl", "no window"(不含 symbol/text 关键词)

**缺失类型**: **对象存在否定** — "no X", "no Y", "with no Z", "without X"

这是**最常见的 negation 类型**,但完全没有函数处理。

### 影响面

**真实 baseline run(24 case)**:
- **4 个假阳性**,全因 negation 盲区:
  - negation_004/006/010 (negation_absence 类)
  - interaction_005 ("without sitting inside")
- negation_absence 类别(4 case): **3/4 中招**
- interaction 类别(4 case): **1/4 中招**

---

## 四、修复实施

### 新增函数: `_negative_object_existence_constraints`

**位置**: `src/constraint_questions.py:1443-1528`

**功能**: 从 negative_constraints 提取禁止对象,生成 "X absent?" 检查

**处理模式**:
- "no bowl and no spoon" → ["bowl", "spoon"]
- "no window and no sign" → ["window", "sign"]
- "with no visible zipper pull and no side pocket" → ["zipper pull", "side pocket"]

**关键正则** (Line 1478-1482):
```python
for match in re.finditer(
    r"\bno\s+(?:visible\s+)?(?P<obj>(?:[a-z0-9-]+)(?:\s+(?!and\b|or\b|but\b|with\b)[a-z0-9-]+){0,3})\b",
    text
):
```
- 匹配 "no <object>",在连词(and/or/but/with)处停止
- 允许多词对象名(zipper pull, side pocket)
- 过滤副词(nearby, inside)和动词(sitting, standing)

**生成问题格式**:
```python
{
  "id": "negative_existence:bowl:absent",
  "category": "negative_object_existence",
  "question": "Is the bowl absent from the image?",
  "expected": "yes",  # yes, it's absent
  "object": "bowl",
  "negative": True,
  "type": "forbidden_object",
}
```

### 集成点 (Line 454-467):
```python
relations.extend(
    _negative_object_existence_constraints(
        (
            constraints.intent_spec.negative_constraints
            if constraints.intent_spec is not None
            else []
        ),
        entities,
    )
)
```

### 配套修改

**1. 新 category 注册** (Line 71):
```python
KNOWN_CATEGORIES = {
    ...,
    "negative_object_existence",  # ← 新增
}
```

**2. VQA instruction** (Line 529):
```python
"For negative_object_existence questions, answer yes only if the specified forbidden object is completely absent from the entire image; any visible instance of that object class means the answer is no.",
```

**3. 错误类型映射** (Line 2536, 2553):
```python
# 问题类型映射
"negative_object_existence": "relation",

# 错误类型映射
"negative_object_existence": "forbidden_object_present",
```

---

## 五、测试覆盖

**文件**: `tests/test_negative_object_existence.py`

**新增测试**: 7 个,全部通过 ✅

1. `test_negative_object_existence_no_x_pattern` — "no bowl and no spoon"
2. `test_negative_object_existence_no_window_sign` — "no window and no sign"
3. `test_negative_object_existence_with_no_visible` — "with no visible zipper pull"
4. `test_negative_object_existence_skip_verbs` — 过滤动词("sitting")
5. `test_negative_object_existence_skip_symbol_text` — 跳过 symbol/text 类(避免重复)
6. `test_negative_object_existence_skip_relation` — 跳过关系否定(避免重复)
7. `test_negative_object_existence_without_pattern` — "without X" 模式

**完整测试套件**: 416/416 通过 ✅

---

## 六、验证计划

### V1: decision-only run (进行中)

**命令**:
```bash
python -m src.run_m4 \
  --prompt "A yellow cereal box stands on a counter with no bowl and no spoon nearby." \
  --decision-only \
  --runs-dir runs_mini_benchmark/stageC-negation-verify \
  --run-id negation-006-fixed
```

**预期结果**:
- `constraint_check` 中出现:
  ```json
  {"category": "negative_object_existence", "object": "bowl", "question": "Is the bowl absent?", ...}
  {"category": "negative_object_existence", "object": "spoon", "question": "Is the spoon absent?", ...}
  ```
- 如果图里真的有 bowl/spoon,硬 VQA 应该 fail(修复假阳性)

### V2: 重跑 4 个 false-pass case

修复验证后,重跑 negation_004/006/010 + interaction_005,确认:
- 硬 VQA 从 pass 变 fail(假阳性消除)
- 硬 VQA 与软评估一致

---

## 七、关键洞察

### 1. 假设反转的价值

**旧假设**(PLAN): negation 类是"软评估错误卡住 case"  
**真相**: 软评估**抓对了**,硬 VQA 有系统性盲区

**启示**: 不要只看指标(hard pass vs eval fail),要看**实际问题和 evidence**。

### 2. constraint_questions.py 的设计债

现有三个 negative 处理函数各自独立:
- `_negative_relation_constraints`: 关系否定
- `_negative_symbol_text_constraints`: 符号文字否定
- `_negative_object_existence_constraints`: **对象存在否定(新增)**

**缺失最常见类型**长达数月未被发现,说明:
- 缺 negation 类的系统性测试覆盖
- 缺人工抽查 VQA 问题生成质量的流程

### 3. 硬 VQA 假阳性的危害

硬 VQA 假阳性 → L2 判断 pass → L3 不触发修复 → case 直接失败

**比假阴性更致命**:
- 假阴性:硬 VQA 误报错误 → L3 尝试修复(可能浪费资源,但不会遗漏问题)
- 假阳性:硬 VQA 漏报错误 → L3 根本不知道有问题 → **直接放行错误图**

negation 盲区导致的 4 个假阳性,使 baseline run 成功率从 25% 虚高到 **实际应该更低**(如果硬 VQA 准确,这 4 个本该触发 L3 修复)。

### 4. 软评估的价值

软评估(factuality)在这次诊断中**救了**分析:
- 硬 VQA 全部 pass → 看起来一切正常
- 软评估 fail → 触发深挖 → 发现图里真的有禁止物

**启示**: 软评估不是"卡住 case 的累赘",而是**硬 VQA 的重要校验层**。

---

## 八、文件变更清单

### 代码
- `src/constraint_questions.py`: +93 行
  - `_negative_object_existence_constraints()` — 新函数(85 行)
  - 调用点集成(8 行)
  - category 注册、instruction、错误映射(3 处修改)

### 测试
- `tests/test_negative_object_existence.py`: +100 行(7 个新测试)
- 完整测试套件: 416/416 通过 ✅

### 文档
- `docs/STAGE_C_REPORT.md` — 本报告

**总计**: +193 行代码和测试

---

## 九、下一步

### 短期(Stage C 收尾)
- [ ] 等 V1 验证完成,确认硬 VQA 生成了 "bowl/spoon absent" 问题
- [ ] 重跑 4 个 false-pass case,确认假阳性消除

### 中期(Stage D2)
- [ ] 开启 `enable_vlm_target_locator`,修复局部编辑链路

### 长期(Stage D1)
- [ ] 重生成策略改造(引入 layout 约束/降级为编辑)

---

## 十、Stage C 总结

**诊断完成** ✅:
- 定位 L2 假阳性根因:constraint_questions.py 缺对象存在否定处理
- 影响面:4/24 case 假阳性,全因 negation 盲区

**修复实施** ✅:
- 新增 `_negative_object_existence_constraints` 函数
- 7/7 单元测试通过,416/416 完整测试通过

**验证进行中** 🔄:
- decision-only run 验证硬 VQA 问题生成
- 待补充验证结果

**关键发现**:
- 假设反转:不是软评估错,而是硬 VQA 盲区
- 硬 VQA 假阳性比假阴性更致命
- 软评估是硬 VQA 的重要校验层

---

**报告完成**: 2026-06-26 02:10  
**验证状态**: V1 进行中,待补充结果  
**下一步**: 等验证完成 → Stage D2(开 locator)
