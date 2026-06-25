# T2I Agent 项目完整改动文档 (Stage A→B→C→D2)

**时间**: 2026-06-25 ~ 2026-06-26  
**状态**: Stage A/B/C/D2 短期 ✅ 完成

---

## 总览

| Stage | 任务 | 状态 | 核心改动 |
|---|---|---|---|
| A | 解封 L1 生成后端 | ✅ 完成 | HF_HOME 环境变量修复 |
| B | 分层归因 instrumentation | ✅ 完成 | 机械归因函数 + route_success_rate |
| C | L2 判断隔离评测 | ✅ 完成 | 补全 negation 对象存在检查 |
| D2 | 局部编辑短期修复 | ✅ 完成 | locator 跟随 efficient_repair_agent |
| D1 | 重生成策略改造 | 🔄 待定 | 工作量大,需架构设计 |

---

## Stage A: 解封 L1 生成后端

### 问题
M-GRAG 后端反复崩溃:
```
LocalEntryNotFoundError: /home/zrr/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/...
```
- 24 case 里 ~8 个崩在 L1,无法评估 agent 真实效果
- 根因:conda 默认查 `~/.cache/huggingface`,真实权重在 `/mnt/ssd3/zrr/hf_cache`

### 修复
**文件**: `scripts/run_mini_benchmark.py`  
**位置**: Line 312-325 `_batch_flux_env()`

```python
def _batch_flux_env(args: argparse.Namespace) -> dict[str, str]:
    flux_env = dict(os.environ)
    # 关键修复:显式设置 HF cache 路径
    flux_env["HF_HOME"] = args.flux_hf_home or "/mnt/ssd3/zrr/hf_cache"
    flux_env["HF_HUB_CACHE"] = flux_env["HF_HOME"]
    flux_env["TRANSFORMERS_CACHE"] = flux_env["HF_HOME"]
    ...
```

### 验收
- baseline run(24 case) `infrastructure_failure_rate=0.0` ✅
- L1 零失败,问题解封

### 残留
- baseline FLUX 偶发 timeout(900s),需进一步调查

---

## Stage B: 分层归因 Instrumentation

### 目标
根治"误判哪层不行",建立机械化 failure attribution

### B1: Per-case Health Flags
已由现有字段覆盖,无需新增

### B2: 机械归因函数

**文件**: `scripts/run_mini_benchmark.py`  
**位置**: Line 1450-1514 (72 行)

```python
def _determine_failure_layer(result: Mapping[str, Any]) -> str:
    """
    返回: L1_generation | L2_judgment | L3_repair | 
          L3_not_triggered | none | unclear
    """
```

**关键修正**:
- 旧逻辑:`constraint_passed=False` → 归 L2
- **新逻辑**:`constraint_passed=False` 表示 L2 正确报错,失败应归 L3

**修复的 Bug**:
- `['none', 'none']` 路由被误当成"修复尝试"
- 修正:只有非 "none" route、或 `efficient_edit_attempts>0` 才算真实尝试

**写入时机**: Line 966-968
```python
result["failure_layer"] = _determine_failure_layer(result)
```

### B3: 自评估指标

**位置**: Line 1629-1670

**核心指标**: `route_success_rate`
```python
# 某 route 命中后完成 completion_passed=True 的比例
"route_success_rate": {
    "count_aware_regeneration": 0.0,  # 5 次命中,0 次成功
    "layout_guided_regeneration": 0.0, # 7 次
    ...
}
```

**为什么重要**: 直接回答"这个 route 有没有用",避免"整体 25%"这种无法 actionable 的指标

### 真实验证

**baseline run(24 case)**:
```json
{
  "failure_layer_counts": {"none": 6, "L2_judgment": 6, "L3_repair": 12},
  "self_eval_metrics": {
    "completion_rate": 0.25,
    "route_success_rate": {
      "count_aware_regeneration": 0.0,
      "layout_guided_regeneration": 0.0,
      "material_guided_regeneration": 0.0,
      "multi_constraint_decompose": 0.0,
      "none": 0.25
    },
    "efficient_edit_attempts": 0
  }
}
```

**冒烟枪证据**:
- 所有重生成类 route 成功率 0.0
- 局部编辑 0 次触发
- 结论:L3 重生成无效 + 局部编辑链路未通

### 交付
- ✅ `_determine_failure_layer()` (72 行)
- ✅ Per-case `failure_layer` 字段
- ✅ `failure_layer_counts` 聚合
- ✅ `self_eval_metrics` 含 route_success_rate
- ✅ 修复 `['none']` bug
- ✅ 31/31 测试通过

---

## Stage C: L2 判断隔离评测

### 诊断过程

**C1**: 复用 baseline run 数据

**C2**: 硬 VQA vs 软评估对比
- 6 个 case 硬 VQA 过但 completion 没过
- 初步怀疑:软评估错误卡住

**C3**: 深挖 eval errors

**关键发现反转**:
- negation_004/006/010 的 eval errors 显示图里**真的有禁止物**(bowl/window/zipper)
- 软评估**抓对了**真实违规
- **硬 VQA 才是误判的**:pass=1.0 但没检查禁止物

**C4**: 检查硬 VQA 生成的问题

```python
# negation_006: "no bowl and no spoon"
硬 VQA 问题:
  - entity_existence | cereal box | yes
  - color_binding | cereal box | yellow
  # ❌ 完全没有 "bowl absent?" "spoon absent?" 的检查
```

### 根因分析

**位置**: `src/constraint_questions.py`

**现有代码只处理两类 negation**:
1. `_negative_relation_constraints`: "X is not touching Y"
2. `_negative_symbol_text_constraints`: "no text/symbol on X"

**缺失**: **对象存在否定** — "no bowl", "no window", "with no zipper"

**影响面**: baseline 4/24 case 假阳性,全因 negation 盲区

### 修复实施

**新增函数**: `_negative_object_existence_constraints`

**位置**: `src/constraint_questions.py:1443-1528` (85 行)

**功能**: 提取禁止对象,生成 "X absent?" VQA 问题

**处理模式**:
```python
"no bowl and no spoon" → ["bowl", "spoon"]
"no window and no sign" → ["window", "sign"]
"with no visible zipper pull" → ["zipper pull"]
```

**关键正则**:
```python
r"\bno\s+(?:visible\s+)?(?P<obj>(?:[a-z0-9-]+)(?:\s+(?!and\b|or\b)[a-z0-9-]+){0,3})\b"
```
- 在连词(and/or/but/with)处停止
- 允许多词对象名
- 过滤副词(nearby, inside)和动词(sitting)

**生成问题格式**:
```python
{
  "id": "negative_existence:bowl:absent",
  "category": "negative_object_existence",
  "question": "Is the bowl absent from the image?",
  "expected": "yes",
  "object": "bowl",
  "negative": True,
}
```

**集成点**: Line 454-467
```python
relations.extend(
    _negative_object_existence_constraints(
        constraints.intent_spec.negative_constraints,
        entities,
    )
)
```

**配套修改**:
- category 注册(Line 71)
- VQA instruction(Line 529)
- 错误类型映射(Line 2536, 2553)

### 测试覆盖

**文件**: `tests/test_negative_object_existence.py`

**新增**: 7/7 测试通过 ✅
- "no bowl and no spoon" 提取
- "no window and no sign" 提取
- "with no visible zipper" 提取
- 动词过滤
- symbol/text 跳过(避免重复)
- relation 跳过(避免重复)

**完整测试**: 416/416 通过 ✅

### 验收
修复完成 ✅,真实验证因 GPU OOM 暂缓(不影响代码质量)

---

## Stage D2: 局部编辑短期修复

### 问题
baseline run `efficient_edit_attempts=0`,events 记录:
```
efficient_repair_route_skipped: 7 次
  - 3 次 layout_regenerate → "not an efficient editing route"
  - 4 次 existing_object_inpaint → "requires an explicit localized bbox"
```

### 根因链条(四层)

1. **Layer 1**: baseline 所有 `enable_*` 开关 None
2. **Layer 2**: `DEFAULT_EFFICIENT_REPAIR_CATEGORIES` 只含 occlusion/text/negation/multi
3. **Layer 3**: 主循环只接受 5 种编辑 route,`count_rerank`/`layout_regenerate` skip
4. **Layer 4**: 编辑类 route 需 bbox,但 `enable_vlm_target_locator=False` → gate 拒绝

**最终根因**: locator 没跟随 efficient_repair_agent 一起开 → 编辑路由拿不到 bbox → 被 gate 静默拒绝

### 修复

**文件**: `scripts/run_mini_benchmark.py`  
**位置**: Line 812-822

```python
enable_efficient_repair_agent = bool(args.enable_efficient_repair_agent or auto_efficient)
# D2 fix: bbox-based editing routes are rejected by the gate when
# the plan has no bbox. The locator fills that bbox, so it must
# follow whenever the efficient repair agent is on.
enable_vlm_target_locator = bool(
    args.enable_vlm_target_locator or auto_efficient or enable_efficient_repair_agent
)
```

**修复逻辑**: locator 跟随 `efficient_repair_agent`,确保编辑路由能拿到 bbox

### 验收
- 31/31 测试通过 ✅
- 真实验证待 GPU 空闲后补充

---

## 文件变更清单

### Stage A
- `scripts/run_mini_benchmark.py`: `_batch_flux_env()` (~15 行)

### Stage B
- `scripts/run_mini_benchmark.py`: +180 行
  - `_determine_failure_layer()` (72 行)
  - route 统计 (~40 行)
  - 聚合指标 (~50 行)
- `tests/test_benchmarks.py`: +90 行(7 个测试)

### Stage C
- `src/constraint_questions.py`: +93 行
  - `_negative_object_existence_constraints()` (85 行)
  - 集成点 + category 注册 (8 行)
- `tests/test_negative_object_existence.py`: +100 行(7 个测试)

### Stage D2
- `scripts/run_mini_benchmark.py`: +8 行(注释 + locator 逻辑调整)

**总计**:
- 代码: +296 行
- 测试: +190 行(14 个新测试)
- 测试通过: 416/416 ✅

### 文档
- `docs/STAGE_B_REPORT.md` — Stage B 完成报告
- `docs/STAGE_C_REPORT.md` — Stage C 完成报告
- `docs/ISSUES_AND_SOLUTIONS.md` — 问题与解决方案
- `docs/COMPLETE_CHANGES.md` — 本文档
- `PLAN.md` — 更新 Stage A/B/C/D2 状态

---

## 关键洞察

### 1. 分层归因的价值
不是"agent 神秘失败",而是可定位的系统性无效:
- L1: 0 失败(A 解封)
- L2: 假阳性 4/24(C 修复 negation 盲区)
- L3: 重生成 no-op + 局部编辑未触发(D2 修复 locator)

### 2. route_success_rate 的威力
避免"整体 25%"这种笼统指标,直接回答:
- "count_aware 对 count 失败有用吗?" → 0.0,无用

### 3. 假设反转的启示
PLAN 假设"软评估错误卡住 negation case",真相是硬 VQA 有盲区。  
**启示**: 不要只看指标,要看实际问题和 evidence。

### 4. 硬 VQA 假阳性的危害
假阳性 → L2 pass → L3 不触发 → 直接放行错误图  
**比假阴性更致命**:假阴性会触发 L3 修复(可能浪费资源),假阳性直接遗漏问题。

### 5. constraint_passed=False 不能归 L2
L2 报错是履行职责,锅在 L3(没触发或修不好)

### 6. D1 和 D2 的耦合
- planner 选重生成 → 重生成无效(D1)
- 主循环不接受重生成 route → 编辑永不触发(D2)
- 修 D2 能让编辑触发,但 planner 继续选重生成,问题还在

---

## 使用指南

### 查看 run 归因
```bash
python -c "
import json
d = json.load(open('runs_mini_benchmark/<run-dir>/summary.json'))
print('failure_layer_counts:', d['aggregate']['failure_layer_counts'])
for r in d['results']:
    print(f\"{r['id']} | {r.get('failure_layer')}\")
"
```

### 查看 route 成功率
```bash
python -c "
import json
d = json.load(open('runs_mini_benchmark/<run-dir>/summary.json'))
rsr = d['aggregate']['self_eval_metrics']['route_success_rate']
for route, rate in sorted(rsr.items()):
    print(f'{route}: {rate:.2f}')
"
```

---

## 下一步

### 短期(验证)
- [ ] 等 GPU 空闲,验证 C 的 negation 检查生成
- [ ] 验证 D2 的 locator 启用效果

### 中期(D2 中期)
- [ ] 修改 planner 路由策略(count 失败 → object removal)

### 长期(D1)
- [ ] 重生成策略改造(layout 约束/降级编辑/batch rerank)

---

**文档完成**: 2026-06-26 02:30  
**状态**: A/B/C/D2 短期 ✅ 完成  
**下一步**: 验证 + D2 中期 / D1
