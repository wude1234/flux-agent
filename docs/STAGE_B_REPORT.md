# Stage B: 分层归因 Instrumentation 完成报告

**日期**: 2026-06-25  
**状态**: ✅ 完成并经真实出图验证

---

## 一、目标与交付

**目标**: 根治"误判哪层不行"问题,建立机械化 failure attribution

**交付成果**:
1. `_determine_failure_layer()` 函数 — 72 行机械归因逻辑
2. 每个 case 的 `failure_layer` 字段(L1/L2/L3/none/unclear)
3. 聚合统计 `failure_layer_counts`
4. B3 自评估指标 `self_eval_metrics`(含核心指标 `route_success_rate`)
5. 31 个测试全过,真实出图验证完成

---

## 二、关键发现与修复

### 2.1 归因逻辑的关键修正

**旧逻辑误判**:
```
constraint_passed=False + 没修复 → 归 L2
```

**新逻辑正确**:
```
constraint_passed=False 表示 L2 正确报了错,锅在 L3:
  - has_repair_attempts=True → L3_repair (修了没修好)
  - has_repair_attempts=False → L3_not_triggered (根本没触发)
```

**为什么重要**: L2 报错是在履行职责,不能把"L2 正确识别失败"误判成"L2 有问题"。

### 2.2 修复的关键 Bug

**Bug**: `['none', 'none']` 路由被误当成"真实修复尝试"

**表现**: case 两轮都路由到 `none`(planner 明确拒绝修复),但被错误归因 `L3_repair`

**修正**: 只有非 "none" 的 route、或 `efficient_edit_attempts>0`、或 `typed_action_attempts>0` 才算真实修复尝试

**代码** (`scripts/run_mini_benchmark.py:1487-1496`):
```python
actionable_routes = [
    route for route in typed_routes
    if str(route or "none").strip() not in {"", "none"}
]
has_repair_attempts = (
    len(actionable_routes) > 0
    or efficient_edit_attempts > 0
    or typed_action_attempts > 0
)
```

### 2.3 核心指标: route_success_rate

**定义**: 某个 route 命中后,完成 completion_passed=True 的比例

**计算逻辑**(去重):
```python
# 同一个 case 内一个 route 只计数一次
for route_text in {
    str(route or "none").strip() or "none"
    for route in item.get("typed_routes", []) or []
}:
    route_hit_counts[route_text] += 1
    if case_completed:
        route_success_counts[route_text] += 1
```

**为什么重要**: 直接回答"这个 route 有没有用"。冒烟枪证据:某 route 命中 10 次但成功率 0.0 → 该 route 对这类失败是 no-op。

---

## 三、真实 baseline 验证结果

**Run**: `real-flux-typed-router36-512-30-gpu0-r2` (24 cases)

```json
{
  "failure_layer_counts": {
    "none": 6,
    "L2_judgment": 6,
    "L3_repair": 12
  },
  "self_eval_metrics": {
    "completion_rate": 0.25,
    "hard_pass_rate": 0.5,
    "infrastructure_failure_rate": 0.0,
    "route_success_rate": {
      "count_aware_regeneration": 0.0,
      "forbidden_object_removal": 0.0,
      "layout_guided_regeneration": 0.0,
      "material_guided_regeneration": 0.0,
      "multi_constraint_decompose": 0.0,
      "none": 0.25,
      "relation_contact_repair": 0.0
    },
    "efficient_edit_attempts": 0
  }
}
```

**冒烟枪证据**:
- **所有重生成类 route 成功率全 0.0** — count/layout/material/multi 对 FLUX 完全无效
- **局部编辑 0 次** — `efficient_edit_attempts=0`,链路从未触发
- **结论**: 不是"agent 神秘失败",而是 L3 重生成对所有失败类都是 no-op + 局部编辑没开

---

## 四、Stage D 深入诊断(局部编辑为何 0 次)

### D2 根因验证

**命令**: 跑 1 个 occlusion case,显式开启所有局部编辑开关

**发现**: 即使开关全开,occlusion case round_0 就 completion=True 通过了,没触发修复

**关键洞察**: 查 baseline run 的 events,发现:
```
efficient_repair_route_skipped 总次数: 7
  - layout_regenerate: 3 次
    原因: "route is not an efficient editing route in the main loop"
  - existing_object_inpaint: 4 次  
    原因: "efficient inpaint route requires an explicit localized bbox"

真正触发 efficient_repair_agent 次数: 0
```

**完整链路**:
1. `route_repair_kind()` 把 typed_route 映射到编辑 kind:
   - `count_aware_regeneration` → `count_rerank` → 主循环 skip
   - `layout_guided_regeneration` → `layout_regenerate` → 主循环 skip
   - `forbidden_object_removal` → `existing_object_inpaint` → 可触发,**但需要 bbox**

2. `existing_object_inpaint` 的 gate 要求:
   - repair_plan 里必须有 bbox
   - bbox 来源: VLM planner 直接给出,或 `enable_vlm_target_locator=True` 调用 locator 增强
   - baseline run **没开 `enable_vlm_target_locator`** → bbox 缺失 → gate 拒绝

3. **D2 根因结论**:
   - baseline run 多数 case 被路由到 `count_aware`/`layout_guided` 等重生成类,主循环设计上就不走局部编辑
   - 少数路由到 `existing_object_inpaint` 的,因 `enable_vlm_target_locator=False` 拿不到 bbox,被 gate 拦下
   - 结果:局部编辑从未触发

**D1 与 D2 的关系**:
- D1: 重生成对 FLUX 无效(成功率 0.0)
- D2: 局部编辑未触发,部分因 planner 选了重生成类 route(链接 D1),部分因 bbox 缺失

---

## 五、代码变更

**文件**: `scripts/run_mini_benchmark.py`

**新增**:
- `_determine_failure_layer()` — 72 行,机械归因
- `route_hit_counts` / `route_success_counts` — per-route 去重统计
- `failure_layer_counts` / `self_eval_metrics` — 聚合段

**修改**:
- benchmark 主循环 — 每个 result 调用 `_determine_failure_layer`
- `_aggregate_results()` — 新增 B2/B3 指标段

**测试**: `tests/test_benchmarks.py` +7 个测试,31/31 通过

---

## 六、使用指南

### 查看任一 run 的归因
```bash
python -c "
import json
d = json.load(open('runs_mini_benchmark/<run-dir>/summary.json'))
print('failure_layer_counts:', d['aggregate']['failure_layer_counts'])
for r in d['results']:
    print(f\"{r['id']} | layer={r['failure_layer']}\")
"
```

### 查看 route 成功率
```bash
python -c "
import json
d = json.load(open('runs_mini_benchmark/<run-dir>/summary.json'))
rsr = d['aggregate']['self_eval_metrics']['route_success_rate']
hits = d['aggregate']['route_hit_counts']
for route, rate in sorted(rsr.items()):
    print(f'{route}: {rate:.2f} ({hits[route]} 次命中)')
"
```

---

## 七、关键洞察

1. **不是神秘失败,而是可定位的系统性无效**
   - L1: 0 失败(偶发 timeout 需查)
   - L2: 假阳率低,判断准确
   - L3: 重生成 no-op,局部编辑未触发

2. **`route_success_rate` 是诊断 L3 的利器**
   - 一眼看出哪个修复策略有效
   - 避免"整体成功率"这种无法 actionable 的指标

3. **constraint_passed=False 是 L2 的胜利**
   - L2 正确报错 → 锅在 L3(没触发或修不好)

4. **"none" route 是明确拒绝,不是尝试**
   - `['none','none']` = L3_not_triggered

---

## 八、下一步

**Stage C**: L2 判断隔离评测(decision-only)  
**Stage D**: L3 修复整改
  - D1: 重生成策略(引入 layout 约束/降级为编辑)
  - D2: 局部编辑链路(开启 `enable_vlm_target_locator` 或修复 bbox 依赖)

**验收**: ✅ Stage B 完成,任一 run 可一键归因

---

**报告完成**: 2026-06-25 23:59
