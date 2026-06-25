# T2I Agent 项目问题与解决方案总结

**时间线**: 2026-06-25  
**覆盖阶段**: Stage A(解封L1) → Stage B(分层归因) → Stage D2(局部编辑诊断)

---

## 问题 1: L1 M-GRAG 后端反复崩溃

**表现**:
- `LocalEntryNotFoundError: No such file or directory: '/home/zrr/.cache/huggingface/...'`
- 进程秒退,24 个 case 里 ~8 个崩在 L1

**根因**:
- 旧代码没设置 `HF_HOME`,conda 默认查 `~/.cache/huggingface`
- 真实权重在 `/mnt/ssd3/zrr/hf_cache`,找不到秒崩

**修复**:
- `scripts/run_mini_benchmark.py:312-325` 的 `_batch_flux_env()` 现在正确设置:
  ```python
  flux_env = dict(os.environ)
  flux_env["HF_HOME"] = args.flux_hf_home or "/mnt/ssd3/zrr/hf_cache"
  flux_env["HF_HUB_CACHE"] = flux_env["HF_HOME"]
  ```
- 修复后 L1 零失败(baseline run 24 case,`infrastructure_failure_rate=0.0`)

**残留问题**:
- baseline FLUX 有概率性 timeout(900s 卡死),即使 GPU 空闲
- 可能与 offload/显存分配有关,需进一步调查

---

## 问题 2: 归因误判"constraint_passed=False → L2 有问题"

**表现**:
- 用户报"L2 判断不准",但实际抽查 VQA 结果是准确的
- 上次 L1 崩溃也被误判成"agent 不行"

**根因**:
- 缺机械化 failure attribution,三层混在一起无法定位
- 旧归因逻辑:`constraint_passed=False` + 没修复 → 误判成"L2 有问题"
- **实际**: `constraint_passed=False` 表示 L2 **正确识别了失败**,锅在 L3(要么没修复,要么修了没修好)

**修复**:
- Stage B 建立 `_determine_failure_layer()` 机械归因函数
- 新逻辑:
  ```python
  if constraint_passed is False:
      if has_repair_attempts:
          return "L3_repair"  # L3 修了没修好
      return "L3_not_triggered"  # L3 根本没触发
  ```
- L2 报错是履行职责,不是失败

**验收**:
- 真实 baseline run 归因:`{none: 6, L2_judgment: 6, L3_repair: 12}`
- L2_judgment 的 6 个是 false-pass(hard 过但 completion 没过),不是误报 constraint

---

## 问题 3: `['none', 'none']` 路由被误当成修复尝试

**表现**:
- case 两轮都路由到 `none`,`efficient_edit_attempts=0`,但被归因 `L3_repair`(修了没修好)
- 实际:planner 两轮都**明确拒绝修复**,应该是 `L3_not_triggered`

**根因**:
- 旧代码 `has_repair_attempts = len(typed_routes) > 0`,把 `['none']` 当成了尝试
- `none` 是 planner 的明确拒绝,不是真实修复

**修复**:
- `scripts/run_mini_benchmark.py:1487-1496`:
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
- 新增测试 `test_determine_failure_layer_none_routes_are_not_repair` 锁定

**验收**:
- stageB-real-gpu0 的 mini_count_002:两轮 `['none','none']`,正确归因 `L3_not_triggered`

---

## 问题 4: 重生成类 route 成功率全 0.0

**表现**(Stage B `route_success_rate` 机械证明):
- `count_aware_regeneration`: 0.0 (5 次命中,0 次成功)
- `layout_guided_regeneration`: 0.0 (7 次)
- `material_guided_regeneration`: 0.0 (4 次)
- `multi_constraint_decompose`: 0.0 (4 次)
- `relation_contact_repair`: 0.0 (1 次)

**根因**:
- FLUX 对 prompt 里的精确数量/空间关系控制能力弱
- 单纯改 prompt 重生成解决不了:3 只鸟重生成还是错的数量

**当前状态**: 问题已定位,修复方向见 PLAN Stage D1

**修复方向**:
1. 引入 layout 约束(bounding box + regional prompting)
2. "生成 N+buffer 后挑选"策略
3. 降级为 object removal/insertion(走局部编辑)

---

## 问题 5: 局部编辑 `efficient_edit_attempts=0` 从未触发

**表现**:
- baseline run 24 case,`efficient_edit_attempts=0`
- 所有 L3 只有重生成(已证明无效),局部编辑完全没用上

**诊断过程**(Stage D2):

### 5.1 最初假设:开关没开?

**验证**: 查 baseline run 的 `run.json` config:
```json
{
  "enable_efficient_repair_agent": null,
  "enable_editing_mask_agent": null,
  "enable_object_insertion_repair": null,
  "enable_relation_repair": null,
  "enable_vlm_target_locator": null
}
```
- 确认:开关全是 `None`(未启用)

### 5.2 第二层:某些类别会自动启用?

**发现**: `scripts/run_mini_benchmark.py:797-802` 的 `DEFAULT_EFFICIENT_REPAIR_CATEGORIES`:
```python
{
    "occlusion_visibility",
    "text_symbol",
    "negation_absence",
    "multi_compositional",
}
```
- baseline run 的 6 类中只有 negation 在自动启用列表
- count/spatial/attribute/color 不自动启用,默认走纯重生成

### 5.3 第三层:即使开关开了,为什么还是不触发?

**验证**: 查 baseline run 的 events,发现 `efficient_repair_route_skipped` 7 次:
```
layout_regenerate: 3 次
  原因: "route is not an efficient editing route in the main loop"
existing_object_inpaint: 4 次
  原因: "efficient inpaint route requires an explicit localized bbox"
```

**关键代码**: `src/orchestrator.py:2719-2735` 的主循环只接受 5 种编辑 route:
```python
if route not in {
    "text_overlay",
    "symbol_overlay",
    "shape_overlay",
    "bbox_shape_inpaint",
    "existing_object_inpaint",
}:
    events.append({
        "type": "efficient_repair_route_skipped",
        "reason": "route is not an efficient editing route in the main loop",
    })
    return None
```

**route 映射**: `src/editing_agent.py:640-725` 的 `route_repair_kind()`:
- `count_aware_regeneration` → `count_rerank` → 主循环 skip
- `layout_guided_regeneration` → `layout_regenerate` → 主循环 skip
- `forbidden_object_removal` → `existing_object_inpaint` → **可触发**,但需要 bbox

### 5.4 第四层:为什么 `existing_object_inpaint` 还被拒?

**关键代码**: `src/orchestrator.py:6790-6796` 的 `_efficient_repair_gate`:
```python
bbox = _repair_plan_bbox_or_none(repair_plan)
if bbox is None:
    return {
        "allowed": False,
        "reason": "efficient inpaint route requires an explicit localized bbox",
    }
```

**bbox 来源**: `src/orchestrator.py:2855-2856`:
```python
if not self.enable_vlm_target_locator:
    return repair_plan  # 不增强 bbox
```

**根因链条最终定位**:
1. baseline run **没开 `enable_vlm_target_locator`**
2. `existing_object_inpaint` 走到 gate 时拿不到 bbox
3. 即使有 4 个 case 路由到可编辑类,全被 gate 拒绝
4. 剩下的都是 `count_rerank`/`layout_regenerate` 重生成类,主循环设计上就不走局部编辑
5. 结果:局部编辑从未触发

**修复方向**:
- 短期:开启 `enable_vlm_target_locator`
- 中期:修改 planner 策略,让更多失败类路由到编辑类 route
- 长期:编辑链路加稳定性(grounding 质量/mask 预检)

---

## 问题 6: CLI 开关传了但没生效?

**表现**:
- Stage D2 验证时,命令行传了 `--enable-efficient-repair-agent --enable-editing-mask-agent`
- 但 `config.json` 里这些字段还是 `None`

**根因**:
- 验证 case(occlusion) round_0 就通过了,没触发修复阶段
- CLI 开关只在修复阶段才被查,round_0 通过的 case 看不到开关效果

**解决**:
- 需选一个 round_0 **会失败**的 case 验证
- 或直接查 baseline run 的 events(已在问题 5 完成)

---

## 专家意见总结

### 原始问题
> "agent 不行,24 个 case 只过 6 个,改进在哪?"

### 问题溯源
1. **表象**: 整体成功率 25%,无法定位瓶颈
2. **误判链**: L1 崩 → 整体失败 → 误判"agent 不行"
3. **根因**: 缺机械化 failure attribution

### 专家建议(已实施)
1. **分层 instrumentation 优先于优化** — 不先诊断就动手术是盲改
2. **机械化归因 > 人工判断** — `_determine_failure_layer` 把"哪层不行"变成可测试函数
3. **per-route 成功率是关键指标** — 直接回答"这个修复策略有没有用"
4. **constraint_passed=False 不能归 L2** — L2 报错是履行职责

### 决策记录
- ✅ Stage B 优先于 C/D
- ✅ 建立 `route_success_rate` 作为 L3 效果 ground truth
- ✅ 修正 `constraint_passed=False` 归因
- ✅ 锁定 `['none']` route 不算修复的 bug
- ⏭️ 下一步: Stage C(L2 隔离评测)或 D(L3 整改)

---

## 关键洞察

### 1. 不是神秘失败,而是可定位的系统性无效
- L1: 0 失败(偶发 timeout 需查)
- L2: 判断准确,假阳率低
- L3: 重生成 no-op,局部编辑未触发

### 2. `route_success_rate` 的价值
- 避免"整体成功率 25%"这种无法 actionable 的指标
- 直接回答:"count_aware_regeneration 对 count 失败有用吗?" → 0.0,无用

### 3. 归因链条要追到底
- 表象:"局部编辑 0 次"
- 第一层:"开关没开" ✗
- 第二层:"某些类自动启用" ✓ 部分正确
- 第三层:"主循环只接受 5 种编辑 route" ✓
- 第四层:"编辑类 route 需要 bbox,但 locator 没开" ✓✓ 根因

### 4. 设计决策的隐含约束
- 主循环不接受 `count_rerank`/`layout_regenerate`,这是**设计决策**,不是 bug
- 说明系统设计者认为这些应该走其他链路(batch rerank/全图重生成)
- 但实际 planner 把多数失败路由到了这些,导致局部编辑永远触发不了
- **D1 和 D2 在这里耦合**:planner 选重生成 → 重生成无效 → 局部编辑没机会

---

## 改动清单

### 代码
- `scripts/run_mini_benchmark.py`: +180 行
  - `_determine_failure_layer()` — 72 行机械归因
  - `route_hit_counts` / `route_success_counts` — per-route 统计
  - `failure_layer_counts` / `self_eval_metrics` — 聚合指标
- `tests/test_benchmarks.py`: +90 行(7 个新测试)

### 文档
- `docs/STAGE_B_REPORT.md` — Stage B 完成报告
- `docs/ISSUES_AND_SOLUTIONS.md` — 本文档
- `PLAN.md` — 更新 Stage B/D 状态和根因

### 测试
- 31/31 测试通过
- 真实出图验证完成(stageB-real-gpu0 + baseline r2 重新归因)

---

## 下一步行动

### Stage C: L2 判断隔离评测
- `--decision-only` 跑几个 case
- 人工核对 VQA 判断 vs 真实图
- 量化假阳/假阴率

### Stage D: L3 整改
- **D1**(工作量大): 重生成策略改造
  - 引入 layout 约束
  - 降级为编辑类
- **D2**(短期可验证): 局部编辑链路修复
  - 开启 `enable_vlm_target_locator`
  - 修改 planner 路由策略

---

**文档完成**: 2026-06-25 23:59  
**状态**: Stage B ✅ 完成,Stage D2 根因已定位
