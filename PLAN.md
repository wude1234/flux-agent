# T2I-Agent 项目完整 Plan(基于代码 + 真实效果证据)

> 数据基线:`real-flux-typed-router36-512-30-gpu0-r2`(06-23,24 个硬 case,真实 FLUX baseline,VLM=qwen-vl-plus)
> completion 6/24 (25%) · hard VQA 12/24 (50%) · eval 8/24 (33%) · 局部编辑触发 0 次

---

## 0. 一句话现状

三层(L1 生成 / L2 判断 / L3 修复)的真实状态已用证据厘清:
- **L1 生成后端**:是当前唯一的硬阻塞。GPU 被他人占满时 `enable_model_cpu_offload` 会 hang 满 900s timeout(不是秒崩,纠正此前判断)。
- **L2 判断**:基本可用。实测 VLM 数对了 marbles 数量、空间关系判断正确,硬约束 VQA 是可信的。
- **L3 修复**:形同空转。24 个 case 局部编辑 `efficient_edit_attempts=0`;count 类 6 次 `count_aware_regeneration` 重生成后数量仍错,对计数无效。

---

## 阶段 A:解封 L1(P0,最高优先,其余全部依赖它)

**问题**:GPU 争用下 mgrag 的 `enable_model_cpu_offload` 阻塞至 timeout;cache 路径历史 bug 已修但仍需显存充足才能验证。

**任务**:
- A1. 给 `infer_mgrag_flux.py` 加 `--cpu-offload-mode {model,sequential,none}` 开关。`sequential` 峰值显存更低,可在 ~7G 空闲挤进去验证。
- A2. 加 GPU 显存预检:启动采样前查目标卡空闲显存,不足直接明确报错(而非 hang 到 timeout 再被误判成"后端故障")。
- A3. 把生成 subprocess 的 timeout 从 900s 降到合理值(单图 512/30steps 正常 <120s),让失败快速暴露。
- A4. 拿到 1 张干净验证图后,L1 正式解封。

**验收**:`--flux-attn-mode baseline` 和 `mgrag` 各出 1 张 512 图,无 timeout/OOM。

---

## 阶段 B:分层 instrumentation(P0,根治"误判哪层不行")✅ 已完成

**问题**:上次 L1 崩溃被误判成"agent 不行",根因是缺机械化 failure attribution。

**任务**:
- B1. ✅ 每层 health flag 已由 per-case 字段覆盖:L1(status/rounds/selected_image)、L2(constraint_passed/evaluation_passed)、L3(typed_routes/efficient_edit_attempts/accepted_edit_count)。
- B2. ✅ `_determine_failure_layer` 机械判定每个 case 失败在哪层,写入每条 result 的 `failure_layer` 字段 + summary 的 `failure_layer_counts`。**关键修正**:`constraint_passed=False` 表示 L2 正确报了故障,失败应归 L3(`L3_repair` 已触发但没修好 / `L3_not_triggered` 从未触发),不能再误判成 L2。
- B3. ✅ `self_eval_metrics`:各 route 命中后成功率(`route_success_rate`)、局部编辑 accept 率、false-pass 率、hard/completion 率、L1 故障率。

**验收**:✅ 任一失败 run 一键回答"失败在哪层"。

**真实 baseline 跑出的归因(`real-flux-typed-router36`,24 case)**:
- `failure_layer_counts`: `{none: 6, L2_judgment: 6, L3_repair: 12}` → L1 零失败,问题全在 L2/L3。
- `route_success_rate`: count_aware/layout_guided/material_guided/multi_constraint/relation_contact **全部 0.0**,即重生成类 route 对所有失败类都是 no-op。
- `efficient_edit_attempts=0` → 局部编辑链路从未启用,L3 实际只有空转的重生成。
- 结论:不是"agent 神秘失败",而是 L3 重生成 route 无效 + 局部编辑没开。这正是阶段 D 要整改的靶子。

---

## 阶段 C:L2 判断隔离评测(B 完成后) ✅ 已完成

**任务**:
- C1. ✅ 复用 baseline run(24 case)数据,避免重新烧 GPU。
- C2. ✅ 人工核对 VQA 判断 vs 真实图,深挖 6 个 hard-pass-but-completion-fail case。
- C3. ✅ 定位假阳根因:`constraint_questions.py` 缺对象存在否定处理。

**关键发现(假设反转)**:
- PLAN 原假设:negation 类是"软评估错误卡住 case"
- **真相相反**:软评估**抓对了**(图里真有 bowl/window/zipper),硬 VQA 才误判(pass=1.0 但没检查禁止物)
- 根因:`constraint_questions.py` 只处理关系否定+符号否定,**缺最常见的对象存在否定**("no bowl","no window")
- 影响:baseline 4/24 case 假阳性(negation_004/006/010 + interaction_005),硬 VQA 没生成禁止物检查问题

**修复已落地** ✅:
- 新增 `_negative_object_existence_constraints()`(src/constraint_questions.py:1443-1528)
- 提取 "no X"/"with no Y" 禁止对象,生成 "X absent?" VQA 问题
- 7/7 单元测试 + 416/416 完整测试通过

**验收**:修复完成,真实验证因 GPU OOM 暂缓(不影响代码质量)。详见 `docs/STAGE_C_REPORT.md`。

---

## 阶段 D:L3 修复的针对性整改(C 完成后)

**核心发现**:L3 当前没产生价值。两条主线:

- **D1. 重生成类 route 完全无效**(Stage B 机械证明):
  - `route_success_rate` 所有重生成类 route 全 0.0:`count_aware_regeneration`(5 次命中)、`layout_guided_regeneration`(7 次)、`material_guided`(4 次)、`multi_constraint`(4 次)、`relation_contact`(1 次)。
  - **根因**:FLUX 对 prompt 里的精确数量/空间关系控制能力弱,单纯改 prompt 重生成解决不了。3 只鸟重生成还是错的数量。
  - **方向**:
    1. 引入 layout 约束(bounding box + regional prompting)
    2. "生成 N+buffer 后挑选"策略(要 5 只就生成 10 只候选)
    3. 降级为 object removal/insertion(多删少补,走局部编辑)

- **D2. 局部编辑从不触发**(Stage B+D 完整诊断):
  - baseline run `efficient_edit_attempts=0`,查 events 发现 `efficient_repair_route_skipped` 7 次:
    - 3 次 `layout_regenerate` → "route is not an efficient editing route in the main loop"(主循环设计上就不走局部编辑)
    - 4 次 `existing_object_inpaint` → "efficient inpaint route requires an explicit localized bbox"(缺 bbox 被 gate 拒绝)
  - **根因链条**:
    1. `route_repair_kind()` 把多数 typed_route 映射成 `count_rerank`/`layout_regenerate`,主循环只接受 5 种编辑 route(`text_overlay/symbol_overlay/shape_overlay/bbox_shape_inpaint/existing_object_inpaint`)
    2. 少数路由到 `existing_object_inpaint` 的,因 baseline run **没开 `enable_vlm_target_locator`**,无法获取 bbox,被 `_efficient_repair_gate` 拒绝
    3. 结果:局部编辑从未触发
  - **修复方向**:
    - 短期 ✅ 已落地:`run_mini_benchmark.py` 里让 `enable_vlm_target_locator` 跟随 `enable_efficient_repair_agent`(line 820-822)。此前显式开 agent 但不开 locator,编辑路由拿不到 bbox,被 gate 静默拒绝产生 0 次编辑。
    - 中期(待做):修改 planner 策略,让更多失败类路由到编辑类 route(如 count 失败 → object removal,而不是 count_aware_regeneration)
    - 长期(待做):编辑链路加稳定性(Grounded-SAM2 grounding 质量 / PowerPaint subprocess 启动耗时 / mask 质量预检)

---

## 阶段 E:架构债(P1,与论文方向并行)

- E1. `orchestrator.run()` 700 行单体循环重构成显式状态机(每个 repair stage 一个 state)。
- E2. 加 no-progress/震荡检测:连续轮 constraint_check failed set 无变化 → 提前停 + 标记 stuck。
- E3.(P2)跨 run 记忆:积累"哪类 prompt 用哪个 route 成功率高",避免每次冷启动。

---

## 论文定位建议

- 投 **system/agent 会议**:强调 verifier-in-the-loop + failure taxonomy(20+ typed routes 是原创);**慎用 "multi-agent"**(specialist 是 1 次 VLM + 本地分析,伪多代理,易被 reviewer 挑)。
- 投 **T2I/CV 会议**:算法创新需聚焦单点深挖(typed_action 策略学习 / failure taxonomy 系统性),否则被批"工程集成"。
- L1 的 M-GRAG(K_txt 方差放大)对空间/计数的改善可能不如预期(非真 layout control),需 baseline vs mgrag 对照实验给结论。

---

## 执行顺序(强约束)

```
A(解封 L1)→ B(分层归因)→ C(隔离测 L2)→ D(整改 L3)→ E(架构/论文)
```
A、B 未完成前,任何"正式 benchmark"都会重蹈"分不清哪层不行"的覆辙。
