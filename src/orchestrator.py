"""Unified M4 orchestration loop for the multimodal T2I agent."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from .binding_strategy import (
    build_binding_retry_prompt,
    build_negative_prompt,
    has_binding_failure,
    merge_negative_prompts,
)
from .binding_variants import build_binding_variants, should_use_binding_variants
from .candidate_arbitration import arbitrate_image_candidates
from .candidate_scorer import CandidateScorer
from .clients import LLMClient, VLMClient
from .constraint_questions import VQAConstraintEvaluator
from .error_analyzer import ErrorAnalyzer
from .evaluators import Evaluator, normalize_error_type
from .editing_agent import EfficientRepairAgent, EfficientRepairRequest, route_repair_kind
from .factuality_qa import FactualityQAEvaluator
from .image_generator import ImageGenerator
from .layout_planner import (
    LayoutPlanner,
    layout_to_enriched_prompt,
    layout_to_generation_hint,
    layout_to_prompt_package,
)
from .local_editor import (
    ColorRecolorEditor,
    InpaintRegion,
    detect_color_region_from_layout,
    measure_color_coverage,
    plan_inpaint_region_from_layout,
)
from .logging_utils import (
    DEFAULT_RUNS_DIR,
    create_run_dir,
    write_final_report,
    write_json,
    write_state_snapshot,
)
from .mask_refiner import (
    MaskRefiner,
    constrain_refined_mask_to_prior,
    refine_bbox_mask,
)
from .memory import MemoryStore
from .object_state import augment_record_with_object_geometry
from .proactive_clarifier import ProactiveClarifier
from .prompt_constraints import (
    DEFAULT_CLIP_TOKEN_BUDGET,
    PromptConstraints,
    constraint_violations,
    extract_constraints,
    lock_prompt_to_user_constraints,
)
from .prompt_optimizer import PromptOptimizer
from .prompt_reviser import PromptReviser
from .repair_base_selector import select_repair_base
from .repair_planner import (
    prompt_needs_lexical_preflight,
    RepairPlanner,
    RuleBasedRepairPlanner,
)
from .repairable_candidate_selector import select_repairable_candidate
from .relation_repair import RelationActionRepairer
from .specialist_agents import (
    arbitrate_specialist_reports,
    build_specialist_observation_request,
    build_specialist_reports,
    parse_specialist_observation_response,
)
from .reward_reranker import RewardReranker
from .state import AgentConfig, AgentState
from .target_locator import layout_with_target_bbox, locate_target_region
from .typed_action_backend import (
    best_typed_action_candidate_index,
    build_typed_action_prompt_variants,
    typed_action_backend_route,
    typed_action_rejected_reasons,
)
from .visual_reflector import build_round_record, VisualReflector


@dataclass(frozen=True)
class OrchestratorResult:
    """Serializable result returned by ``OrchestratorAgent.run``."""

    run_id: str
    run_dir: str
    status: str
    mode: str
    state: dict[str, Any]
    config: dict[str, Any]
    round_records: list[dict[str, Any]]
    events: list[dict[str, Any]]
    final_report_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "status": self.status,
            "mode": self.mode,
            "state": deepcopy(self.state),
            "config": deepcopy(self.config),
            "round_records": deepcopy(self.round_records),
            "events": deepcopy(self.events),
            "final_report_path": self.final_report_path,
        }


class OrchestratorAgent:
    """Control prompt generation, image generation, critique, and optimization."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        vlm: VLMClient,
        image_generator: ImageGenerator,
        config: AgentConfig | Mapping[str, Any] | None = None,
        memory: MemoryStore | None = None,
        runs_dir: str | Path = DEFAULT_RUNS_DIR,
        mode: str = "mock",
        score_threshold: float = 0.85,
        prompt_candidates_per_round: int = 2,
        enable_clarifier: bool = True,
        auto_merge_clarification: str | None = None,
        clip_token_budget: int = DEFAULT_CLIP_TOKEN_BUDGET,
        enable_constraint_check: bool = False,
        auto_negative_prompt: bool = True,
        negative_prompt: str | None = None,
        enable_layout_planner: bool = False,
        layout_planner: LayoutPlanner | None = None,
        layout_canvas_size: tuple[int, int] = (1024, 1024),
        enable_binding_variants: bool = False,
        evaluator: Evaluator | None = None,
        enable_evaluator: bool = False,
        factuality_evaluator: FactualityQAEvaluator | None = None,
        enable_factuality_qa: bool = False,
        reward_reranker: RewardReranker | None = None,
        enable_reward_reranker: bool = False,
        reward_rerank_override: bool = True,
        enable_local_repair: bool = False,
        enable_vlm_target_locator: bool = False,
        relation_repairer: RelationActionRepairer | None = None,
        enable_relation_repair: bool = False,
        enable_object_insertion_repair: bool = False,
        efficient_repair_agent: EfficientRepairAgent | None = None,
        enable_efficient_repair_agent: bool = False,
        repair_planner: RepairPlanner | None = None,
        enable_repair_planner: bool = True,
        mask_refiner: MaskRefiner | None = None,
        enable_mask_refiner: bool = False,
        enable_specialist_reports: bool = True,
        enable_specialist_vlm_observation: bool = False,
        enable_typed_action_backend: bool = False,
        typed_action_candidates: int = 3,
        typed_action_max_candidates: int = 4,
    ) -> None:
        self.config = (
            config
            if isinstance(config, AgentConfig)
            else AgentConfig.from_dict(config or {})
        )
        self.llm = llm
        self.vlm = vlm
        self.image_generator = image_generator
        self.memory = memory or MemoryStore()
        self.runs_dir = Path(runs_dir)
        self.mode = _clean_mode(mode)
        self.score_threshold = _coerce_score_threshold(score_threshold)
        self.prompt_candidates_per_round = max(1, int(prompt_candidates_per_round))
        self.enable_clarifier = bool(enable_clarifier)
        self.auto_merge_clarification = auto_merge_clarification
        self.clip_token_budget = int(clip_token_budget)
        self.enable_constraint_check = bool(enable_constraint_check)
        self.auto_negative_prompt = bool(auto_negative_prompt)
        self.negative_prompt = _clean_optional_text(negative_prompt)
        self.enable_layout_planner = bool(enable_layout_planner)
        self.layout_planner = layout_planner
        self.layout_canvas_size = _coerce_canvas_size(layout_canvas_size)
        self.enable_binding_variants = bool(enable_binding_variants)
        self.evaluator = evaluator
        self.enable_evaluator = bool(enable_evaluator and evaluator is not None)
        self.factuality_evaluator = factuality_evaluator
        self.enable_factuality_qa = bool(
            enable_factuality_qa and factuality_evaluator is not None
        )
        self.reward_reranker = reward_reranker
        self.enable_reward_reranker = bool(
            enable_reward_reranker and reward_reranker is not None
        )
        self.reward_rerank_override = bool(reward_rerank_override)
        self.enable_local_repair = bool(enable_local_repair)
        self.enable_vlm_target_locator = bool(enable_vlm_target_locator)
        self.relation_repairer = relation_repairer
        self.enable_relation_repair = bool(
            enable_relation_repair and relation_repairer is not None
        )
        self.enable_object_insertion_repair = bool(
            (enable_object_insertion_repair or enable_relation_repair)
            and relation_repairer is not None
        )
        self.efficient_repair_agent = efficient_repair_agent
        self.enable_efficient_repair_agent = bool(
            enable_efficient_repair_agent and efficient_repair_agent is not None
        )
        planner_vlm = vlm if self.mode != "mock" else None
        self.repair_planner = repair_planner or RuleBasedRepairPlanner(planner_vlm)
        self.enable_repair_planner = bool(enable_repair_planner)
        self.mask_refiner = mask_refiner
        self.enable_mask_refiner = bool(enable_mask_refiner and mask_refiner is not None)
        self.enable_specialist_reports = bool(enable_specialist_reports)
        self.enable_specialist_vlm_observation = bool(enable_specialist_vlm_observation)
        self.enable_typed_action_backend = bool(enable_typed_action_backend)
        self.typed_action_candidates = max(1, int(typed_action_candidates))
        self.typed_action_max_candidates = max(
            self.typed_action_candidates,
            int(typed_action_max_candidates),
        )

        self.prompt_agent = PromptReviser(llm)
        self.visual_critic = VisualReflector(vlm)
        self.constraint_question_evaluator = VQAConstraintEvaluator(vlm)
        self.error_analyzer = ErrorAnalyzer()
        self.prompt_optimizer = PromptOptimizer(llm)
        self.candidate_scorer = CandidateScorer(vlm)
        self.clarifier = ProactiveClarifier(
            llm,
            creativity_level=self.config.creativity_level,
        )

    def run(self, user_prompt: str, *, run_id: str | None = None) -> OrchestratorResult:
        """Run the full mockable M4 loop and write a reproducible run folder."""

        state = AgentState.from_config(user_prompt, self.config)
        constraints = extract_constraints(state.user_prompt)
        run_dir = create_run_dir(self.runs_dir, run_id=run_id)
        image_dir = run_dir / "images"
        allocated_run_id = run_dir.name
        events: list[dict[str, Any]] = [
            {
                "type": "state_initialized",
                "round": state.round_index,
                "user_prompt": state.user_prompt,
            },
            {
                "type": "input_interpreted",
                "agent": "InputInterpreterAgent",
                "round": state.round_index,
                "intent_spec": (
                    constraints.intent_spec.to_dict()
                    if constraints.intent_spec is not None
                    else None
                ),
                "prompt_constraints": constraints.to_dict(),
            }
        ]
        round_records: list[dict[str, Any]] = []
        layout_context: dict[str, Any] | None = None
        status = "running"

        config_payload = _config_payload(self.config, self.mode, constraints)
        config_payload["m6"] = self._m6_config_payload()
        write_json(run_dir / "config.json", config_payload)

        clarification = self._maybe_clarify(state, events)
        if clarification and clarification.get("status") == "ask_user":
            status = "awaiting_clarification"
            final_report_path = write_final_report(
                run_dir,
                run_id=allocated_run_id,
                status=status,
                mode=self.mode,
                user_prompt=state.user_prompt,
                final_prompt=state.active_prompt,
                final_score=None,
                selected_image=None,
                round_records=round_records,
            )
            write_json(
                run_dir / "run.json",
                _run_payload(
                    allocated_run_id,
                    status,
                    self.mode,
                    self.config,
                    constraints,
                    state,
                    round_records,
                    events,
                    final_report_path,
                    layout_context=layout_context,
                ),
            )
            return OrchestratorResult(
                run_id=allocated_run_id,
                run_dir=str(run_dir),
                status=status,
                mode=self.mode,
                state=state.to_dict(),
                config=self.config.to_dict(),
                round_records=round_records,
                events=events,
                final_report_path=str(final_report_path),
            )

        layout_context = self._maybe_plan_layout(state, run_dir, events)

        for round_index in range(self.config.max_rounds):
            state.round_index = round_index
            current_prompt = self._select_prompt_for_round(
                state,
                round_records,
                events,
                constraints,
                layout_context,
            )
            image_paths, image_prompts = self._generate_images(
                current_prompt,
                image_dir,
                events,
                round_index,
                constraints,
            )
            state.image_paths = image_paths
            reward_ranking = self._maybe_reward_rerank(
                current_prompt,
                image_paths,
                run_dir,
                events,
                round_index,
            )

            try:
                selection = self.visual_critic.select_best(
                    state.user_prompt,
                    image_prompts,
                    image_paths,
                )
            except Exception as exc:
                selection = _failed_selection(image_prompts, image_paths, exc)
                events.append(
                    {
                        "type": "selection_failed",
                        "agent": "QualityEvaluatorAgent",
                        "round": round_index,
                        "error": str(exc),
                        "fallback_selected_index": selection["selected_index"],
                    }
                )
            if reward_ranking and self.reward_rerank_override:
                reward_index = int(reward_ranking["selected_index"])
                selection = {
                    **selection,
                    "selected_index": reward_index,
                    "selected_image": image_paths[reward_index],
                    "selected_prompt": image_prompts[reward_index],
                    "reward_override": _strip_large_prompt(reward_ranking),
                }
                events.append(
                    {
                        "type": "reward_selection_override",
                        "agent": "QualityEvaluatorAgent",
                        "round": round_index,
                        "selected_index": reward_index,
                        "selected_image": image_paths[reward_index],
                        "score": reward_ranking["scores"][0]["score"],
                    }
                )
            arbitration = self._maybe_arbitrate_image_selection(
                state,
                image_prompts,
                image_paths,
                selection,
                reward_ranking,
                constraints,
                run_dir,
                events,
                round_index,
            )
            if arbitration:
                selected_index = int(arbitration["selected_index"])
                selection = {
                    **selection,
                    "selected_index": selected_index,
                    "selected_image": image_paths[selected_index],
                    "selected_prompt": image_prompts[selected_index],
                    "constraint_arbitration": _strip_large_prompt(arbitration),
                }
                events.append(
                    {
                        "type": "constraint_aware_selection",
                        "agent": "QualityEvaluatorAgent",
                        "round": round_index,
                        "selected_index": selected_index,
                        "selected_image": image_paths[selected_index],
                        "visual_selected_index": arbitration["visual_selected_index"],
                        "reward_selected_index": arbitration["reward_selected_index"],
                        "overrode_reward_selection": arbitration.get(
                            "overrode_reward_selection",
                            False,
                        ),
                        "used_candidate_constraints": arbitration.get(
                            "used_candidate_constraints",
                            False,
                        ),
                    }
                )
            selected_image = selection["selected_image"]
            selected_prompt = str(selection.get("selected_prompt") or current_prompt)
            try:
                critique = self.visual_critic.reflect(
                    state.user_prompt,
                    selected_prompt,
                    selected_image,
                    history=round_records,
                )
            except Exception as exc:
                critique = _failed_critique(selected_prompt, selected_image, exc)
                events.append(
                    {
                        "type": "critique_failed",
                        "agent": "QualityEvaluatorAgent",
                        "round": round_index,
                        "error": str(exc),
                    }
                )
            if arbitration:
                critique["constraint_arbitration"] = _strip_large_prompt(arbitration)
            if self.enable_constraint_check:
                try:
                    constraint_check = _selected_candidate_constraint_check(
                        arbitration,
                        int(selection.get("selected_index", 0)),
                    )
                    if constraint_check is None:
                        record = self._check_constraints_with_questions(
                            state.user_prompt,
                            selected_prompt,
                            selected_image,
                            constraints,
                            history=round_records,
                        )
                        constraint_check = record["constraint_check"]
                    critique = _merge_constraint_check(critique, constraint_check)
                    event_type = (
                        "constraint_check_failed"
                        if constraint_check.get("failed")
                        else "constraint_check"
                    )
                    event = {
                        "type": event_type,
                        "agent": "QualityEvaluatorAgent",
                        "round": round_index,
                        "passed": constraint_check["passed"],
                        "score": constraint_check["score"],
                        "error_count": len(constraint_check.get("errors", [])),
                    }
                    if constraint_check.get("error"):
                        event["error"] = str(constraint_check.get("error"))
                    events.append(event)
                except Exception as exc:
                    constraint_check = _failed_constraint_check(
                        selected_prompt,
                        selected_image,
                        exc,
                    )
                    critique["constraint_check"] = _strip_large_prompt(constraint_check)
                    events.append(
                        {
                            "type": "constraint_check_failed",
                            "agent": "QualityEvaluatorAgent",
                            "round": round_index,
                            "error": str(exc),
                        }
                    )
            evaluation = self._maybe_evaluate(
                state,
                selected_prompt,
                selected_image,
                critique,
                run_dir,
                events,
                round_index,
            )
            if evaluation:
                critique = _merge_evaluation(critique, evaluation)
                if _question_level_constraints_passed(critique.get("constraint_check")):
                    _downgrade_top_level_errors_contradicting_passed_questions(
                        critique,
                        critique.get("constraint_check", {}),
                    )
            hard_pass_guard = _hard_pass_guard(critique, constraints=constraints)
            if hard_pass_guard:
                events.append(
                    {
                        "type": "hard_pass_guard",
                        "round": round_index,
                        **hard_pass_guard,
                    }
                )
                critique = _apply_hard_pass_guard(critique, hard_pass_guard)
            else:
                hard_pass_guard = None
            repair_plan: dict[str, Any] | Mapping[str, Any] | None = None
            repair_base_selection = None
            if not hard_pass_guard:
                repair_plan = self._maybe_plan_repair(
                    state,
                    selected_prompt,
                    selected_image,
                    critique,
                    constraints,
                    run_dir,
                    events,
                    round_index,
                )
                repair_plan = _route_all_count_failures_to_regenerate(
                    repair_plan,
                    arbitration,
                    can_regenerate=self._has_retry_budget(round_index),
                )
                if (
                    isinstance(repair_plan, Mapping)
                    and repair_plan.get("source") == "m6212_count_failure_route"
                ):
                    write_json(
                        run_dir / f"repair_plan_round_{round_index}.json",
                        _strip_large_prompt(repair_plan),
                    )
                    events.append(
                        {
                            "type": "count_failure_regeneration_route",
                            "round": round_index,
                            "primary_action": repair_plan.get("primary_action"),
                            "can_regenerate": self._has_retry_budget(round_index),
                            "reason": repair_plan.get("reason"),
                        }
                    )
                if repair_plan:
                    critique["repair_plan"] = _strip_large_prompt(repair_plan)
                repair_base_selection = self._maybe_select_repair_base(
                    image_prompts,
                    image_paths,
                    selected_index=int(selection.get("selected_index", 0)),
                    constraints=constraints,
                    arbitration=_arbitration_with_current_feedback(
                        arbitration,
                        int(selection.get("selected_index", 0)),
                        critique,
                    ),
                    repair_plan=repair_plan,
                    reward_ranking=reward_ranking,
                    current_feedback=critique,
                    run_dir=run_dir,
                    events=events,
                    round_index=round_index,
                )
            if repair_base_selection:
                selected_index = int(repair_base_selection["selected_index"])
                selected_image = str(repair_base_selection["selected_image"])
                selected_prompt = str(repair_base_selection["selected_prompt"])
                selection = {
                    **selection,
                    "selected_index": selected_index,
                    "selected_image": selected_image,
                    "selected_prompt": selected_prompt,
                    "repair_base_selection": _strip_large_prompt(repair_base_selection),
                }
                critique["repair_base_selection"] = _strip_large_prompt(
                    repair_base_selection
                )
                selected_check = _selected_candidate_constraint_check(
                    arbitration,
                    selected_index,
                )
                if selected_check is not None:
                    critique = _merge_constraint_check(critique, selected_check)
            repairable_selection = self._maybe_select_repairable_candidate(
                image_prompts,
                image_paths,
                selected_index=int(selection.get("selected_index", 0)),
                selected_prompt=selected_prompt,
                selected_image=selected_image,
                critique=critique,
                constraints=constraints,
                arbitration=arbitration,
                repair_plan=repair_plan,
                run_dir=run_dir,
                events=events,
                round_index=round_index,
            )
            if repairable_selection:
                critique["repairability_selection"] = _strip_large_prompt(
                    repairable_selection
                )
                selection = {
                    **selection,
                    "repairability_selection": _strip_large_prompt(repairable_selection),
                }
                if not repairable_selection.get("blocked"):
                    selected_index = int(repairable_selection["selected_index"])
                    selected_image = str(repairable_selection["selected_image"])
                    selected_prompt = str(repairable_selection["selected_prompt"])
                    selection = {
                        **selection,
                        "selected_index": selected_index,
                        "selected_image": selected_image,
                        "selected_prompt": selected_prompt,
                    }
                    plan_override = repairable_selection.get("repair_plan_override")
                    if isinstance(plan_override, Mapping):
                        repair_plan = deepcopy(dict(plan_override))
                        repair_plan["round"] = round_index
                        repair_plan["enabled_tools"] = {
                            "recolor": bool(self.enable_local_repair),
                            "relation_repair": bool(self.enable_relation_repair),
                            "object_insertion": bool(
                                self.enable_object_insertion_repair
                                and self.relation_repairer is not None
                            ),
                            "regenerate": self._has_retry_budget(round_index),
                        }
                        critique["repair_plan"] = _strip_large_prompt(repair_plan)
                    selected_check = repairable_selection.get("selected_constraint_check")
                    if isinstance(selected_check, Mapping):
                        critique = _merge_constraint_check(
                            {**critique, "constraint_check": {}},
                            selected_check,
                        )
            efficient_repair = self._maybe_efficient_repair(
                state,
                selected_prompt,
                selected_image,
                critique,
                constraints,
                run_dir,
                events,
                round_index,
                repair_plan=repair_plan,
            )
            if efficient_repair and efficient_repair.get("accepted"):
                selected_image = str(efficient_repair["edited_image"])
                if selected_image not in image_paths:
                    image_paths = [*image_paths, selected_image]
                    state.image_paths = image_paths
                post_efficient_check = efficient_repair.get("post_repair_constraint_check")
                if isinstance(post_efficient_check, Mapping):
                    critique = _merge_object_repair_check(critique, post_efficient_check)
                critique = _mark_accepted_local_edit(critique, efficient_repair)
            efficient_repair_accepted = bool(efficient_repair and efficient_repair.get("accepted"))
            object_repair = None
            if (
                not efficient_repair_accepted
                and _efficient_repair_attempted_editing_backend(efficient_repair)
            ):
                events.append(
                    {
                        "type": "object_insertion_repair_skipped",
                        "round": round_index,
                        "reason": (
                            "efficient repair already attempted an editing backend "
                            "for this repair plan"
                        ),
                        "efficient_route": (
                            efficient_repair.get("route")
                            if isinstance(efficient_repair, Mapping)
                            else None
                        ),
                        "efficient_ok": (
                            efficient_repair.get("ok")
                            if isinstance(efficient_repair, Mapping)
                            else None
                        ),
                    }
                )
            elif not efficient_repair_accepted:
                object_repair = self._maybe_object_insertion_repair(
                    state,
                    selected_prompt,
                    selected_image,
                    critique,
                    constraints,
                    layout_context,
                    run_dir,
                    events,
                    round_index,
                    repair_plan=repair_plan,
                )
            if object_repair and object_repair.get("accepted"):
                selected_image = str(object_repair["edited_image"])
                if selected_image not in image_paths:
                    image_paths = [*image_paths, selected_image]
                    state.image_paths = image_paths
                post_object_check = object_repair.get("post_repair_constraint_check")
                if isinstance(post_object_check, Mapping):
                    critique = _merge_constraint_check(
                        {**critique, "constraint_check": {}},
                        post_object_check,
                    )
                    if post_object_check.get("passed") is True:
                        critique = _merge_object_repair_check(critique, post_object_check)
            efficient_repair_accepted = bool(efficient_repair and efficient_repair.get("accepted"))
            repair = None
            if not efficient_repair_accepted:
                repair = self._maybe_local_repair(
                    state,
                    selected_prompt,
                    selected_image,
                    critique,
                    evaluation,
                    constraints,
                    layout_context,
                    run_dir,
                    events,
                    round_index,
                    repair_plan=repair_plan,
                )
            if repair and repair.get("accepted"):
                selected_image = str(repair["edited_image"])
                if selected_image not in image_paths:
                    image_paths = [*image_paths, selected_image]
                    state.image_paths = image_paths
                repair_evaluation = repair.get("evaluation")
                if isinstance(repair_evaluation, Mapping):
                    critique = _merge_repair_evaluation(critique, repair_evaluation)
                post_repair_check = repair.get("post_repair_constraint_check")
                if isinstance(post_repair_check, Mapping):
                    critique = _merge_constraint_check(
                        {**critique, "constraint_check": {}},
                        post_repair_check,
                    )
            efficient_repair_accepted = bool(efficient_repair and efficient_repair.get("accepted"))
            relation_repair = None
            if not efficient_repair_accepted:
                relation_repair = self._maybe_relation_repair(
                    state,
                    selected_prompt,
                    selected_image,
                    critique,
                    constraints,
                    layout_context,
                    run_dir,
                    events,
                    round_index,
                    repair_plan=repair_plan,
                )
            if relation_repair and relation_repair.get("accepted"):
                original_relation_image = selected_image
                candidate_relation_image = str(relation_repair["edited_image"])
                verification = relation_repair.get("verification") or {}
                if not verification and isinstance(relation_repair.get("candidates"), list):
                    selected_index = int(relation_repair.get("selected_index", 0))
                    candidates = relation_repair.get("candidates", [])
                    if 0 <= selected_index < len(candidates):
                        verification = candidates[selected_index].get("verification", {})
                post_relation_check = self._check_relation_repair_constraints(
                    state,
                    selected_prompt,
                    candidate_relation_image,
                    constraints,
                    relation_repair,
                    round_index,
                )
                relation_repair["post_repair_constraint_check"] = post_relation_check
                post_failures = _post_repair_constraint_failures(post_relation_check)
                relation_repair["acceptance"] = {
                    **dict(relation_repair.get("acceptance") or {}),
                    "accepted": not post_failures,
                    "post_repair_constraint_failures": post_failures,
                    "non_regression_rule": (
                        "reject relation repair if the edited image violates any "
                        "original user hard constraint"
                    ),
                }
                if post_failures:
                    relation_repair["accepted"] = False
                    relation_repair["rejected_image"] = candidate_relation_image
                    relation_repair["edited_image"] = original_relation_image
                    events.append(
                        {
                            "type": "post_relation_repair_constraint_check",
                            "round": round_index,
                            "passed": post_relation_check["passed"],
                            "score": post_relation_check["score"],
                            "error_count": len(post_relation_check.get("errors", [])),
                            "accepted": False,
                            "rejected": True,
                        }
                    )
                else:
                    relation_repair["accepted"] = True
                    events.append(
                        {
                            "type": "post_relation_repair_constraint_check",
                            "round": round_index,
                            "passed": post_relation_check["passed"],
                            "score": post_relation_check["score"],
                            "error_count": len(post_relation_check.get("errors", [])),
                            "accepted": True,
                        }
                    )
                write_json(
                    run_dir / f"relation_repair_round_{round_index}.json",
                    _strip_large_prompt(relation_repair),
                )
            if relation_repair and relation_repair.get("accepted"):
                selected_image = str(relation_repair["edited_image"])
                if selected_image not in image_paths:
                    image_paths = [*image_paths, selected_image]
                    state.image_paths = image_paths
                verification = relation_repair.get("verification") or {}
                if not verification and isinstance(relation_repair.get("candidates"), list):
                    selected_index = int(relation_repair.get("selected_index", 0))
                    candidates = relation_repair.get("candidates", [])
                    if 0 <= selected_index < len(candidates):
                        verification = candidates[selected_index].get("verification", {})
                if isinstance(verification, Mapping):
                    critique = _merge_relation_repair_verification(critique, verification)
                critique = _merge_constraint_check(
                    {**critique, "constraint_check": {}},
                    relation_repair.get("post_repair_constraint_check", {}),
                )
            typed_action = None
            if not (
                efficient_repair_accepted
                or (object_repair and object_repair.get("accepted"))
                or (repair and repair.get("accepted"))
                or (relation_repair and relation_repair.get("accepted"))
            ):
                typed_action = self._maybe_run_typed_action_backend(
                    state,
                    selected_prompt,
                    selected_image,
                    critique,
                    constraints,
                    run_dir,
                    events,
                    round_index,
                    repair_plan=repair_plan,
                )
            if typed_action and typed_action.get("accepted"):
                selected_image = str(typed_action["selected_image"])
                selected_prompt = str(typed_action["selected_prompt"])
                image_paths = [*image_paths, *list(typed_action.get("image_paths", []))]
                image_prompts = [*image_prompts, *list(typed_action.get("image_prompts", []))]
                state.image_paths = image_paths
                post_check = typed_action.get("selected_constraint_check")
                if isinstance(post_check, Mapping):
                    critique = _merge_constraint_check(
                        {**critique, "constraint_check": {}},
                        post_check,
                    )
                critique = _mark_accepted_typed_action(critique, typed_action)
            specialist_report = None
            if hard_pass_guard:
                events.append(
                    {
                        "type": "specialist_reports_skipped",
                        "round": round_index,
                        "reason": "question_level_hard_constraints_passed",
                    }
                )
            else:
                specialist_report = self._maybe_run_specialist_reports(
                    state,
                    selected_prompt,
                    selected_image,
                    critique,
                    constraints,
                    run_dir,
                    events,
                    round_index,
                )
                if specialist_report:
                    critique = _merge_specialist_report(critique, specialist_report)
            completion_gate = self.completion_gate(
                critique,
                round_index,
                object_repair=object_repair,
                local_repair=repair,
                relation_repair=relation_repair,
            )
            critique["score"] = completion_gate["score"]
            critique["completion_gate"] = completion_gate
            state.add_feedback(
                {
                    "round": round_index,
                    "source": "visual_reflector",
                    "selection": _strip_large_prompt(selection),
                    "critique": _strip_large_prompt(critique),
                }
            )

            revision_base_prompt = selected_prompt if self.enable_binding_variants else (
                state.refined_prompt or current_prompt
            )
            revised_prompt, optimizer_event = self._revise_prompt(
                state,
                revision_base_prompt,
                critique,
                image_paths,
                events,
                constraints,
            )
            if optimizer_event:
                events.append(optimizer_event)

            record = build_round_record(
                round_index=round_index,
                prompt=selected_prompt,
                images=image_paths,
                selected_image=selected_image,
                feedback=critique,
                revised_prompt=revised_prompt,
            )
            round_records.append(record)
            events.append(
                {
                    "type": "round_completed",
                    "round": round_index,
                    "prompt": selected_prompt,
                    "selected_image": selected_image,
                    "score": critique["score"],
                    "revised_prompt": revised_prompt,
                    "completion_gate": completion_gate,
                }
            )

            if self.should_stop(critique, round_index, completion_gate=completion_gate):
                state.refined_prompt = selected_prompt
                status = "completed"
                events.append(
                    {
                        "type": "stop",
                        "round": round_index,
                        "reason": "completion_gate_passed",
                        "score": critique["score"],
                        "completion_gate": completion_gate,
                    }
                )

            elif _repair_plan_needs_clarification(repair_plan):
                state.refined_prompt = selected_prompt
                status = "needs_clarification"
                events.append(
                    {
                        "type": "stop",
                        "round": round_index,
                        "reason": "unverifiable_or_clarify",
                        "repair_plan": _strip_large_prompt(repair_plan),
                        "completion_gate": completion_gate,
                    }
                )

            elif not self._has_retry_budget(round_index):
                state.refined_prompt = revised_prompt
                status = "max_rounds_reached"
                events.append(
                    {
                        "type": "stop",
                        "round": round_index,
                        "reason": "max_rounds_reached",
                        "completion_gate": completion_gate,
                    }
                )

            else:
                state.refined_prompt = revised_prompt

            self._remember_round(state, record, critique)
            write_state_snapshot(
                run_dir,
                round_index,
                state=state.to_dict(),
                round_record=record,
                status=status,
            )

            if status in {"completed", "max_rounds_reached", "needs_clarification"}:
                break

        final_score = _last_score(round_records)
        selected_image = _last_selected_image(round_records)
        final_selection = _best_final_selection(round_records)
        if final_selection:
            final_score = _coerce_float(final_selection["score"], default=final_score)
            selected_image = str(final_selection["selected_image"])
            state.refined_prompt = str(final_selection["prompt"])
            if int(final_selection["round"]) != len(round_records) - 1:
                events.append(
                    {
                        "type": "best_history_final_selection",
                        "round": final_selection["round"],
                        "selected_image": selected_image,
                        "score": final_score,
                        "reason": final_selection["reason"],
                    }
                )
        final_report_path = write_final_report(
            run_dir,
            run_id=allocated_run_id,
            status=status,
            mode=self.mode,
            user_prompt=state.user_prompt,
            final_prompt=state.active_prompt,
            final_score=final_score,
            selected_image=selected_image,
            round_records=round_records,
        )
        write_json(
            run_dir / "run.json",
            _run_payload(
                allocated_run_id,
                status,
                self.mode,
                self.config,
                constraints,
                state,
                round_records,
                events,
                final_report_path,
                layout_context=layout_context,
            ),
        )

        return OrchestratorResult(
            run_id=allocated_run_id,
            run_dir=str(run_dir),
            status=status,
            mode=self.mode,
            state=state.to_dict(),
            config=self.config.to_dict(),
            round_records=round_records,
            events=events,
            final_report_path=str(final_report_path),
        )

    def should_stop(
        self,
        critique: Mapping[str, Any],
        round_index: int,
        *,
        completion_gate: Mapping[str, Any] | None = None,
    ) -> bool:
        """Return true when a round is good enough."""

        if completion_gate is None and isinstance(critique.get("completion_gate"), Mapping):
            completion_gate = critique["completion_gate"]
        gate = completion_gate or self.completion_gate(critique, round_index)
        return bool(gate.get("passed"))

    def completion_gate(
        self,
        critique: Mapping[str, Any],
        round_index: int,
        *,
        object_repair: Mapping[str, Any] | None = None,
        local_repair: Mapping[str, Any] | None = None,
        relation_repair: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a hard-constraint-aware decision for ending the run."""

        del round_index
        score = _completion_score(critique)
        blockers = _completion_blockers(
            critique,
            object_repair=object_repair,
            local_repair=local_repair,
            relation_repair=relation_repair,
            score_threshold=self.score_threshold,
        )
        return {
            "passed": score >= self.score_threshold and not blockers,
            "score": score,
            "score_threshold": self.score_threshold,
            "score_passed": score >= self.score_threshold,
            "blockers": blockers,
        }

    def _has_retry_budget(self, round_index: int) -> bool:
        return round_index < self.config.max_rounds - 1

    def _maybe_clarify(
        self,
        state: AgentState,
        events: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self.enable_clarifier:
            return None
        result = self.clarifier.decide(
            state.user_prompt,
            history=[
                item
                for item in state.feedback
                if item.get("source") == "proactive_clarifier"
            ],
        )
        state.add_feedback(
            {
                "round": -1,
                "source": "proactive_clarifier",
                "decision": _strip_large_prompt(result),
            }
        )
        events.append(
            {
                "type": "clarification_decision",
                "status": result["status"],
                "question": result.get("question"),
                "ask_score": result.get("ask_score"),
                "missing_slot": result.get("missing_slot"),
            }
        )
        if result["status"] == "ask_user" and self.auto_merge_clarification:
            merged = self.clarifier.merge_answer(
                state.user_prompt,
                result["question"],
                self.auto_merge_clarification,
                belief_state=result.get("belief_state"),
            )
            state.apply_update(merged["update"])
            events.append(
                {
                    "type": "clarification_merged",
                    "question": merged["question"],
                    "answer": merged["answer"],
                    "merged_prompt": merged["merged_prompt"],
                }
            )
            return merged
        return result

    def _select_prompt_for_round(
        self,
        state: AgentState,
        round_records: Sequence[Mapping[str, Any]],
        events: list[dict[str, Any]],
        constraints: PromptConstraints,
        layout_context: Mapping[str, Any] | None = None,
        ) -> str:
        if state.refined_prompt:
            return self._lock_prompt(
                state.refined_prompt,
                constraints,
                events,
                state.round_index,
            )

        prompts = self.prompt_agent.generate_initial_prompts(
            state.user_prompt,
            n=self.prompt_candidates_per_round,
            history=round_records,
        )
        candidates = []
        for prompt in prompts:
            base_prompt = self._lock_prompt(
                prompt,
                constraints,
                events,
                state.round_index,
            )
            guided_prompt = self._apply_layout_guidance(
                base_prompt,
                layout_context,
                events,
                state.round_index,
            )
            locked_prompt = self._lock_prompt(
                guided_prompt,
                constraints,
                events,
                state.round_index,
            )
            candidates.append(
                state.add_candidate(
                    {
                        "prompt": locked_prompt,
                        "base_prompt": base_prompt,
                        "strategy": "initial",
                        "source": "idea2img",
                        "reason": "Initial prompt candidate from PromptAgent.",
                    }
                )
            )
        scored = self.candidate_scorer.score_candidates(
            state.user_prompt,
            candidates,
        )
        selected_candidate = scored[0]["candidate"] if scored else candidates[0]
        selected_prompt = selected_candidate["prompt"]
        state.refined_prompt = str(
            selected_candidate.get("base_prompt") or selected_prompt
        )
        events.append(
            {
                "type": "prompt_selected",
                "round": state.round_index,
                "strategy": "initial",
                "prompt": selected_prompt,
                "base_prompt": state.refined_prompt,
                "candidate_count": len(candidates),
            }
        )
        return selected_prompt

    def _maybe_plan_layout(
        self,
        state: AgentState,
        run_dir: Path,
        events: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self.enable_layout_planner:
            return None
        if self.layout_planner is None:
            raise ValueError("layout_planner is required when enable_layout_planner=True")

        layout = self.layout_planner.plan(
            state.user_prompt,
            canvas_size=self.layout_canvas_size,
        )
        package = layout_to_prompt_package(layout, user_prompt=state.user_prompt)
        enriched_prompt = layout_to_enriched_prompt(state.user_prompt, package)
        generation_hint = layout_to_generation_hint(package)
        context = {
            "layout": _strip_layout_runtime(layout),
            "prompt_package": package,
            "enriched_prompt": enriched_prompt,
            "generation_hint": generation_hint,
        }
        layout_path = write_json(run_dir / "layout.json", context)
        state.add_feedback(
            {
                "round": -1,
                "source": "layout_planner",
                "layout_path": str(layout_path),
                "object_count": len(package["objects"]),
                "generation_order": list(package["generation_order"]),
            }
        )
        events.append(
            {
                "type": "layout_planned",
                "round": -1,
                "layout_path": str(layout_path),
                "object_count": len(package["objects"]),
                "generation_order": list(package["generation_order"]),
                "generation_hint": generation_hint,
            }
        )
        return context

    def _apply_layout_guidance(
        self,
        prompt: str,
        layout_context: Mapping[str, Any] | None,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> str:
        if not layout_context:
            return prompt
        if round_index > 0:
            return prompt
        generation_hint = str(layout_context.get("generation_hint") or "").strip()
        if not generation_hint or _prompt_has_layout_guidance(prompt):
            return prompt
        guided_prompt = f"{generation_hint}, {prompt}"
        events.append(
            {
                "type": "layout_prompt_applied",
                "round": round_index,
                "generation_hint": generation_hint,
                "prompt": guided_prompt,
            }
        )
        return guided_prompt

    def _generate_images(
        self,
        prompt: str,
        image_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
        constraints: PromptConstraints,
    ) -> tuple[list[str], list[str]]:
        _configure_mock_placeholder_dir(self.image_generator, image_dir)
        negative_prompt = merge_negative_prompts(
            build_negative_prompt(constraints) if self.auto_negative_prompt else None,
            self.negative_prompt,
            _generator_negative_prompt(self.image_generator),
        )
        prompt_variants = self._prompt_variants(prompt, constraints, events, round_index)
        prompts = [variant["prompt"] for variant in prompt_variants]
        image_paths = self.image_generator.generate(
            prompts,
            n=len(prompts),
            negative_prompt=negative_prompt,
        )
        generator_metadata = _generator_generation_metadata(self.image_generator)
        image_prompts = _expand_image_prompts(prompts, image_paths, generator_metadata)
        events.append(
            {
                "type": "images_generated",
                "agent": "GenerationEngineAgent",
                "round": round_index,
                "prompt": prompt,
                "prompts": prompts,
                "image_prompts": image_prompts,
                "prompt_variants": prompt_variants,
                "negative_prompt": negative_prompt,
                "image_paths": list(image_paths),
                "generator_metadata": generator_metadata,
                "mode": self.mode,
            }
        )
        return list(image_paths), image_prompts

    def _prompt_variants(
        self,
        prompt: str,
        constraints: PromptConstraints,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> list[dict[str, Any]]:
        if not should_use_binding_variants(
            constraints,
            enabled=self.enable_binding_variants,
        ):
            guarded_prompt, guard_event = _guard_prompt_relation_drift(
                prompt,
                constraints,
                round_index,
                strategy="single",
            )
            if guard_event:
                events.append(guard_event)
            return [
                {
                    "prompt": guarded_prompt,
                    "strategy": "single",
                    "reason": "variants disabled",
                }
                for _ in range(self.config.n_images)
            ]
        variants = build_binding_variants(
            prompt,
            constraints,
            max_variants=max(1, self.config.n_images),
            token_budget=self.clip_token_budget,
        )
        guarded_variants: list[dict[str, Any]] = []
        for variant in variants:
            guarded_prompt, guard_event = _guard_prompt_relation_drift(
                str(variant.get("prompt") or ""),
                constraints,
                round_index,
                strategy=str(variant.get("strategy") or "variant"),
            )
            if guard_event:
                events.append(guard_event)
            guarded_variants.append({**variant, "prompt": guarded_prompt})
        events.append(
            {
                "type": "binding_prompt_variants",
                "round": round_index,
                "variant_count": len(guarded_variants),
                "strategies": [variant["strategy"] for variant in guarded_variants],
            }
        )
        return guarded_variants

    def _revise_prompt(
        self,
        state: AgentState,
        current_prompt: str,
        critique: Mapping[str, Any],
        image_paths: Sequence[str],
        events: list[dict[str, Any]],
        constraints: PromptConstraints,
    ) -> tuple[str, dict[str, Any] | None]:
        if self.should_stop(critique, state.round_index) or not self._has_retry_budget(
            state.round_index
        ):
            return current_prompt, None

        specialist_patch = _specialist_patch_from_critique(critique)
        if specialist_patch:
            revised_prompt = _apply_specialist_prompt_patch(
                current_prompt,
                specialist_patch,
                constraints,
            )
            revised_prompt = self._lock_prompt(
                revised_prompt,
                constraints,
                events,
                state.round_index,
            )
            state.add_candidate(
                {
                    "prompt": revised_prompt,
                    "modified_sentence": specialist_patch["prompt_patch"],
                    "fixes": [specialist_patch["dominant_failure"]],
                    "expected_improvement": (
                        "Apply specialist typed patch while preserving original "
                        "user constraints."
                    ),
                    "risk": "Rule-based patch may over-emphasize the failed fragment.",
                    "source": "specialist_patch_gate",
                    "strategy": "specialist_patch_gate",
                    "round": state.round_index,
                }
            )
            return revised_prompt, {
                "type": "prompt_specialist_patch_gated",
                "round": state.round_index,
                "selected_prompt": revised_prompt,
                "prompt_patch": specialist_patch["prompt_patch"],
                "forbidden_phrases": specialist_patch["forbidden_phrases"],
                "dominant_failure": specialist_patch["dominant_failure"],
                "selected_action": specialist_patch["selected_action"],
            }

        errors = _prioritize_user_grounded_errors(
            current_prompt,
            [
                *_question_level_prompt_errors(current_prompt, critique),
                *[
                    item.to_dict()
                    for item in self.error_analyzer.analyze(current_prompt, critique)
                ],
            ],
            constraints,
        )
        if errors:
            candidates = self.prompt_optimizer.optimize(
                errors,
                num_candidates=self.prompt_candidates_per_round,
                memory=self.memory.to_list(),
            )
            scored = self.candidate_scorer.score_candidates(
                state.user_prompt,
                candidates,
                image_paths=image_paths,
                errors=errors,
            )
            selected = scored[0]["candidate"] if scored else candidates[0]
            revised_prompt = self._lock_prompt(
                selected["prompt"],
                constraints,
                events,
                state.round_index,
            )
            binding_event = None
            if has_binding_failure(critique, constraints):
                revised_prompt, binding_event = self._apply_binding_retry(
                    revised_prompt,
                    critique,
                    constraints,
                    events,
                    state.round_index,
                )
            state.add_candidate(
                {
                    **selected,
                    "prompt": revised_prompt,
                    "strategy": "revision",
                    "round": state.round_index,
                }
            )
            return revised_prompt, {
                "type": "prompt_optimized",
                "round": state.round_index,
                "error_count": len(errors),
                "candidate_count": len(candidates),
                "selected_prompt": revised_prompt,
                "binding_retry": binding_event,
            }

        revised_prompt = self.prompt_agent.revise(
            state.user_prompt,
            current_prompt,
            critique,
            history=events,
        )
        revised_prompt = self._lock_prompt(
            revised_prompt,
            constraints,
            events,
            state.round_index,
        )
        binding_event = None
        if has_binding_failure(critique, constraints):
            revised_prompt, binding_event = self._apply_binding_retry(
                revised_prompt,
                critique,
                constraints,
                events,
                state.round_index,
            )
        state.add_candidate(
            {
                "prompt": revised_prompt,
                "strategy": "revision",
                "source": "idea2img",
                "round": state.round_index,
                "reason": "Fallback revision when no localized prompt errors were found.",
            }
        )
        return revised_prompt, {
            "type": "prompt_revised",
            "round": state.round_index,
            "selected_prompt": revised_prompt,
            "binding_retry": binding_event,
        }

    def _apply_binding_retry(
        self,
        prompt: str,
        critique: Mapping[str, Any],
        constraints: PromptConstraints,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> tuple[str, dict[str, Any]]:
        retry = build_binding_retry_prompt(
            prompt,
            constraints,
            critique,
            token_budget=self.clip_token_budget,
        )
        revised_prompt = retry["prompt"]
        event = {
            "type": "binding_retry_prompt",
            "round": round_index,
            "prompt": revised_prompt,
            "negative_prompt": retry["negative_prompt"],
            "reasons": retry["reasons"],
        }
        events.append(event)
        return revised_prompt, event

    def _maybe_run_specialist_reports(
        self,
        state: AgentState,
        selected_prompt: str,
        selected_image: str,
        critique: Mapping[str, Any],
        constraints: PromptConstraints,
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> dict[str, Any] | None:
        if not self.enable_specialist_reports:
            return None
        observation = _observation_from_existing_feedback(critique, constraints)
        request = ""
        raw_response = ""
        api_call_count = 0
        source = "existing_feedback_and_prompt_drift"
        reports = build_specialist_reports(
            observation,
            constraints,
            generated_prompt=selected_prompt,
        )
        arbitration = arbitrate_specialist_reports(reports, constraints)
        if (
            self.enable_specialist_vlm_observation
            and not _specialist_report_should_gate(
                {
                    "reports": [report.to_dict() for report in reports],
                    "arbitration": arbitration.to_dict(),
                }
            )
        ):
            request = build_specialist_observation_request(
                user_prompt=state.user_prompt,
                generated_prompt=selected_prompt,
                constraints=constraints,
            )
            raw_response = self.vlm.vision(request, [selected_image])
            observation = parse_specialist_observation_response(raw_response)
            reports = build_specialist_reports(
                observation,
                constraints,
                generated_prompt=selected_prompt,
            )
            arbitration = arbitrate_specialist_reports(reports, constraints)
            api_call_count = 1
            source = "structured_vlm_observation"
        specialist_report = {
            "round": round_index,
            "image_path": selected_image,
            "user_prompt": state.user_prompt,
            "generated_prompt": selected_prompt,
            "source": source,
            "api_call_count": api_call_count,
            "request": request,
            "raw_response": raw_response,
            "observation": observation,
            "reports": [report.to_dict() for report in reports],
            "arbitration": arbitration.to_dict(),
        }
        path = write_json(
            run_dir / f"specialist_reports_round_{round_index}.json",
            _strip_large_prompt(specialist_report),
        )
        specialist_report["path"] = str(path)
        events.append(
            {
                "type": "specialist_reports",
                "agent": "ConstraintFusionArbiter",
                "round": round_index,
                "path": str(path),
                "global_passed": arbitration.global_passed,
                "dominant_failure": arbitration.dominant_failure,
                "selected_action": arbitration.selected_action,
                "api_call_count": api_call_count,
            }
        )
        return specialist_report

    def _maybe_reward_rerank(
        self,
        prompt: str,
        image_paths: Sequence[str],
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> dict[str, Any] | None:
        if not self.enable_reward_reranker or self.reward_reranker is None:
            return None
        try:
            ranking = self.reward_reranker.rank(prompt, image_paths)
        except Exception as exc:
            ranking = {
                "reranker": "reward_reranker",
                "prompt": prompt,
                "selected_index": 0,
                "selected_image": image_paths[0] if image_paths else None,
                "scores": [],
                "failed": True,
                "error": str(exc),
            }
            path = write_json(run_dir / f"reward_round_{round_index}.json", ranking)
            events.append(
                {
                    "type": "reward_rerank_failed",
                    "round": round_index,
                    "path": str(path),
                    "error": str(exc),
                }
            )
            return None
        path = write_json(run_dir / f"reward_round_{round_index}.json", ranking)
        events.append(
            {
                "type": "reward_reranked",
                "round": round_index,
                "path": str(path),
                "selected_image": ranking["selected_image"],
                "score": ranking["scores"][0]["score"],
            }
        )
        return ranking

    def _maybe_arbitrate_image_selection(
        self,
        state: AgentState,
        image_prompts: Sequence[str],
        image_paths: Sequence[str],
        selection: Mapping[str, Any],
        reward_ranking: Mapping[str, Any] | None,
        constraints: PromptConstraints,
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> dict[str, Any] | None:
        if len(image_paths) <= 1:
            return None
        if not self.enable_constraint_check:
            return None

        candidate_checks: list[dict[str, Any]] = []
        candidate_question_records: list[dict[str, Any]] = []
        for index, image_path in enumerate(image_paths):
            prompt = image_prompts[index] if index < len(image_prompts) else ""
            record = self._check_constraints_with_questions(
                state.user_prompt,
                prompt,
                image_path,
                constraints,
                history=state.feedback,
            )
            check = record["constraint_check"]
            candidate_question_records.append(
                {
                    "index": index,
                    "image_path": image_path,
                    "prompt": prompt,
                    "questions": record.get("questions", []),
                    "answers": record.get("answers", []),
                    "summary": record.get("summary", {}),
                    "constraint_check": _strip_large_prompt(check),
                    "object_state": _strip_large_prompt(record.get("object_state", {})),
                    "geometry_verification": record.get("geometry_verification", {}),
                    "object_evidence_verification": record.get(
                        "object_evidence_verification", {}
                    ),
                    "evidence_chain": list(record.get("evidence_chain", []) or []),
                    "source": record.get("source", "question_level_vqa"),
                }
            )
            candidate_checks.append(
                {
                    "index": index,
                    "image_path": image_path,
                    "prompt": prompt,
                    "constraint_check": _strip_large_prompt(check),
                }
            )

        arbitration = arbitrate_image_candidates(
            image_paths=image_paths,
            prompts=image_prompts,
            selection=selection,
            reward_ranking=reward_ranking,
            candidate_checks=candidate_checks,
            constraints=constraints.to_dict(),
        )
        if candidate_checks:
            arbitration["candidate_checks"] = candidate_checks
            if candidate_question_records:
                arbitration["candidate_question_records"] = candidate_question_records
            trace_path = write_json(
                run_dir / f"selection_trace_round_{round_index}.json",
                {
                    "round": round_index,
                    **dict(arbitration.get("selection_trace", {})),
                },
            )
            path = write_json(
                run_dir / f"candidate_constraints_round_{round_index}.json",
                {
                    "round": round_index,
                    "selected_index": arbitration["selected_index"],
                    "selected_image": arbitration["selected_image"],
                    "selection_trace_path": str(trace_path),
                    "candidate_checks": candidate_checks,
                    "arbitration": _strip_large_prompt(arbitration),
                },
            )
            question_path = None
            if candidate_question_records:
                question_path = write_json(
                    run_dir / f"candidate_questions_round_{round_index}.json",
                    {
                        "round": round_index,
                        "selected_index": arbitration["selected_index"],
                        "selected_image": arbitration["selected_image"],
                        "selection_trace_path": str(trace_path),
                        "candidate_questions": candidate_question_records,
                        "arbitration": _strip_large_prompt(arbitration),
                    },
                )
            events.append(
                {
                    "type": "candidate_constraints_checked",
                    "round": round_index,
                    "path": str(path),
                    "trace_path": str(trace_path),
                    "question_path": str(question_path) if question_path else None,
                    "selected_index": arbitration["selected_index"],
                    "checked": len(candidate_checks),
                }
            )
            if question_path:
                events.append(
                    {
                        "type": "candidate_questions_checked",
                        "round": round_index,
                        "path": str(question_path),
                        "selected_index": arbitration["selected_index"],
                        "checked": len(candidate_question_records),
                    }
                )
        return arbitration

    def _check_constraints_with_questions(
        self,
        user_prompt: str,
        prompt: str,
        image_path: str,
        constraints: PromptConstraints,
        *,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            record = self.constraint_question_evaluator.evaluate(
                user_prompt,
                prompt,
                image_path,
                constraints,
                history=history,
            )
            return augment_record_with_object_geometry(
                self.vlm,
                record,
                user_prompt=user_prompt,
                prompt=prompt,
                image_path=image_path,
            )
        except Exception as question_exc:
            try:
                check = self.visual_critic.check_constraints(
                    user_prompt,
                    prompt,
                    image_path,
                    constraints,
                    history=history,
                )
            except Exception as legacy_exc:
                check = _failed_constraint_check(prompt, image_path, question_exc)
                check["fallback_error"] = str(legacy_exc)
                return {
                    "image_path": image_path,
                    "prompt": prompt,
                    "questions": [],
                    "answers": [],
                    "summary": {
                        "passed": None,
                        "score": 0.0,
                        "hard_failures": 0,
                        "soft_failures": 0,
                        "uncertain_hard_checks": 0,
                        "failed_constraints": [],
                        "passed_constraints": [],
                        "blocked_constraints": [],
                        "source": "constraint_question_failed",
                    },
                "constraint_check": check,
                "source": "constraint_question_failed",
                "api_failed": True,
            }
            check["source"] = check.get("source", "legacy_constraint_check_fallback")
            return {
                "image_path": image_path,
                "prompt": prompt,
                "questions": [],
                "answers": [],
                "summary": {
                    "passed": check.get("passed"),
                    "score": check.get("score"),
                    "hard_failures": 0,
                    "soft_failures": 0,
                    "uncertain_hard_checks": 0,
                    "failed_constraints": [],
                    "passed_constraints": [],
                    "blocked_constraints": [],
                    "source": "legacy_constraint_check_fallback",
                },
                "constraint_check": check,
                "source": "legacy_constraint_check_fallback",
            }

    def _maybe_evaluate(
        self,
        state: AgentState,
        selected_prompt: str,
        selected_image: str,
        critique: Mapping[str, Any],
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> dict[str, Any] | None:
        if not self.enable_evaluator or self.evaluator is None:
            return None
        try:
            evaluation = self.evaluator.evaluate(
                state.user_prompt,
                selected_prompt,
                selected_image,
                context={"critique": _strip_large_prompt(critique), "round": round_index},
            )
        except Exception as exc:
            evaluation = _failed_evaluation(
                state.user_prompt,
                selected_prompt,
                selected_image,
                exc,
                context={"critique": _strip_large_prompt(critique), "round": round_index},
            )
            path = write_json(run_dir / f"evaluation_round_{round_index}.json", evaluation)
            events.append(
                {
                    "type": "evaluation_failed",
                    "agent": "QualityEvaluatorAgent",
                    "round": round_index,
                    "path": str(path),
                    "error": str(exc),
                }
            )
            return evaluation
        if self.enable_factuality_qa and self.factuality_evaluator is not None:
            try:
                factuality = self.factuality_evaluator.evaluate(
                    state.user_prompt,
                    selected_image,
                )
            except Exception as exc:
                factuality = {"failed": True, "error": str(exc), "score": None}
            evaluation["factuality_qa"] = _strip_large_prompt(factuality)
            if not factuality.get("skipped") and factuality.get("score") is not None:
                evaluation["score"] = min(
                    _coerce_float(evaluation.get("score"), default=0.0),
                    _coerce_float(factuality.get("score"), default=0.0),
                )
                evaluation["passed"] = bool(evaluation.get("passed", False)) and bool(
                    _coerce_float(factuality.get("score"), default=0.0) >= 0.75
                )
                evaluation["errors"] = [
                    *list(evaluation.get("errors", []) or []),
                    *list(factuality.get("errors", []) or []),
                ]
        path = write_json(run_dir / f"evaluation_round_{round_index}.json", evaluation)
        events.append(
            {
                "type": "evaluated",
                "agent": "QualityEvaluatorAgent",
                "round": round_index,
                "path": str(path),
                "score": evaluation.get("score"),
                "passed": evaluation.get("passed"),
                "error_count": len(evaluation.get("errors", []) or []),
            }
        )
        return evaluation

    def _maybe_plan_repair(
        self,
        state: AgentState,
        selected_prompt: str,
        selected_image: str,
        critique: Mapping[str, Any],
        constraints: PromptConstraints,
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> dict[str, Any] | None:
        if not self.enable_repair_planner or self.repair_planner is None:
            return None
        enabled_tools = {
            "recolor": bool(self.enable_local_repair),
            "relation_repair": bool(self.enable_relation_repair),
            "object_insertion": bool(
                self.enable_object_insertion_repair and self.relation_repairer is not None
            ),
            "regenerate": self._has_retry_budget(round_index),
        }
        try:
            plan = self.repair_planner.plan(
                user_prompt=state.user_prompt,
                prompt=selected_prompt,
                image_path=selected_image,
                critique=critique,
                constraints=constraints,
                enabled_tools=enabled_tools,
            )
        except Exception as exc:
            plan = {
                "primary_action": "none",
                "tool_sequence": [],
                "repairable": False,
                "reason": "repair planner failed",
                "error": str(exc),
                "source": "repair_planner_error",
            }
        plan = deepcopy(dict(plan))
        plan = _merge_localized_repair_hint(plan, critique)
        plan["round"] = round_index
        plan["enabled_tools"] = enabled_tools
        _normalize_repair_plan_contract(
            plan,
            critique,
            can_regenerate=bool(enabled_tools.get("regenerate")),
        )
        path = write_json(run_dir / f"repair_plan_round_{round_index}.json", _strip_large_prompt(plan))
        events.append(
            {
                "type": "repair_planned",
                "agent": "QualityEvaluatorAgent",
                "round": round_index,
                "path": str(path),
                "primary_action": plan.get("primary_action"),
                "selected_action": plan.get("selected_action"),
                "fallback_action": plan.get("fallback_action"),
                "error_type": plan.get("error_type"),
                "repairable": plan.get("repairable"),
                "target_object": plan.get("target_object"),
                "reason": plan.get("reason"),
            }
        )
        return plan

    def _maybe_select_repairable_candidate(
        self,
        image_prompts: Sequence[str],
        image_paths: Sequence[str],
        *,
        selected_index: int,
        selected_prompt: str,
        selected_image: str,
        critique: Mapping[str, Any],
        constraints: PromptConstraints,
        arbitration: Mapping[str, Any] | None,
        repair_plan: Mapping[str, Any] | None,
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> dict[str, Any] | None:
        del selected_prompt, selected_image
        if len(image_paths) <= 1:
            return None
        selection = select_repairable_candidate(
            arbitration=arbitration,
            current_index=selected_index,
            critique=critique,
            repair_plan=repair_plan,
            constraints=constraints,
        )
        if not selection:
            return None
        index = int(selection["selected_index"])
        if 0 <= index < len(image_paths):
            selection["selected_image"] = str(image_paths[index])
        if 0 <= index < len(image_prompts):
            selection["selected_prompt"] = str(image_prompts[index])
        path = write_json(
            run_dir / f"repairable_candidate_round_{round_index}.json",
            _strip_large_prompt(selection),
        )
        events.append(
            {
                "type": (
                    "repairability_switch_blocked"
                    if selection.get("blocked")
                    else "repairability_aware_selection"
                ),
                "round": round_index,
                "path": str(path),
                "primary_action": selection.get("primary_action"),
                "previous_index": selection.get("previous_index"),
                "selected_index": selection.get("selected_index"),
                "selected_image": selection.get("selected_image"),
                "reason": selection.get("reason"),
                "blocked": bool(selection.get("blocked", False)),
            }
        )
        return selection

    def _maybe_select_repair_base(
        self,
        image_prompts: Sequence[str],
        image_paths: Sequence[str],
        *,
        selected_index: int,
        constraints: PromptConstraints,
        arbitration: Mapping[str, Any] | None,
        repair_plan: Mapping[str, Any] | None,
        reward_ranking: Mapping[str, Any] | None,
        current_feedback: Mapping[str, Any] | None,
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> dict[str, Any] | None:
        if len(image_paths) <= 1:
            return None
        if not _repair_base_selection_enabled(repair_plan):
            return None
        if not isinstance(arbitration, Mapping):
            return None
        candidate_checks = arbitration.get("candidate_checks", [])
        if not isinstance(candidate_checks, list) or not candidate_checks:
            return None
        try:
            selection = select_repair_base(
                image_paths=image_paths,
                prompts=image_prompts,
                candidate_checks=candidate_checks,
                constraints=constraints,
                repair_plan=repair_plan,
                reward_ranking=reward_ranking,
                current_index=selected_index,
                current_feedback=current_feedback,
            )
        except Exception as exc:
            events.append(
                {
                    "type": "repair_base_selection_failed",
                    "round": round_index,
                    "error": str(exc),
                }
            )
            return None
        path = write_json(
            run_dir / f"repair_base_round_{round_index}.json",
            _strip_large_prompt(selection),
        )
        events.append(
            {
                "type": "repair_base_selected",
                "round": round_index,
                "path": str(path),
                "selected_index": selection.get("selected_index"),
                "selected_image": selection.get("selected_image"),
                "current_index": selection.get("current_index"),
                "overrode_current_selection": selection.get(
                    "overrode_current_selection",
                    False,
                ),
                "intended_action": selection.get("intended_action"),
                "repairability_score": selection.get("repairability_score"),
                "edit_risk_score": selection.get("edit_risk_score"),
            }
        )
        return selection

    def _maybe_local_repair(
        self,
        state: AgentState,
        selected_prompt: str,
        selected_image: str,
        critique: Mapping[str, Any],
        evaluation: Mapping[str, Any] | None,
        constraints: PromptConstraints,
        layout_context: Mapping[str, Any] | None,
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
        repair_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enable_local_repair or not layout_context or not constraints.colors:
            return None
        if not _repair_plan_allows(repair_plan, "recolor"):
            return None
        combined = _merge_evaluation(critique, evaluation or {})
        if not has_binding_failure(combined, constraints):
            return None
        repair_plan = _first_recolor_repair_plan(
            constraints,
            combined,
            preferred_object=(
                str((repair_plan or {}).get("target_object") or "")
                if isinstance(repair_plan, Mapping)
                else None
            ),
        )
        if not repair_plan:
            return None
        edit_dir = run_dir / f"local_edit_round_{round_index}"
        try:
            localized_layout_context, target_localization = self._maybe_locate_local_repair_target(
                selected_image,
                state.user_prompt,
                selected_prompt,
                combined,
                repair_plan,
                layout_context,
            )
            region, detection = detect_color_region_from_layout(
                selected_image,
                localized_layout_context,
                repair_plan["target_name"],
                prompt=repair_plan["prompt"],
                negative_prompt=repair_plan["negative_prompt"],
                source_color=repair_plan["source_color"],
                saturation_threshold=55,
                value_threshold=40,
                search_expand=0.85,
                component_padding=8,
                min_component_area=256,
                selection_strategy="layout_overlap",
                target_region=repair_plan.get("target_region", "full"),
                subtract_target_names=repair_plan.get("subtract_target_names", ()),
                subtract_other_objects=bool(repair_plan.get("subtract_other_objects", False)),
                prefer_object_mask=True,
                image_grounded_bbox=bool(
                    isinstance(target_localization, Mapping)
                    and target_localization.get("applied") is True
                ),
                mask_output_dir=edit_dir,
            )
            if target_localization:
                detection["target_localization"] = target_localization
            mask_refinement = self._maybe_refine_local_repair_mask(
                selected_image=selected_image,
                repair_plan=repair_plan,
                region=region,
                detection=detection,
                constraints=constraints,
                edit_dir=edit_dir,
                run_dir=run_dir,
                events=events,
                round_index=round_index,
            )
            if mask_refinement:
                detection["mask_refinement"] = mask_refinement
                detection["precomputed_mask_path"] = mask_refinement["result"].get(
                    "mask_path"
                )
            before_coverage = measure_color_coverage(
                selected_image,
                bbox=detection["constrained_bbox"],
                target_color=repair_plan["target_color"],
                source_color=repair_plan["source_color"],
                mask_path=detection.get("precomputed_mask_path"),
                exclude_bboxes=detection.get("subtract_bboxes", ()),
                saturation_threshold=55,
                value_threshold=40,
            )
            detection["bbox_provenance"] = _local_repair_bbox_provenance(
                detection,
                before_coverage,
                repair_plan,
            )
            pre_edit_failures = _local_repair_pre_edit_gate_failures(
                detection,
                before_coverage,
                repair_plan,
            )
            if pre_edit_failures:
                repair = {
                    "type": "local_repair",
                    "round": round_index,
                    "accepted": False,
                    "source_image": selected_image,
                    "repair_plan": repair_plan,
                    "region": region.to_dict(),
                    "detection": detection,
                    "coverage_before": before_coverage,
                    "acceptance": {
                        "accepted": False,
                        "score_improved": None,
                        "old_score": _coerce_float(combined.get("score"), default=0.0),
                        "new_score": None,
                        "color_preservation_errors": [],
                        "hard_gate_failures": pre_edit_failures,
                        "coverage_before": before_coverage,
                    },
                    "error": "local repair target is not image-grounded",
                }
                repair["target_evidence"] = _local_repair_target_evidence(
                    repair_plan,
                    detection,
                    {},
                    constraints,
                )
                repair["protected_objects"] = repair["target_evidence"][
                    "protected_objects"
                ]
                repair["expected_post_edit_constraints"] = repair["target_evidence"][
                    "expected_post_edit_constraints"
                ]
                path = write_json(
                    run_dir / f"local_repair_round_{round_index}.json",
                    repair,
                )
                events.append(
                    {
                        "type": "local_repair",
                        "round": round_index,
                        "path": str(path),
                        "accepted": False,
                        "edited_image": None,
                        "post_repair_constraint_passed": None,
                        "error": repair["error"],
                    }
                )
                return repair
            editor = ColorRecolorEditor(
                target_color=repair_plan["target_color"],
                source_color=repair_plan["source_color"],
                saturation_threshold=55,
                value_threshold=40,
                feather_radius=2.5,
                exclude_bboxes=detection.get("subtract_bboxes", ()),
                precomputed_mask_path=detection.get("precomputed_mask_path"),
            )
            edit_result = editor.edit(selected_image, region, edit_dir)
            repair: dict[str, Any] = {
                "type": "local_repair",
                "round": round_index,
                "accepted": False,
                "source_image": selected_image,
                "edited_image": edit_result["edited_image"],
                "repair_plan": repair_plan,
                "region": region.to_dict(),
                "detection": detection,
                "edit_result": edit_result,
                "coverage_before": before_coverage,
            }
            repair["target_evidence"] = _local_repair_target_evidence(
                repair_plan,
                detection,
                edit_result,
                constraints,
            )
            repair["protected_objects"] = repair["target_evidence"]["protected_objects"]
            repair["expected_post_edit_constraints"] = repair["target_evidence"][
                "expected_post_edit_constraints"
            ]
            repair["coverage_after"] = measure_color_coverage(
                str(edit_result["edited_image"]),
                bbox=detection["constrained_bbox"],
                target_color=repair_plan["target_color"],
                source_color=repair_plan["source_color"],
                mask_path=detection.get("precomputed_mask_path"),
                exclude_bboxes=detection.get("subtract_bboxes", ()),
                saturation_threshold=55,
                value_threshold=40,
            )
            component_bbox = detection.get("detected_bbox") or region.bbox
            repair["component_coverage_before"] = measure_color_coverage(
                selected_image,
                bbox=component_bbox,
                target_color=repair_plan["target_color"],
                source_color=repair_plan["source_color"],
                mask_path=detection.get("precomputed_mask_path"),
                exclude_bboxes=detection.get("subtract_bboxes", ()),
                saturation_threshold=55,
                value_threshold=40,
            )
            repair["component_coverage_after"] = measure_color_coverage(
                str(edit_result["edited_image"]),
                bbox=component_bbox,
                target_color=repair_plan["target_color"],
                source_color=repair_plan["source_color"],
                mask_path=detection.get("precomputed_mask_path"),
                exclude_bboxes=detection.get("subtract_bboxes", ()),
                saturation_threshold=55,
                value_threshold=40,
            )
            full_object_mask_path = _full_object_repair_mask_path(detection)
            if full_object_mask_path:
                repair["full_object_coverage_before"] = measure_color_coverage(
                    selected_image,
                    bbox=detection["constrained_bbox"],
                    target_color=repair_plan["target_color"],
                    source_color=repair_plan["source_color"],
                    mask_path=full_object_mask_path,
                    exclude_bboxes=detection.get("subtract_bboxes", ()),
                    saturation_threshold=55,
                    value_threshold=40,
                )
                repair["full_object_coverage_after"] = measure_color_coverage(
                    str(edit_result["edited_image"]),
                    bbox=detection["constrained_bbox"],
                    target_color=repair_plan["target_color"],
                    source_color=repair_plan["source_color"],
                    mask_path=full_object_mask_path,
                    exclude_bboxes=detection.get("subtract_bboxes", ()),
                    saturation_threshold=55,
                    value_threshold=40,
                )
            if self.enable_evaluator and self.evaluator is not None:
                repair_context = {
                    "repair": _strip_large_prompt(repair),
                    "round": round_index,
                    "acceptance_constraints": _repair_acceptance_constraints(
                        constraints,
                        repair_plan,
                    ),
                    "target_verification": _local_repair_target_verification_context(
                        repair_plan,
                        detection,
                        edit_result,
                    ),
                }
                try:
                    repaired_eval = self.evaluator.evaluate(
                        state.user_prompt,
                        selected_prompt,
                        str(edit_result["edited_image"]),
                        context=repair_context,
                    )
                except Exception as exc:
                    repaired_eval = _failed_evaluation(
                        state.user_prompt,
                        selected_prompt,
                        str(edit_result["edited_image"]),
                        exc,
                        context=repair_context,
                    )
                repair["evaluation"] = repaired_eval
                old_score = _coerce_float(combined.get("score"), default=0.0)
                new_score = _coerce_float(repaired_eval.get("score"), default=0.0)
                acceptance = _local_repair_acceptance(
                    repaired_eval,
                    constraints,
                    repair_plan,
                    old_score=old_score,
                    new_score=new_score,
                    detection=detection,
                    coverage_before=repair["coverage_before"],
                    coverage_after=repair["coverage_after"],
                    component_coverage_before=repair["component_coverage_before"],
                    component_coverage_after=repair["component_coverage_after"],
                    full_object_coverage_before=repair.get("full_object_coverage_before"),
                    full_object_coverage_after=repair.get("full_object_coverage_after"),
                )
                repair["acceptance"] = acceptance
                repair["accepted"] = acceptance["accepted"]
                if repair["accepted"]:
                    post_check = self._check_local_repair_constraints(
                        state,
                        selected_prompt,
                        str(edit_result["edited_image"]),
                        constraints,
                        repair,
                        round_index,
                    )
                    _apply_local_repair_post_check(repair, post_check)
            else:
                hard_gate_failures = _local_recolor_hard_gate_failures(
                    detection,
                    repair["coverage_before"],
                    repair["coverage_after"],
                    component_coverage_before=repair["component_coverage_before"],
                    component_coverage_after=repair["component_coverage_after"],
                    full_object_coverage_before=repair.get("full_object_coverage_before"),
                    full_object_coverage_after=repair.get("full_object_coverage_after"),
                )
                repair["acceptance"] = {
                    "accepted": not hard_gate_failures,
                    "score_improved": None,
                    "old_score": None,
                    "new_score": None,
                    "color_preservation_errors": [],
                    "hard_gate_failures": hard_gate_failures,
                    "coverage_before": repair["coverage_before"],
                    "coverage_after": repair["coverage_after"],
                    "component_coverage_before": repair["component_coverage_before"],
                    "component_coverage_after": repair["component_coverage_after"],
                    "full_object_coverage_before": repair.get("full_object_coverage_before", {}),
                    "full_object_coverage_after": repair.get("full_object_coverage_after", {}),
                }
                repair["accepted"] = repair["acceptance"]["accepted"]
                if repair["accepted"]:
                    post_check = self._check_local_repair_constraints(
                        state,
                        selected_prompt,
                        str(edit_result["edited_image"]),
                        constraints,
                        repair,
                        round_index,
                    )
                    _apply_local_repair_post_check(repair, post_check)
        except Exception as exc:
            repair = {
                "type": "local_repair",
                "round": round_index,
                "accepted": False,
                "error": str(exc),
            }
        path = write_json(run_dir / f"local_repair_round_{round_index}.json", repair)
        events.append(
            {
                "type": "local_repair",
                "round": round_index,
                "path": str(path),
                "accepted": repair.get("accepted", False),
                "edited_image": repair.get("edited_image"),
                "post_repair_constraint_passed": (
                    repair.get("post_repair_constraint_check", {}).get("passed")
                    if isinstance(repair.get("post_repair_constraint_check"), Mapping)
                    else None
                ),
                "error": repair.get("error"),
            }
        )
        return repair

    def _maybe_locate_local_repair_target(
        self,
        selected_image: str,
        user_prompt: str,
        selected_prompt: str,
        critique: Mapping[str, Any],
        repair_plan: Mapping[str, Any],
        layout_context: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], dict[str, Any] | None]:
        if not self.enable_vlm_target_locator:
            return layout_context, None
        target_name = str(repair_plan.get("target_name") or "").strip()
        if not target_name:
            return layout_context, None
        target_region = str(repair_plan.get("target_region") or "full")
        bbox, diagnostics = locate_target_region(
            self.vlm,
            selected_image,
            user_prompt=user_prompt,
            prompt=selected_prompt,
            target_name=target_name,
            target_region=target_region,
            repair_goal=str(repair_plan.get("prompt") or ""),
            critique=critique,
            layout_context=layout_context,
        )
        if bbox is None:
            return layout_context, diagnostics
        image_size = tuple(diagnostics.get("image_size", []))
        if len(image_size) != 2:
            return layout_context, diagnostics
        localized_layout = layout_with_target_bbox(
            layout_context,
            target_name,
            bbox,
            image_size=(int(image_size[0]), int(image_size[1])),
        )
        diagnostics["applied"] = True
        return localized_layout, diagnostics

    def _maybe_refine_local_repair_mask(
        self,
        *,
        selected_image: str,
        repair_plan: Mapping[str, Any],
        region: Any,
        detection: Mapping[str, Any],
        constraints: PromptConstraints,
        edit_dir: Path,
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> dict[str, Any] | None:
        if not self.enable_mask_refiner or self.mask_refiner is None:
            return None
        target_name = str(repair_plan.get("target_name") or region.name or "").strip()
        prompt_bbox = _mask_refinement_prompt_bbox(detection, region)
        result = refine_bbox_mask(
            self.mask_refiner,
            selected_image,
            target_name,
            prompt_bbox,
            output_dir=edit_dir / "mask_refine",
            protected_bboxes=detection.get("subtract_bboxes", ()),
            source=_mask_refinement_source(detection),
        )
        prior_mask_path = detection.get("precomputed_mask_path")
        if prior_mask_path:
            result = constrain_refined_mask_to_prior(
                result,
                prior_mask_path=str(prior_mask_path),
                output_dir=edit_dir / "mask_refine",
                protected_bboxes=detection.get("subtract_bboxes", ()),
                prefix="target_prior",
            )
        log = {
            "type": "mask_refinement",
            "round": round_index,
            "target_name": target_name,
            "target_region": repair_plan.get("target_region", "full"),
            "source_image": selected_image,
            "prompt_bbox": prompt_bbox,
            "locator": _strip_large_prompt(detection.get("target_localization", {})),
            "detection_method": detection.get("method"),
            "layout_bbox_scaled": detection.get("layout_bbox_scaled"),
            "constrained_bbox": detection.get("constrained_bbox"),
            "target_prior_mask_path": str(prior_mask_path) if prior_mask_path else None,
            "protected_bboxes": list(detection.get("subtract_bboxes", []) or []),
            "protected_objects": _protected_objects_for_target(constraints, target_name),
            "result": result,
            "geometry_checks": _mask_refinement_geometry_checks(result),
            "vram_note": result.get("vram_note"),
        }
        path = write_json(run_dir / f"mask_refine_round_{round_index}.json", log)
        log["path"] = str(path)
        events.append(
            {
                "type": "mask_refined",
                "round": round_index,
                "path": str(path),
                "target_name": target_name,
                "method": result.get("method"),
                "mask_path": result.get("mask_path"),
                "area_ratio": result.get("area_ratio"),
                "fallback_used": result.get("fallback_used"),
            }
        )
        return log

    def _maybe_object_insertion_repair(
        self,
        state: AgentState,
        selected_prompt: str,
        selected_image: str,
        critique: Mapping[str, Any],
        constraints: PromptConstraints,
        layout_context: Mapping[str, Any] | None,
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
        *,
        repair_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        del critique
        if not _repair_plan_allows(repair_plan, "object_insertion"):
            return None
        if not self.enable_object_insertion_repair or self.relation_repairer is None:
            return None
        editor = getattr(self.relation_repairer, "editor", None)
        if editor is None:
            return None
        target_name = str((repair_plan or {}).get("target_object") or "").strip()
        if not target_name:
            return None
        edit_dir = run_dir / f"object_repair_round_{round_index}"
        prompt = _object_insertion_prompt(target_name, state.user_prompt, constraints)
        negative_prompt = _object_insertion_negative_prompt(target_name, constraints)
        try:
            region = _object_insertion_region(
                layout_context,
                target_name,
                repair_plan or {},
                prompt=prompt,
                negative_prompt=negative_prompt,
                canvas_size=self.layout_canvas_size,
            )
            edit_result = editor.edit(selected_image, region, edit_dir)
            hard_gate_failures = _object_insertion_hard_gate_failures(
                region,
                edit_result,
            )
            repair: dict[str, Any] = {
                "type": "object_insertion_repair",
                "round": round_index,
                "accepted": False,
                "source_image": selected_image,
                "edited_image": str(edit_result["edited_image"]),
                "repair_plan": deepcopy(dict(repair_plan or {})),
                "region": region.to_dict(),
                "edit_result": edit_result,
            }
            repair["target_evidence"] = _object_insertion_target_evidence(
                target_name,
                region.to_dict(),
                edit_result,
                constraints,
            )
            repair["protected_objects"] = repair["target_evidence"]["protected_objects"]
            repair["expected_post_edit_constraints"] = repair["target_evidence"][
                "expected_post_edit_constraints"
            ]
            post_check = self._check_object_insertion_constraints(
                state,
                selected_prompt,
                str(edit_result["edited_image"]),
                constraints,
                repair,
                round_index,
            )
            repair["post_repair_constraint_check"] = post_check
            post_failures = _post_repair_constraint_failures(post_check)
            repair["acceptance"] = {
                "accepted": (
                    bool(post_check.get("passed"))
                    and not post_check.get("failed")
                    and not hard_gate_failures
                    and not post_failures
                ),
                "hard_gate_failures": hard_gate_failures,
                "post_repair_constraint_failures": post_failures,
            }
            repair["accepted"] = bool(repair["acceptance"]["accepted"])
        except Exception as exc:
            repair = {
                "type": "object_insertion_repair",
                "round": round_index,
                "accepted": False,
                "error": str(exc),
                "repair_plan": deepcopy(dict(repair_plan or {})),
            }
        path = write_json(run_dir / f"object_repair_round_{round_index}.json", repair)
        events.append(
            {
                "type": "object_insertion_repair",
                "round": round_index,
                "path": str(path),
                "accepted": repair.get("accepted", False),
                "edited_image": repair.get("edited_image"),
                "target_object": target_name,
                "error": repair.get("error"),
            }
        )
        return repair

    def _maybe_efficient_repair(
        self,
        state: AgentState,
        selected_prompt: str,
        selected_image: str,
        critique: Mapping[str, Any],
        constraints: PromptConstraints,
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
        *,
        repair_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        del critique
        if not self.enable_efficient_repair_agent or self.efficient_repair_agent is None:
            return None
        if not isinstance(repair_plan, Mapping):
            return None
        route = route_repair_kind(repair_plan, state.user_prompt)
        if route not in {
            "text_overlay",
            "symbol_overlay",
            "shape_overlay",
            "bbox_shape_inpaint",
            "existing_object_inpaint",
        }:
            events.append(
                {
                    "type": "efficient_repair_route_skipped",
                    "round": round_index,
                    "route": route,
                    "reason": "route is not an efficient editing route in the main loop",
                }
            )
            return None
        repair_plan = self._augment_efficient_repair_plan_with_target_bbox(
            route=route,
            repair_plan=repair_plan,
            selected_image=selected_image,
            selected_prompt=selected_prompt,
            user_prompt=state.user_prompt,
            constraints=constraints,
            run_dir=run_dir,
            events=events,
            round_index=round_index,
        )
        gate = _efficient_repair_gate(
            route,
            repair_plan,
            self.efficient_repair_agent,
            canvas_size=self.layout_canvas_size,
        )
        if not gate["allowed"]:
            events.append(
                {
                    "type": "efficient_repair_route_skipped",
                    "round": round_index,
                    "route": route,
                    "reason": gate["reason"],
                    "gate": gate,
                }
            )
            return None
        request = _efficient_repair_request_from_plan(
            route,
            repair_plan,
            selected_image,
            state.user_prompt,
            constraints,
            output_dir=run_dir / f"efficient_repair_round_{round_index}",
            canvas_size=self.layout_canvas_size,
        )
        try:
            repair = self.efficient_repair_agent.repair(request)
        except Exception as exc:
            repair = {
                "ok": False,
                "accepted": False,
                "type": "efficient_repair_agent",
                "route": route,
                "source_image": selected_image,
                "error": str(exc),
                "gpu_used": False,
                "sam2_used": False,
                "powerpaint_used": False,
            }
        repair = deepcopy(dict(repair))
        repair["round"] = round_index
        repair["repair_plan"] = deepcopy(dict(repair_plan))
        repair["accepted"] = False
        if repair.get("ok") and repair.get("edited_image"):
            post_check = self._check_object_insertion_constraints(
                state,
                selected_prompt,
                str(repair["edited_image"]),
                constraints,
                repair,
                round_index,
            )
            repair["post_repair_constraint_check"] = post_check
            post_failures = _post_repair_constraint_failures(post_check)
            ocr_failures = _ocr_repair_failures(repair)
            repair["acceptance"] = {
                "accepted": (
                    bool(post_check.get("passed"))
                    and not post_check.get("failed")
                    and not post_failures
                    and not ocr_failures
                ),
                "post_repair_constraint_failures": post_failures,
                "ocr_failures": ocr_failures,
            }
            repair["accepted"] = bool(repair["acceptance"]["accepted"])
        path = write_json(run_dir / f"efficient_repair_round_{round_index}.json", _strip_large_prompt(repair))
        events.append(
            {
                "type": "efficient_repair",
                "round": round_index,
                "path": str(path),
                "route": route,
                "accepted": repair.get("accepted", False),
                "edited_image": repair.get("edited_image"),
                "gpu_used": repair.get("gpu_used", False),
                "sam2_used": repair.get("sam2_used", False),
                "powerpaint_used": repair.get("powerpaint_used", False),
                "error": repair.get("error"),
            }
        )
        return repair

    def _augment_efficient_repair_plan_with_target_bbox(
        self,
        *,
        route: str,
        repair_plan: Mapping[str, Any],
        selected_image: str,
        selected_prompt: str,
        user_prompt: str,
        constraints: PromptConstraints,
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> Mapping[str, Any]:
        if route not in {"bbox_shape_inpaint", "existing_object_inpaint", "shape_overlay"}:
            return repair_plan
        if _repair_plan_bbox_or_none(repair_plan) is not None:
            return repair_plan
        typed_route = str(repair_plan.get("typed_route") or "")
        if typed_route == "occlusion_object_insertion":
            return repair_plan
        if not self.enable_vlm_target_locator:
            return repair_plan
        target = _target_for_efficient_bbox_localization(repair_plan, constraints)
        if not target:
            return repair_plan
        bbox, diagnostics = locate_target_region(
            self.vlm,
            selected_image,
            user_prompt=user_prompt,
            prompt=selected_prompt,
            target_name=target,
            target_region=str(repair_plan.get("target_region") or "full"),
            repair_goal=str(repair_plan.get("reason") or repair_plan.get("edit_prompt") or ""),
            critique={},
            layout_context={},
        )
        diagnostics = dict(diagnostics)
        diagnostics_path = write_json(
            run_dir / f"efficient_repair_target_locator_round_{round_index}.json",
            _strip_large_prompt(diagnostics),
        )
        events.append(
            {
                "type": "efficient_repair_target_locator",
                "round": round_index,
                "route": route,
                "typed_route": typed_route,
                "target_object": target,
                "found": bbox is not None,
                "bbox": bbox,
                "confidence": diagnostics.get("confidence"),
                "path": str(diagnostics_path),
            }
        )
        if bbox is None:
            return repair_plan
        updated = deepcopy(dict(repair_plan))
        updated["target_object"] = target
        updated["target_bbox"] = [int(value) for value in bbox]
        updated["bbox_confidence"] = diagnostics.get("confidence", 0.0)
        updated["target_locator"] = {
            "source": "vlm_target_region_locator",
            "path": str(diagnostics_path),
            "reason": diagnostics.get("reason", ""),
        }
        return updated

    def _maybe_relation_repair(
        self,
        state: AgentState,
        selected_prompt: str,
        selected_image: str,
        critique: Mapping[str, Any],
        constraints: PromptConstraints,
        layout_context: Mapping[str, Any] | None,
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
        repair_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enable_relation_repair or self.relation_repairer is None:
            return None
        if not _repair_plan_allows(repair_plan, "relation_repair"):
            return None
        if not self.relation_repairer.should_repair(
            state.user_prompt,
            critique,
            constraints,
        ):
            return None
        edit_dir = run_dir / f"relation_repair_round_{round_index}"
        try:
            repair = self.relation_repairer.repair(
                user_prompt=state.user_prompt,
                prompt=selected_prompt,
                image_path=selected_image,
                critique=critique,
                constraints=constraints,
                output_dir=edit_dir,
                layout_context=layout_context,
                round_index=round_index,
            )
            if isinstance(repair, dict):
                repair["target_evidence"] = _relation_repair_target_evidence(
                    repair,
                    constraints,
                )
                repair["protected_objects"] = repair["target_evidence"][
                    "protected_objects"
                ]
                repair["expected_post_edit_constraints"] = repair["target_evidence"][
                    "expected_post_edit_constraints"
                ]
        except Exception as exc:
            repair = {
                "type": "relation_action_repair",
                "round": round_index,
                "accepted": False,
                "error": str(exc),
            }
        path = write_json(run_dir / f"relation_repair_round_{round_index}.json", repair)
        events.append(
            {
                "type": "relation_action_repair",
                "round": round_index,
                "path": str(path),
                "accepted": repair.get("accepted", False),
                "edited_image": repair.get("edited_image"),
                "score": repair.get("score"),
                "error": repair.get("error"),
            }
        )
        return repair

    def _maybe_run_typed_action_backend(
        self,
        state: AgentState,
        selected_prompt: str,
        selected_image: str,
        critique: Mapping[str, Any],
        constraints: PromptConstraints,
        run_dir: Path,
        events: list[dict[str, Any]],
        round_index: int,
        *,
        repair_plan: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.enable_typed_action_backend:
            return None
        if not self._has_retry_budget(round_index):
            return None
        if not isinstance(repair_plan, Mapping):
            return None
        route = str(repair_plan.get("typed_route") or "").strip()
        if not typed_action_backend_route(route):
            return None

        variants = build_typed_action_prompt_variants(
            route=route,
            user_prompt=state.user_prompt,
            selected_prompt=selected_prompt,
            repair_plan=repair_plan,
            critique=critique,
            constraints=constraints,
            count=min(self.typed_action_candidates, self.typed_action_max_candidates),
        )
        if not variants:
            return None

        image_dir = run_dir / "images"
        prompts = [
            self._lock_prompt(str(item["prompt"]), constraints, events, round_index)
            for item in variants
        ]
        _configure_mock_placeholder_dir(self.image_generator, image_dir)
        negative_prompt = merge_negative_prompts(
            build_negative_prompt(constraints) if self.auto_negative_prompt else None,
            self.negative_prompt,
            _generator_negative_prompt(self.image_generator),
        )
        try:
            image_paths = self.image_generator.generate(
                prompts,
                n=len(prompts),
                negative_prompt=negative_prompt,
            )
        except Exception as exc:
            payload = {
                "round": round_index,
                "route": route,
                "accepted": False,
                "error": str(exc),
                "variants": variants,
                "selected_prompt": selected_prompt,
                "selected_image": selected_image,
            }
            path = write_json(
                run_dir / f"typed_action_round_{round_index}.json",
                _strip_large_prompt(payload),
            )
            events.append(
                {
                    "type": "typed_action_backend_failed",
                    "round": round_index,
                    "route": route,
                    "path": str(path),
                    "error": str(exc),
                }
            )
            return payload

        generator_metadata = _generator_generation_metadata(self.image_generator)
        image_prompts = _expand_image_prompts(prompts, image_paths, generator_metadata)
        candidate_checks: list[dict[str, Any]] = []
        for index, (prompt, image_path) in enumerate(zip(image_prompts, image_paths)):
            try:
                check = self.visual_critic.check_constraints(
                    state.user_prompt,
                    prompt,
                    image_path,
                    constraints,
                    history=state.feedback,
                )
            except Exception as exc:
                check = _failed_constraint_check(prompt, image_path, exc)
            check = deepcopy(dict(check))
            check["candidate_index"] = index
            check["prompt"] = prompt
            check["image_path"] = image_path
            candidate_checks.append(check)

        best_index = best_typed_action_candidate_index(candidate_checks)
        selected_check = candidate_checks[best_index] if candidate_checks else {}
        accepted = bool(selected_check.get("passed")) and not selected_check.get("failed")
        payload = {
            "round": round_index,
            "route": route,
            "accepted": accepted,
            "repair_plan": deepcopy(dict(repair_plan)),
            "variants": variants,
            "candidate_checks": candidate_checks,
            "selected_index": best_index,
            "selected_prompt": image_prompts[best_index] if image_prompts else selected_prompt,
            "selected_image": image_paths[best_index] if image_paths else selected_image,
            "selected_constraint_check": selected_check,
            "image_paths": list(image_paths),
            "image_prompts": list(image_prompts),
            "negative_prompt": negative_prompt,
            "generator_metadata": generator_metadata,
            "rejected_reasons": typed_action_rejected_reasons(candidate_checks, best_index),
        }
        path = write_json(
            run_dir / f"typed_action_round_{round_index}.json",
            _strip_large_prompt(payload),
        )
        events.append(
            {
                "type": "typed_action_backend",
                "round": round_index,
                "route": route,
                "accepted": accepted,
                "path": str(path),
                "candidate_count": len(candidate_checks),
                "selected_index": best_index,
                "selected_score": selected_check.get("score"),
            }
        )
        return payload

    def _check_local_repair_constraints(
        self,
        state: AgentState,
        selected_prompt: str,
        edited_image: str,
        constraints: PromptConstraints,
        repair: Mapping[str, Any],
        round_index: int,
    ) -> dict[str, Any]:
        prompt_with_repair_context = _prompt_with_local_repair_context(
            state.user_prompt,
            repair,
        )
        try:
            check = self.visual_critic.check_constraints(
                state.user_prompt,
                prompt_with_repair_context,
                edited_image,
                constraints,
                history=[],
            )
        except Exception as exc:
            return _failed_constraint_check(prompt_with_repair_context, edited_image, exc)
        check["round"] = round_index
        check["source"] = "post_local_repair_constraint_check"
        return check

    def _check_object_insertion_constraints(
        self,
        state: AgentState,
        selected_prompt: str,
        edited_image: str,
        constraints: PromptConstraints,
        repair: Mapping[str, Any],
        round_index: int,
    ) -> dict[str, Any]:
        prompt_with_repair_context = _prompt_with_object_repair_context(
            selected_prompt,
            repair,
        )
        try:
            check = self.visual_critic.check_constraints(
                state.user_prompt,
                prompt_with_repair_context,
                edited_image,
                constraints,
                history=state.feedback,
            )
        except Exception as exc:
            return _failed_constraint_check(prompt_with_repair_context, edited_image, exc)
        check["round"] = round_index
        check["source"] = "post_object_insertion_constraint_check"
        return check

    def _check_relation_repair_constraints(
        self,
        state: AgentState,
        selected_prompt: str,
        edited_image: str,
        constraints: PromptConstraints,
        repair: Mapping[str, Any],
        round_index: int,
    ) -> dict[str, Any]:
        prompt_with_repair_context = _prompt_with_relation_repair_context(
            selected_prompt,
            repair,
        )
        try:
            check = self.visual_critic.check_constraints(
                state.user_prompt,
                prompt_with_repair_context,
                edited_image,
                constraints,
                history=state.feedback,
            )
        except Exception as exc:
            return _failed_constraint_check(prompt_with_repair_context, edited_image, exc)
        check["round"] = round_index
        check["source"] = "post_relation_repair_constraint_check"
        return check

    def _lock_prompt(
        self,
        prompt: str,
        constraints: PromptConstraints,
        events: list[dict[str, Any]],
        round_index: int,
    ) -> str:
        result = lock_prompt_to_user_constraints(
            prompt,
            constraints,
            token_budget=self.clip_token_budget,
        )
        if result["prompt"] != prompt or result["violations"] or result["warnings"]:
            events.append(
                {
                    "type": "prompt_constraints_applied",
                    "round": round_index,
                    "token_count": result["token_count"],
                    "token_budget": result["token_budget"],
                    "applied": result["applied"],
                    "violations": result["violations"],
                    "warnings": result["warnings"],
                }
            )
        return result["prompt"]

    def _remember_round(
        self,
        state: AgentState,
        record: Mapping[str, Any],
        critique: Mapping[str, Any],
    ) -> None:
        self.memory.append(
            {
                "round": record["round"],
                "user_prompt": state.user_prompt,
                "prompt": record["prompt"],
                "selected_image": record["selected_image"],
                "score": critique.get("score"),
                "revision_hint": critique.get("revision_hint"),
                "revised_prompt": record["revised_prompt"],
                "mode": self.mode,
            }
        )
        state.memory = self.memory.to_list()

    def _m6_config_payload(self) -> dict[str, Any]:
        return {
            "enable_evaluator": self.enable_evaluator,
            "enable_factuality_qa": self.enable_factuality_qa,
            "enable_reward_reranker": self.enable_reward_reranker,
            "reward_rerank_override": self.reward_rerank_override,
            "enable_local_repair": self.enable_local_repair,
            "enable_vlm_target_locator": self.enable_vlm_target_locator,
            "enable_relation_repair": self.enable_relation_repair,
            "enable_object_insertion_repair": self.enable_object_insertion_repair,
            "enable_repair_planner": self.enable_repair_planner,
            "enable_specialist_reports": self.enable_specialist_reports,
            "enable_specialist_vlm_observation": self.enable_specialist_vlm_observation,
            "enable_typed_action_backend": self.enable_typed_action_backend,
            "typed_action_candidates": self.typed_action_candidates,
            "typed_action_max_candidates": self.typed_action_max_candidates,
            "enable_mask_refiner": self.enable_mask_refiner,
            "mask_refiner": type(self.mask_refiner).__name__
            if self.mask_refiner is not None
            else None,
        }


def _config_payload(
    config: AgentConfig,
    mode: str,
    constraints: PromptConstraints | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "adapter_contract": {
            "llm": "LLMClient",
            "vlm": "VLMClient",
            "image_generator": "ImageGenerator",
        },
        "config": config.to_dict(),
        "prompt_constraints": constraints.to_dict() if constraints else None,
    }


def _failed_constraint_check(
    prompt: str,
    image_path: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "passed": None,
        "score": None,
        "checks": [],
        "errors": [],
        "strengths": [],
        "revision_hint": "",
        "user_grounded": True,
        "failed": True,
        "error": str(exc),
        "prompt": prompt,
        "image_path": image_path,
    }


def _failed_selection(
    prompts: Sequence[str],
    image_paths: Sequence[str],
    exc: Exception,
) -> dict[str, Any]:
    if not image_paths:
        raise ValueError("cannot fallback-select without image paths")
    selected_prompt = prompts[0] if prompts else ""
    return {
        "selected_index": 0,
        "selected_image": image_paths[0],
        "selected_prompt": selected_prompt,
        "scores": [
            {
                "index": 0,
                "score": 0.0,
                "reason": "fallback selection after VLM selection failed",
            }
        ],
        "failed": True,
        "error": str(exc),
    }


def _selected_candidate_constraint_check(
    arbitration: Mapping[str, Any] | None,
    selected_index: int,
) -> dict[str, Any] | None:
    if not isinstance(arbitration, Mapping):
        return None
    candidate_checks = arbitration.get("candidate_checks", [])
    if not isinstance(candidate_checks, list):
        return None
    for item in candidate_checks:
        if not isinstance(item, Mapping):
            continue
        if int(item.get("index", -1)) != selected_index:
            continue
        check = item.get("constraint_check")
        if isinstance(check, Mapping):
            return deepcopy(dict(check))
    return None


def _arbitration_with_current_feedback(
    arbitration: Mapping[str, Any] | None,
    selected_index: int,
    critique: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(arbitration, Mapping):
        return None
    result = deepcopy(dict(arbitration))
    candidate_checks = result.get("candidate_checks", [])
    if not isinstance(candidate_checks, list):
        return result
    for item in candidate_checks:
        if not isinstance(item, Mapping):
            continue
        if int(item.get("index", -1)) != int(selected_index):
            continue
        check = item.get("constraint_check")
        if isinstance(check, Mapping):
            item["constraint_check"] = _merge_current_feedback_constraint_check(
                check,
                critique,
            )
        break
    return result


def _merge_current_feedback_constraint_check(
    check: Mapping[str, Any],
    critique: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(check))
    checks = _list_mapping_records(result.get("checks"))
    errors = _list_mapping_records(result.get("errors"))
    fresh_checks: list[dict[str, Any]] = []
    fresh_errors: list[dict[str, Any]] = []
    for source_key in ("constraint_check", "evaluation", "relation_repair_verification"):
        nested = critique.get(source_key)
        if isinstance(nested, Mapping):
            fresh_checks.extend(_list_mapping_records(nested.get("checks")))
            fresh_errors.extend(_list_mapping_records(nested.get("errors")))
    fresh_checks.extend(_list_mapping_records(critique.get("checks")))
    fresh_errors.extend(_list_mapping_records(critique.get("errors")))
    if fresh_checks or fresh_errors:
        checks = _remove_contradicted_passes(
            [*checks, *fresh_checks],
            fresh_checks,
            fresh_errors,
        )
        errors.extend(fresh_errors)
        result["checks"] = _dedupe_mapping_records(checks)
        result["errors"] = _dedupe_mapping_records(errors)
        if any(item.get("passed") is False for item in checks) or errors:
            result["passed"] = False
    score_values = [
        _coerce_float(value, default=None)
        for value in (
            result.get("score"),
            critique.get("score"),
            _nested_mapping_value(critique, "constraint_check", "score"),
            _nested_mapping_value(critique, "evaluation", "score"),
        )
    ]
    valid_scores = [value for value in score_values if value is not None]
    if valid_scores:
        result["score"] = min(valid_scores)
        result["constraint_score"] = result["score"]
    result["merged_current_feedback"] = True
    return result


def _remove_contradicted_passes(
    checks: Sequence[Mapping[str, Any]],
    fresh_checks: Sequence[Mapping[str, Any]],
    fresh_errors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    failed_targets = _failed_feedback_targets(fresh_checks, fresh_errors)
    if not failed_targets:
        return [deepcopy(dict(item)) for item in checks]
    result: list[dict[str, Any]] = []
    for item in checks:
        target = _canonical_feedback_target(item)
        if item.get("passed") is True and target and target in failed_targets:
            continue
        result.append(deepcopy(dict(item)))
    return result


def _failed_feedback_targets(
    fresh_checks: Sequence[Mapping[str, Any]],
    fresh_errors: Sequence[Mapping[str, Any]],
) -> set[str]:
    targets: set[str] = set()
    for item in fresh_checks:
        if item.get("passed") is False:
            target = _canonical_feedback_target(item)
            if target:
                targets.add(target)
    for item in fresh_errors:
        target = _canonical_feedback_target(item)
        if target:
            targets.add(target)
    return targets


def _canonical_feedback_target(item: Mapping[str, Any]) -> str:
    for key in ("target", "prompt_span"):
        target = _generic_feedback_target(str(item.get(key) or ""))
        if target:
            return target
    text = " ".join(
        str(item.get(key) or "").lower()
        for key in ("expected", "evidence", "description")
    )
    if any(term in text for term in _RELATION_FEEDBACK_TERMS):
        return "relation_action"
    return _generic_feedback_target(text)


_RELATION_FEEDBACK_TERMS = (
    "handle",
    "contact",
    "hold",
    "holding",
    "grip",
    "gripping",
    "carry",
    "carrying",
    "touch",
    "touching",
    "wear",
    "wearing",
    "ride",
    "riding",
    "attach",
    "attached",
)


_FEEDBACK_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "be",
    "but",
    "by",
    "clearly",
    "color",
    "constraint",
    "expected",
    "failed",
    "for",
    "from",
    "in",
    "is",
    "not",
    "of",
    "on",
    "or",
    "passed",
    "relation",
    "should",
    "the",
    "to",
    "visible",
    "visibly",
    "with",
    "wrong",
}


def _generic_feedback_target(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()
    if not cleaned:
        return ""
    words = [word for word in cleaned.split() if word not in _FEEDBACK_STOP_WORDS]
    if not words:
        return ""
    if len(words) >= 2 and words[0] in {"red", "blue", "green", "yellow", "cyan", "purple", "orange", "black", "white", "silver", "gold", "pink", "brown", "gray", "grey"}:
        words = words[1:]
    return words[0].rstrip("s") if words else ""


def _list_mapping_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [deepcopy(dict(value))]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]
    return []


def _dedupe_mapping_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in records:
        key = json.dumps(dict(item), ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            result.append(deepcopy(dict(item)))
    return result


def _nested_mapping_value(mapping: Mapping[str, Any], nested_key: str, value_key: str) -> Any:
    nested = mapping.get(nested_key)
    if isinstance(nested, Mapping):
        return nested.get(value_key)
    return None


def _repair_base_selection_enabled(repair_plan: Mapping[str, Any] | None) -> bool:
    if not isinstance(repair_plan, Mapping):
        return False
    primary = str(repair_plan.get("primary_action") or "").strip().lower()
    if primary in {"regenerate", "none"}:
        return False
    if primary in {"recolor", "object_insertion", "relation_repair"}:
        return True
    sequence = repair_plan.get("tool_sequence", [])
    if isinstance(sequence, str):
        items: Sequence[Any] = [sequence]
    elif isinstance(sequence, Sequence):
        items = sequence
    else:
        items = []
    return any(
        str(item.get("action") if isinstance(item, Mapping) else item).strip().lower()
        in {"recolor", "object_insertion", "relation_repair"}
        for item in items
    )


def _route_all_count_failures_to_regenerate(
    repair_plan: Mapping[str, Any] | None,
    arbitration: Mapping[str, Any] | None,
    *,
    can_regenerate: bool,
) -> Mapping[str, Any] | None:
    if not isinstance(arbitration, Mapping):
        return repair_plan
    ranking = arbitration.get("ranking", [])
    if not isinstance(ranking, Sequence) or isinstance(ranking, (str, bytes)) or not ranking:
        return repair_plan
    tiers: list[str] = []
    for item in ranking:
        if not isinstance(item, Mapping):
            continue
        summary = item.get("constraint_summary", {})
        if isinstance(summary, Mapping):
            tiers.append(str(summary.get("human_rule_tier") or ""))
    if not tiers or any(tier != "reject_missing_or_wrong_count" for tier in tiers):
        return repair_plan
    previous = deepcopy(dict(repair_plan)) if isinstance(repair_plan, Mapping) else {}
    previous_preconditions = (
        dict(previous.get("preconditions", {}))
        if isinstance(previous.get("preconditions"), Mapping)
        else {}
    )
    return {
        **previous,
        "primary_action": "regenerate" if can_regenerate else "none",
        "tool_sequence": ["regenerate"] if can_regenerate else [],
        "repairable": False,
        "target_attribute": previous.get("target_attribute") or "count",
        "source": "m6212_count_failure_route",
        "override_from": previous.get("primary_action"),
        "reason": (
            "All candidates failed required object/count constraints. Disable local "
            "repair because count/missing-entity failures need regeneration or a "
            "count-focused prompt/layout retry."
        ),
        "preconditions": {
            **previous_preconditions,
            "all_candidates_failed_count_or_required_object": True,
        },
    }


def _failed_critique(
    prompt: str,
    image_path: str,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "score": 0.5,
        "errors": [],
        "strengths": [],
        "revision_hint": "",
        "user_grounded": True,
        "failed": True,
        "error": str(exc),
        "prompt": prompt,
        "image_path": image_path,
        "warnings": ["visual critique failed; score is a neutral fallback"],
    }


def _failed_evaluation(
    user_prompt: str,
    prompt: str,
    image_path: str,
    exc: Exception,
    *,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "evaluator": "vlm_judge",
        "failed": True,
        "error": str(exc),
        "score": None,
        "passed": None,
        "criteria_scores": {},
        "errors": [],
        "strengths": [],
        "revision_hint": "",
        "user_prompt": user_prompt,
        "prompt": prompt,
        "image_path": image_path,
        "context": deepcopy(dict(context)),
    }


def _hard_pass_guard(
    critique: Mapping[str, Any],
    *,
    constraints: PromptConstraints | None = None,
) -> dict[str, Any] | None:
    """Return a guard when narrow hard VQA has already verified the image."""

    if constraints is not None and prompt_needs_lexical_preflight(constraints):
        return None
    constraint_check = critique.get("constraint_check")
    if not _question_level_constraints_passed(constraint_check):
        return None
    if _hard_pass_blocked_by_evaluator_failure(critique):
        return None
    score = _coerce_float(
        _nested_mapping_value(critique, "constraint_check", "score"),
        default=_completion_score(critique),
    )
    return {
        "reason": "question_level_hard_constraints_passed",
        "score": score,
        "protected_from": [
            "broad_evaluator_soft_failures",
            "repair_planner",
            "local_repair",
            "specialist_gate",
        ],
    }


def _hard_pass_blocked_by_evaluator_failure(critique: Mapping[str, Any]) -> bool:
    evaluation = critique.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return False
    if evaluation.get("failed") is True or evaluation.get("passed") is not False:
        return False
    errors = _filter_positive_evaluation_errors(evaluation.get("errors", []) or [])
    if not errors:
        return False
    return any(_blocks_question_level_hard_pass(item) for item in errors)


def _blocks_question_level_hard_pass(error: Mapping[str, Any]) -> bool:
    text = _evaluation_error_text(error)
    error_type = _semantic_evaluation_error_type(error, text)
    if error_type in {
        "missing_object",
        "wrong_count",
        "wrong_symbol_text",
        "forbidden_object_present",
    }:
        return True
    if error_type != "wrong_relation":
        return False
    return _looks_like_relation_failure_requiring_repair(text)


def _looks_like_relation_failure_requiring_repair(text: str) -> bool:
    text = str(text or "").lower()
    if not text:
        return False
    repair_terms = (
        "occlud",
        "hide",
        "hides",
        "hidden",
        "cover",
        "covered",
        "screen",
        "not gripping",
        "not holding",
        "detached",
        "not touching",
        "not attached",
    )
    return any(term in text for term in repair_terms)


def _apply_hard_pass_guard(
    critique: Mapping[str, Any],
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(critique))
    result["hard_pass_guard"] = deepcopy(dict(guard))
    result["score"] = max(
        _coerce_float(result.get("score"), default=0.0),
        _coerce_float(guard.get("score"), default=0.0),
    )
    errors = _list_mapping_records(result.get("errors"))
    if errors:
        soft = _list_mapping_records(result.get("soft_evaluation_errors"))
        result["soft_evaluation_errors"] = _dedupe_prompt_errors([*soft, *errors])
    result["errors"] = []
    result["user_grounded"] = False
    result["revision_hint"] = "All hard user-grounded VQA constraints passed."
    return result


def _completion_blockers(
    critique: Mapping[str, Any],
    *,
    object_repair: Mapping[str, Any] | None = None,
    local_repair: Mapping[str, Any] | None = None,
    relation_repair: Mapping[str, Any] | None = None,
    score_threshold: float = 0.85,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    constraint_check = critique.get("constraint_check")
    if isinstance(constraint_check, Mapping):
        if _constraint_check_unavailable(constraint_check):
            blockers.append(
                {
                    "type": "constraint_check_unavailable",
                    "score": constraint_check.get("score"),
                    "source": constraint_check.get("source"),
                    "error": constraint_check.get("error")
                    or constraint_check.get("fallback_error")
                    or "",
                    "message": (
                        "original user constraints could not be verified; "
                        "completion must fail closed"
                    ),
                }
            )
        elif constraint_check.get("passed") is not True:
            blockers.append(
                {
                    "type": "constraint_check_failed",
                    "score": constraint_check.get("score"),
                    "error_count": len(constraint_check.get("errors", []) or []),
                    "message": "original user constraints are not satisfied",
                }
            )
        elif _constraint_check_score_below_threshold(constraint_check, score_threshold):
            blockers.append(
                {
                    "type": "constraint_score_below_threshold",
                    "score": constraint_check.get("score"),
                    "score_threshold": score_threshold,
                    "message": (
                        "question-level hard constraints returned no explicit "
                        "errors but the aggregate hard score is below threshold"
                    ),
                }
            )
    evaluation = critique.get("evaluation")
    if isinstance(evaluation, Mapping) and evaluation.get("failed") is not True:
        if (
            evaluation.get("passed") is False
            and not _soft_evaluation_only(critique)
        ):
            blockers.append(
                {
                    "type": "evaluation_failed",
                    "score": evaluation.get("score"),
                    "error_count": len(evaluation.get("errors", []) or []),
                    "message": "M6 evaluator did not pass the selected image",
                }
            )
    for error in _hard_user_errors(critique):
        blockers.append(
            {
                "type": "user_grounded_error",
                "error_type": error["type"],
                "evidence": error.get("evidence", ""),
                "prompt_span": error.get("prompt_span", ""),
            }
        )
    if object_repair and object_repair.get("accepted") is False:
        acceptance = object_repair.get("acceptance")
        blockers.append(
            {
                "type": "object_insertion_repair_rejected",
                "target_object": object_repair.get("repair_plan", {}).get("target_object")
                if isinstance(object_repair.get("repair_plan"), Mapping)
                else None,
                "post_repair_constraint_failures": (
                    acceptance.get("post_repair_constraint_failures", [])
                    if isinstance(acceptance, Mapping)
                    else []
                ),
                "hard_gate_failures": (
                    acceptance.get("hard_gate_failures", [])
                    if isinstance(acceptance, Mapping)
                    else []
                ),
                "message": "object insertion repair was attempted but rejected",
            }
        )
    if local_repair and local_repair.get("accepted") is False:
        acceptance = local_repair.get("acceptance")
        blockers.append(
            {
                "type": "local_repair_rejected",
                "hard_gate_failures": (
                    acceptance.get("hard_gate_failures", [])
                    if isinstance(acceptance, Mapping)
                    else []
                ),
                "message": "local recolor repair was attempted but rejected",
            }
        )
    if relation_repair and relation_repair.get("accepted") is False:
        blockers.append(
            {
                "type": "relation_repair_rejected",
                "score": relation_repair.get("score"),
                "message": "relation/action repair was attempted but rejected",
            }
        )
    return _dedupe_completion_blockers(blockers)


def _completion_score(critique: Mapping[str, Any]) -> float:
    """Use verified hard-constraint/evaluator scores when they are stronger.

    A VLM free-form critique can fail because of a transient API/payload problem.
    If the independent question-level hard-constraint check and optional M6
    evaluator both produced usable passing evidence, the neutral 0.5 critique
    fallback should not be the only completion score. Likewise, soft evaluator
    notes should not keep an image below threshold after the hard user
    constraints have passed.
    """

    base_score = _coerce_float(critique.get("score"), default=0.0)
    verified_scores: list[float] = []
    constraint_check = critique.get("constraint_check")
    if (
        isinstance(constraint_check, Mapping)
        and not _constraint_check_unavailable(constraint_check)
        and constraint_check.get("passed") is True
        and _constraint_check_score_below_threshold(constraint_check, 0.85)
    ):
        hard_score = _coerce_float(
            constraint_check.get("score", constraint_check.get("constraint_score")),
            default=0.0,
        )
        return min(base_score, hard_score)
    if (
        isinstance(constraint_check, Mapping)
        and not _constraint_check_unavailable(constraint_check)
        and _question_level_constraints_passed(constraint_check)
    ):
        verified_scores.append(
            _coerce_float(constraint_check.get("score"), default=base_score)
        )
    evaluation = critique.get("evaluation")
    if (
        isinstance(evaluation, Mapping)
        and evaluation.get("failed") is not True
        and evaluation.get("passed") is True
    ):
        verified_scores.append(
            _coerce_float(evaluation.get("score"), default=base_score)
        )
    if critique.get("failed") is not True and not _question_level_constraints_passed(
        constraint_check
    ):
        return base_score
    if not verified_scores:
        return base_score
    return max(base_score, min(verified_scores))


def _constraint_check_unavailable(value: Mapping[str, Any]) -> bool:
    """Return true when a hard constraint check failed to produce evidence."""

    if value.get("failed") is True:
        return True
    if value.get("source") == "constraint_question_failed":
        return True
    if value.get("passed") is None:
        return True
    summary = value.get("question_summary")
    if isinstance(summary, Mapping) and summary.get("source") == "constraint_question_failed":
        return True
    return False


def _constraint_check_score_below_threshold(
    value: Mapping[str, Any],
    score_threshold: float,
) -> bool:
    if value.get("passed") is not True:
        return False
    if _constraint_check_unavailable(value):
        return False
    score = _coerce_float(
        value.get("score", value.get("constraint_score")),
        default=1.0,
    )
    return score < float(score_threshold)


def _repair_plan_allows(
    repair_plan: Mapping[str, Any] | None,
    action: str,
) -> bool:
    if not repair_plan:
        return True
    primary = str(repair_plan.get("primary_action", "")).strip().lower()
    sequence = repair_plan.get("tool_sequence", [])
    if isinstance(sequence, str):
        sequence_items = [sequence]
    elif isinstance(sequence, Sequence):
        sequence_items = list(sequence)
    else:
        sequence_items = []
    normalized_sequence = {
        str(item.get("action") if isinstance(item, Mapping) else item).strip().lower()
        for item in sequence_items
    }
    return primary == action or action in normalized_sequence


def _merge_localized_repair_hint(
    plan: Mapping[str, Any],
    critique: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(plan))
    hint = _first_localized_repair_hint(critique)
    if not hint:
        return result
    for key in (
        "typed_route",
        "repair_kind",
        "bbox",
        "bbox_confidence",
        "target_object",
        "target_attribute",
        "text",
        "exact_text",
        "symbol",
        "fill_color",
        "text_color",
        "prompt_patch",
        "editability",
    ):
        if key not in result or result.get(key) in (None, "", [], {}):
            if key in hint:
                result[key] = deepcopy(hint[key])
    if result.get("primary_action") in (None, "", "none") and hint.get("repair_kind"):
        result["primary_action"] = "efficient_repair"
    if not result.get("tool_sequence") and hint.get("repair_kind"):
        result["tool_sequence"] = [hint["repair_kind"]]
    if not result.get("reason") and hint.get("repair_instruction"):
        result["reason"] = hint["repair_instruction"]
    result["localized_repair_hint"] = deepcopy(hint)
    return result


def _first_localized_repair_hint(critique: Mapping[str, Any]) -> dict[str, Any] | None:
    for source_key in ("constraint_check", "evaluation"):
        source = critique.get(source_key)
        if not isinstance(source, Mapping):
            continue
        plan = source.get("repair_plan")
        if isinstance(plan, Mapping) and plan:
            normalized = deepcopy(dict(plan))
            if "repair_kind" not in normalized and normalized.get("typed_route"):
                normalized["repair_kind"] = normalized.get("typed_route")
            return normalized
        localized = source.get("localized_errors")
        if isinstance(localized, Sequence) and not isinstance(localized, (str, bytes)):
            for item in localized:
                if not isinstance(item, Mapping):
                    continue
                hint = deepcopy(dict(item))
                if "repair_kind" not in hint and hint.get("typed_route"):
                    hint["repair_kind"] = hint.get("typed_route")
                return hint
    localized = critique.get("localized_errors")
    if isinstance(localized, Sequence) and not isinstance(localized, (str, bytes)):
        for item in localized:
            if isinstance(item, Mapping):
                return deepcopy(dict(item))
    return None


def _hard_user_errors(critique: Mapping[str, Any]) -> list[dict[str, str]]:
    hard_types = {
        "wrong_attribute",
        "wrong_count",
        "wrong_relation",
        "missing_object",
        "forbidden_object_present",
    }
    errors = critique.get("errors", [])
    if isinstance(errors, Mapping):
        errors = [errors]
    if not isinstance(errors, list):
        return []
    grounded = bool(critique.get("user_grounded"))
    nested_failures = _has_failed_user_constraint(critique.get("constraint_check")) or (
        _has_failed_evaluation(critique.get("evaluation"))
    )
    if not grounded and not nested_failures:
        return []
    if _question_level_constraints_passed(critique.get("constraint_check")) and not any(
        _semantic_evaluation_error_type(
            item if isinstance(item, Mapping) else {"evidence": str(item)},
        )
        in {
            "wrong_count",
            "wrong_symbol_text",
            "wrong_relation",
            "missing_object",
            "forbidden_object_present",
        }
        for item in errors
    ):
        return []
    hard_errors: list[dict[str, str]] = []
    for item in errors:
        if not isinstance(item, Mapping):
            item = {"type": "wrong_attribute", "evidence": str(item), "prompt_span": ""}
        error_type = _semantic_evaluation_error_type(item)
        if error_type not in hard_types:
            continue
        hard_errors.append(
            {
                "type": error_type,
                "evidence": str(item.get("evidence") or item.get("description") or ""),
                "prompt_span": str(item.get("prompt_span") or ""),
            }
        )
    return _dedupe_prompt_errors(hard_errors)


def _has_failed_user_constraint(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("failed") is not True and value.get("passed") is False


def _has_failed_evaluation(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("failed") is not True and value.get("passed") is False


def _soft_evaluation_only(critique: Mapping[str, Any]) -> bool:
    if not _question_level_constraints_passed(critique.get("constraint_check")):
        return False
    errors = critique.get("errors", [])
    if isinstance(errors, Mapping):
        errors = [errors]
    return not bool(errors)


def _dedupe_completion_blockers(blockers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for blocker in blockers:
        item = dict(blocker)
        key = (
            str(item.get("type", "")),
            str(item.get("error_type", "")),
            str(item.get("evidence", "")),
            str(item.get("prompt_span", "")),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _run_payload(
    run_id: str,
    status: str,
    mode: str,
    config: AgentConfig,
    constraints: PromptConstraints,
    state: AgentState,
    round_records: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    final_report_path: str | Path,
    layout_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "run_id": run_id,
        "status": status,
        "mode": mode,
        "config": config.to_dict(),
        "prompt_constraints": constraints.to_dict(),
        "state": state.to_dict(),
        "round_records": [deepcopy(dict(item)) for item in round_records],
        "events": [deepcopy(dict(item)) for item in events],
        "final_report_path": str(final_report_path),
    }
    if layout_context:
        payload["layout"] = deepcopy(dict(layout_context))
    final_selection = _best_final_selection(round_records)
    if final_selection:
        payload["final_selection"] = final_selection
    return payload


def _configure_mock_placeholder_dir(image_generator: ImageGenerator, image_dir: Path) -> None:
    children = getattr(image_generator, "children", None)
    if isinstance(children, Mapping):
        for label, child in children.items():
            _configure_mock_placeholder_dir(child, image_dir / str(label))
    elif isinstance(children, Sequence) and not isinstance(children, (str, bytes)):
        for index, child in enumerate(children):
            _configure_mock_placeholder_dir(child, image_dir / f"backend_{index}")
    if hasattr(image_generator, "placeholder_dir"):
        setattr(image_generator, "placeholder_dir", image_dir)
    if hasattr(image_generator, "create_placeholders"):
        setattr(image_generator, "create_placeholders", True)
    if hasattr(image_generator, "output_dir"):
        setattr(image_generator, "output_dir", Path(image_dir).resolve())


def _prioritize_user_grounded_errors(
    prompt: str,
    errors: Sequence[Mapping[str, Any]],
    constraints: PromptConstraints,
) -> list[dict[str, Any]]:
    user_errors = [
        {
            "original_prompt": prompt,
            "failed_sentence": str(item.get("prompt_span") or prompt),
            "error": str(item.get("evidence", "")),
            "error_type": _map_constraint_error_type(str(item.get("type", ""))),
            "source": "user_constraints",
        }
        for item in constraint_violations(prompt, constraints)
    ]
    normalized = [deepcopy(dict(item)) for item in errors]
    if user_errors:
        for item in normalized:
            item["source"] = f"secondary_{item.get('source', 'visual_reflector')}"
        return user_errors + normalized

    grounded: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []
    for item in normalized:
        evidence = str(item.get("error") or item.get("evidence", "")).lower()
        if any(
            phrase.lower() in evidence
            for phrase in constraints.protected_phrases + constraints.actions + constraints.relations
        ):
            grounded.append(item)
        else:
            secondary.append(item)
    return grounded + secondary


def _map_constraint_error_type(value: str) -> str:
    if "color" in value or "attribute" in value:
        return "wrong_attribute"
    if "action" in value or "relation" in value:
        return "wrong_relation"
    return "wrong_attribute"


def _merge_constraint_check(
    critique: Mapping[str, Any],
    constraint_check: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(critique))
    check = deepcopy(dict(constraint_check))
    result["constraint_check"] = _strip_large_prompt(check)
    if not bool(check.get("passed", True)):
        result["score"] = min(
            _coerce_float(result.get("score"), default=0.0),
            _coerce_float(check.get("score"), default=0.0),
        )
        result["errors"] = _dedupe_prompt_errors(
            [
                *list(check.get("errors", []) or []),
                *list(result.get("errors", []) or []),
            ]
        )
        check_hint = str(check.get("revision_hint", "")).strip()
        current_hint = str(result.get("revision_hint", "")).strip()
        if check_hint and current_hint and check_hint != current_hint:
            result["revision_hint"] = f"{check_hint} {current_hint}"
        elif check_hint:
            result["revision_hint"] = check_hint
        result["user_grounded"] = True
    elif _question_level_constraints_passed(check):
        _downgrade_top_level_errors_contradicting_passed_questions(result, check)
    return result


def _downgrade_top_level_errors_contradicting_passed_questions(
    result: dict[str, Any],
    constraint_check: Mapping[str, Any],
) -> None:
    errors = _filter_positive_evaluation_errors(result.get("errors", []) or [])
    if not errors:
        return
    hard_errors, soft_errors = _partition_evaluation_errors(
        errors,
        constraint_check,
    )
    if not soft_errors:
        return
    existing_soft = _list_mapping_records(result.get("soft_evaluation_errors"))
    result["soft_evaluation_errors"] = _strip_large_prompt(
        {"errors": _dedupe_prompt_errors([*existing_soft, *soft_errors])}
    )["errors"]
    existing_disagreements = _list_mapping_records(result.get("judge_disagreements"))
    disagreements = [
        {
            **dict(item),
            "source": "visual_critic_vs_question_level_vqa",
            "resolution": "question_level_hard_constraints_passed",
        }
        for item in soft_errors
    ]
    result["judge_disagreements"] = _strip_large_prompt(
        {"errors": _dedupe_judge_disagreements([*existing_disagreements, *disagreements])}
    )["errors"]
    result["errors"] = _dedupe_prompt_errors(hard_errors)
    if not hard_errors:
        result["score"] = max(
            _coerce_float(result.get("score"), default=0.0),
            _coerce_float(constraint_check.get("score"), default=0.0),
        )
        result["user_grounded"] = False


def _dedupe_judge_disagreements(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for record in records:
        item = deepcopy(dict(record))
        key = (
            str(item.get("type", "")),
            str(item.get("prompt_span", "")),
            str(item.get("evidence", item.get("description", ""))),
            str(item.get("source", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _merge_evaluation(
    critique: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    if not evaluation:
        return deepcopy(dict(critique))
    result = deepcopy(dict(critique))
    eval_record = deepcopy(dict(evaluation))
    result["evaluation"] = _strip_large_prompt(eval_record)
    if eval_record.get("failed"):
        result.setdefault("warnings", [])
        if isinstance(result["warnings"], list):
            result["warnings"].append(
                f"evaluation failed: {eval_record.get('error', 'unknown error')}"
            )
        return result
    eval_errors = _filter_positive_evaluation_errors(eval_record.get("errors", []) or [])
    if _question_level_constraints_passed(result.get("constraint_check")):
        hard_eval_errors, soft_eval_errors = _partition_evaluation_errors(
            eval_errors,
            result.get("constraint_check"),
        )
        if soft_eval_errors:
            result["soft_evaluation_errors"] = _strip_large_prompt(
                {"errors": soft_eval_errors}
            )["errors"]
        eval_errors = hard_eval_errors
    if eval_errors or not _question_level_constraints_passed(result.get("constraint_check")):
        result["score"] = min(
            _coerce_float(result.get("score"), default=0.0),
            _coerce_float(eval_record.get("score"), default=0.0),
        )
    result["errors"] = _dedupe_prompt_errors(
        [*list(eval_errors), *list(result.get("errors", []) or [])]
    )
    eval_hint = str(eval_record.get("revision_hint", "")).strip()
    current_hint = str(result.get("revision_hint", "")).strip()
    if eval_hint and current_hint and eval_hint != current_hint:
        result["revision_hint"] = f"{eval_hint} {current_hint}"
    elif eval_hint:
        result["revision_hint"] = eval_hint
    if not bool(eval_record.get("passed", True)) and eval_errors:
        result["user_grounded"] = True
    return result


def _merge_specialist_report(
    critique: Mapping[str, Any],
    specialist_report: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(critique))
    report = _strip_large_prompt(specialist_report)
    result["specialist_report"] = report
    arbitration = report.get("arbitration")
    if not isinstance(arbitration, Mapping):
        return result
    if arbitration.get("global_passed") is True or not _specialist_report_should_gate(report):
        return result
    errors = _specialist_prompt_errors(report)
    if errors:
        result["errors"] = _dedupe_prompt_errors(
            [*errors, *list(result.get("errors", []) or [])]
        )
    prompt_patch = str(arbitration.get("prompt_patch") or "").strip()
    current_hint = str(result.get("revision_hint") or "").strip()
    if prompt_patch and current_hint and prompt_patch not in current_hint:
        result["revision_hint"] = f"{prompt_patch}. {current_hint}"
    elif prompt_patch:
        result["revision_hint"] = prompt_patch
    result["score"] = min(_coerce_float(result.get("score"), default=0.0), 0.5)
    result["user_grounded"] = True
    result.setdefault("warnings", [])
    if isinstance(result["warnings"], list):
        forbidden = arbitration.get("forbidden_phrases")
        result["warnings"].append(
            "specialist arbiter blocked completion"
            + (f"; forbidden={forbidden}" if forbidden else "")
        )
    return result


def _specialist_prompt_errors(
    specialist_report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    arbitration = specialist_report.get("arbitration")
    if not isinstance(arbitration, Mapping):
        return []
    prompt_patch = str(arbitration.get("prompt_patch") or "").strip()
    dominant = str(arbitration.get("dominant_failure") or "").strip()
    error_type = _specialist_error_type(dominant)
    errors: list[dict[str, Any]] = []
    for report in _list_mapping_records(specialist_report.get("reports")):
        for failure in [
            *_list_mapping_records(report.get("failures")),
            *_list_mapping_records(report.get("uncertain")),
        ]:
            evidence = str(failure.get("evidence") or prompt_patch).strip()
            target = str(failure.get("target") or failure.get("prompt_span") or "").strip()
            errors.append(
                {
                    "type": _specialist_error_type(str(failure.get("type") or dominant)),
                    "evidence": evidence,
                    "prompt_span": target,
                    "source": report.get("agent") or "specialist_agent",
                }
            )
    if not errors and prompt_patch:
        errors.append(
            {
                "type": error_type,
                "evidence": prompt_patch,
                "prompt_span": "",
                "source": "ConstraintFusionArbiter",
            }
        )
    return errors


def _specialist_error_type(value: str) -> str:
    text = str(value or "").lower()
    if "spatial" in text or "layout" in text:
        return "wrong_spatial_relation"
    if "material" in text:
        return "wrong_material"
    if "object_type" in text or "object type" in text:
        return "wrong_object_type"
    if "count" in text:
        return "wrong_count"
    if "symbol" in text or "text" in text:
        return "wrong_symbol_text"
    if "interaction" in text or "relation" in text:
        return "wrong_relation"
    if "attribute" in text or "color" in text:
        return "wrong_attribute"
    if "subject" in text or "missing" in text or "object" in text:
        return "missing_object"
    return normalize_error_type(text)


def _specialist_report_should_gate(
    specialist_report: Mapping[str, Any],
) -> bool:
    arbitration = specialist_report.get("arbitration")
    if not isinstance(arbitration, Mapping):
        return False
    forbidden = arbitration.get("forbidden_phrases")
    if isinstance(forbidden, Sequence) and not isinstance(forbidden, (str, bytes)):
        if any(str(item).strip() for item in forbidden):
            return True
    for report in _list_mapping_records(specialist_report.get("reports")):
        for failure in [
            *_list_mapping_records(report.get("failures")),
            *_list_mapping_records(report.get("uncertain")),
        ]:
            evidence = str(failure.get("evidence") or "").lower()
            if "prompt drift contradicts" in evidence:
                return True
    return False


def _specialist_patch_from_critique(
    critique: Mapping[str, Any],
) -> dict[str, Any] | None:
    report = critique.get("specialist_report")
    if not isinstance(report, Mapping):
        return None
    arbitration = report.get("arbitration")
    if not isinstance(arbitration, Mapping):
        return None
    if arbitration.get("global_passed") is True:
        return None
    if not _specialist_report_should_gate(report):
        return None
    prompt_patch = str(arbitration.get("prompt_patch") or "").strip()
    if not prompt_patch:
        return None
    forbidden = [
        str(item).strip()
        for item in arbitration.get("forbidden_phrases", []) or []
        if str(item).strip()
    ]
    return {
        "prompt_patch": prompt_patch,
        "forbidden_phrases": forbidden,
        "dominant_failure": str(arbitration.get("dominant_failure") or "unknown"),
        "selected_action": str(arbitration.get("selected_action") or "regenerate"),
    }


def _apply_specialist_prompt_patch(
    prompt: str,
    specialist_patch: Mapping[str, Any],
    constraints: PromptConstraints,
) -> str:
    revised = str(prompt or "").strip()
    for phrase in specialist_patch.get("forbidden_phrases", []) or []:
        revised = _remove_forbidden_phrase(revised, str(phrase))
    patch_text = str(specialist_patch.get("prompt_patch") or "").strip()
    if patch_text and patch_text.lower() not in revised.lower():
        revised = f"{patch_text}, {revised}" if revised else patch_text
    if constraints.intent_spec is not None:
        for relation in constraints.intent_spec.interaction_relations:
            expected = str(relation.get("phrase") or "").strip()
            if expected and expected.lower() not in revised.lower():
                revised = f"{expected}, {revised}"
    return _clean_prompt_commas(revised)


def _guard_prompt_relation_drift(
    prompt: str,
    constraints: PromptConstraints,
    round_index: int,
    *,
    strategy: str,
) -> tuple[str, dict[str, Any] | None]:
    conflicts = _prompt_relation_conflicts(prompt, constraints)
    if not conflicts:
        return prompt, None
    revised = str(prompt or "").strip()
    removed: list[str] = []
    restored: list[str] = []
    for conflict in conflicts:
        phrase = conflict.get("forbidden_phrase", "")
        if phrase:
            revised = _remove_forbidden_phrase(revised, phrase)
            removed.append(phrase)
        expected = str(conflict.get("expected") or "").strip()
        if expected and expected.lower() not in revised.lower():
            revised = f"{expected}, {revised}" if revised else expected
            restored.append(expected)
    revised = _clean_prompt_commas(revised)
    return revised, {
        "type": "prompt_relation_drift_guarded",
        "round": round_index,
        "strategy": strategy,
        "original_prompt": prompt,
        "prompt": revised,
        "removed": _dedupe_strings(removed),
        "restored": _dedupe_strings(restored),
        "conflicts": conflicts,
    }


def _prompt_relation_conflicts(
    prompt: str,
    constraints: PromptConstraints,
) -> list[dict[str, str]]:
    if constraints.intent_spec is None:
        return []
    lowered = str(prompt or "").lower()
    conflicts: list[dict[str, str]] = []
    for relation in constraints.intent_spec.interaction_relations:
        action = str(relation.get("action") or "").strip().lower()
        expected_subject = str(relation.get("subject") or "").strip()
        expected_object = str(relation.get("object") or "").strip()
        expected_phrase = str(relation.get("phrase") or "").strip()
        if not action or not expected_object:
            continue
        if _relation_action_norm(action) not in _PHYSICAL_RELATION_ACTIONS:
            continue
        for subject in constraints.subjects:
            if _loose_name_match(subject, expected_object):
                continue
            phrases = _action_target_phrases(action, subject)
            for phrase in phrases:
                if phrase not in lowered:
                    continue
                conflicts.append(
                    {
                        "forbidden_phrase": phrase,
                        "observed": phrase,
                        "expected": expected_phrase
                        or f"{expected_subject} {action} {expected_object}".strip(),
                        "subject": expected_subject,
                        "object": expected_object,
                        "wrong_object": subject,
                    }
                )
    return conflicts


def _action_target_phrases(action: str, target: str) -> list[str]:
    target = str(target or "").strip().lower()
    if not target:
        return []
    action_norm = _relation_action_norm(action)
    if action_norm == "hold":
        return [f"holds the {target}", f"holding the {target}"]
    if action_norm == "grip":
        return [f"gripping the {target}", f"grips the {target}"]
    if action_norm == "carry":
        return [f"carrying the {target}", f"carries the {target}"]
    if action_norm == "touch":
        return [f"touching the {target}", f"touches the {target}"]
    if action_norm == "wear":
        return [f"wearing the {target}", f"wears the {target}"]
    if action_norm == "ride":
        return [f"riding the {target}", f"rides the {target}"]
    if action_norm == "attach":
        return [f"attached to the {target}", f"attaches to the {target}"]
    return []


_PHYSICAL_RELATION_ACTIONS = {"hold", "grip", "carry", "touch", "wear", "ride", "attach"}


def _relation_action_norm(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if text in {"hold", "holds", "holding"}:
        return "hold"
    if text in {"grip", "grips", "gripping", "grasp", "grasps", "grasping"}:
        return "grip"
    if text in {"carry", "carries", "carrying"}:
        return "carry"
    if text in {"touch", "touches", "touching"}:
        return "touch"
    if text in {"wear", "wears", "wearing"}:
        return "wear"
    if text in {"ride", "rides", "riding"}:
        return "ride"
    if text in {"attach", "attaches", "attached", "attached_to", "attaching"}:
        return "attach"
    return text


def _remove_forbidden_phrase(prompt: str, phrase: str) -> str:
    if not phrase.strip():
        return prompt
    pattern = re.compile(re.escape(phrase.strip()), flags=re.I)
    revised = pattern.sub("", prompt)
    revised = re.sub(r"\bthe subject visibly\s*,", "", revised, flags=re.I)
    revised = re.sub(r"\s+,", ",", revised)
    revised = re.sub(r",\s*,+", ",", revised)
    return _clean_prompt_commas(revised)


def _clean_prompt_commas(prompt: str) -> str:
    prompt = re.sub(r"\s+", " ", str(prompt or "")).strip()
    prompt = re.sub(r"\s*,\s*", ", ", prompt)
    prompt = re.sub(r"(,\s*)+", ", ", prompt)
    return prompt.strip(" ,")


def _dedupe_strings(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value or "").strip()
        key = cleaned.lower()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result


def _observation_from_existing_feedback(
    critique: Mapping[str, Any],
    constraints: PromptConstraints,
) -> dict[str, Any]:
    records = _iter_feedback_records(critique)
    observation = _positive_observation_from_constraints(
        constraints,
        relation_pass_default=not records,
    )
    for record in records:
        _apply_feedback_record_to_observation(record, observation, constraints)
    return observation


def _positive_observation_from_constraints(
    constraints: PromptConstraints,
    *,
    relation_pass_default: bool = True,
) -> dict[str, Any]:
    subjects = [
        {
            "name": subject,
            "visible": True,
            "count": constraints.intent_spec.counts.get(subject, 1)
            if constraints.intent_spec is not None
            else 1,
            "confidence": 0.51,
            "evidence": "No specialist failure evidence found in existing feedback.",
        }
        for subject in constraints.subjects
    ]
    attributes = [
        {
            "object": object_name,
            "attribute": "color",
            "expected": color,
            "observed": color,
            "passed": True,
            "confidence": 0.51,
            "evidence": "No specialist failure evidence found in existing feedback.",
        }
        for object_name, color in constraints.colors.items()
    ]
    interaction_relations = []
    spatial_relations = []
    if constraints.intent_spec is not None:
        for relation in constraints.intent_spec.relations:
            spatial_relations.append(
                {
                    "subject": relation.get("subject", ""),
                    "phrase": relation.get("phrase", ""),
                    "object": relation.get("object", ""),
                    "passed": bool(relation_pass_default),
                    "confidence": 0.51,
                    "evidence": (
                        "No specialist failure evidence found in existing feedback."
                        if relation_pass_default
                        else "Spatial relation not explicitly verified in failing feedback."
                    ),
                }
            )
        for relation in constraints.intent_spec.interaction_relations:
            interaction_relations.append(
                {
                    "subject": relation.get("subject", ""),
                    "action": relation.get("action", ""),
                    "object": relation.get("object", ""),
                    "passed": bool(relation_pass_default),
                    "confidence": 0.51,
                    "evidence": (
                        "No specialist failure evidence found in existing feedback."
                        if relation_pass_default
                        else "Interaction relation not explicitly verified in failing feedback."
                    ),
                    "confused_with": None,
                }
            )
    negative_constraints = [
        {
            "constraint": item,
            "passed": True,
            "confidence": 0.51,
            "evidence": "No specialist failure evidence found in existing feedback.",
        }
        for item in (
            constraints.intent_spec.negative_constraints
            if constraints.intent_spec is not None
            else []
        )
    ]
    return {
        "subjects": subjects,
        "attributes": attributes,
        "spatial_relations": spatial_relations,
        "interaction_relations": interaction_relations,
        "negative_constraints": negative_constraints,
        "summary": {
            "global_passed": True,
            "dominant_failure": "none",
            "repair_hint": "",
        },
    }


def _iter_feedback_records(value: Any) -> list[dict[str, Any]]:
    return _iter_feedback_records_with_soft_filter(value, set())


def _iter_feedback_records_with_soft_filter(
    value: Any,
    soft_keys: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        local_soft_keys = set(soft_keys)
        for item in _list_mapping_records(value.get("soft_evaluation_errors")):
            local_soft_keys.add(_feedback_record_key(item))
        if _question_level_constraints_passed(value.get("constraint_check")):
            evaluation = value.get("evaluation")
            if isinstance(evaluation, Mapping):
                _, soft_eval_errors = _partition_evaluation_errors(
                    evaluation.get("errors", []) or [],
                    value.get("constraint_check"),
                )
                for item in soft_eval_errors:
                    local_soft_keys.add(_feedback_record_key(item))
                if soft_eval_errors and _record_keys_match_all(
                    evaluation.get("errors", []) or [],
                    soft_eval_errors,
                ):
                    local_soft_keys.add(_feedback_record_key(evaluation))
        if _record_has_direct_failure_evidence(value):
            key = _feedback_record_key(value)
            if not key or key not in local_soft_keys:
                records.append(deepcopy(dict(value)))
        for key in (
            "constraint_check",
            "evaluation",
            "relation_repair_verification",
            "specialist_report",
        ):
            nested = value.get(key)
            if isinstance(nested, Mapping):
                records.extend(_iter_feedback_records_with_soft_filter(nested, local_soft_keys))
        for key in ("errors", "checks"):
            for item in _list_mapping_records(value.get(key)):
                record_key = _feedback_record_key(item)
                if record_key and record_key in local_soft_keys:
                    continue
                records.append(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            records.extend(_iter_feedback_records_with_soft_filter(item, soft_keys))
    return records


def _record_keys_match_all(records: Any, soft_records: Sequence[Mapping[str, Any]]) -> bool:
    record_keys = {
        _feedback_record_key(item)
        for item in _list_mapping_records(records)
        if _feedback_record_key(item)
    }
    soft_keys = {
        _feedback_record_key(item)
        for item in soft_records
        if _feedback_record_key(item)
    }
    return bool(record_keys) and record_keys <= soft_keys


def _feedback_record_key(record: Mapping[str, Any]) -> str:
    error_type = normalize_error_type(record.get("type") or record.get("error_type") or "")
    text = _record_text(record).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return f"{error_type}:{text}" if text else ""


def _record_has_direct_failure_evidence(record: Mapping[str, Any]) -> bool:
    if record.get("passed") is True:
        return False
    if record.get("failed") is True:
        return True
    if record.get("passed") is False:
        return True
    if record.get("type") or record.get("error_type"):
        return bool(
            str(record.get("evidence") or record.get("description") or record.get("reason") or "").strip()
        )
    return False


def _apply_feedback_record_to_observation(
    record: Mapping[str, Any],
    observation: dict[str, Any],
    constraints: PromptConstraints,
) -> None:
    passed = record.get("passed")
    if passed is True:
        return
    text = _record_text(record)
    if not text:
        return
    error_type = normalize_error_type(record.get("type") or record.get("error_type") or "")
    evidence_text = _record_evidence_text(record)
    if error_type == "wrong_count" or _looks_like_count_failure(record, evidence_text):
        _mark_count_failure(observation, constraints, record, text)
        return
    if error_type == "missing_object" or _looks_like_missing_object(evidence_text, error_type):
        _mark_subject_failure(observation, constraints, text)
        return
    if error_type == "wrong_attribute" or _looks_like_attribute_failure(evidence_text, error_type):
        _mark_attribute_failure(observation, constraints, record, text)
        return
    if _looks_like_spatial_failure(evidence_text, error_type):
        _mark_spatial_failure(observation, constraints, text)
        return
    if error_type == "wrong_relation" or _looks_like_relation_failure(evidence_text, error_type):
        _mark_interaction_failure(observation, constraints, text)


def _record_text(record: Mapping[str, Any]) -> str:
    pieces = [
        str(record.get(key) or "")
        for key in (
            "evidence",
            "description",
            "reason",
            "message",
            "revision_hint",
            "prompt_span",
            "target",
            "target_object",
            "observed",
            "expected",
            "question_id",
        )
    ]
    return " ".join(piece for piece in pieces if piece).strip()


def _record_evidence_text(record: Mapping[str, Any]) -> str:
    pieces = [
        str(record.get(key) or "")
        for key in (
            "evidence",
            "description",
            "reason",
            "message",
            "revision_hint",
            "observed",
        )
    ]
    return " ".join(piece for piece in pieces if piece).strip()


def _looks_like_relation_failure(text: str, error_type: str) -> bool:
    lowered = text.lower()
    return error_type == "wrong_relation" or any(
        needle in lowered
        for needle in (
            "hold",
            "holding",
            "grip",
            "gripping",
            "touching",
            "attached",
            "relation",
            "handle instead",
        )
    )


def _looks_like_spatial_failure(text: str, error_type: str) -> bool:
    lowered = text.lower()
    return error_type in {"wrong_spatial_relation", "spatial_relation"} or any(
        needle in lowered
        for needle in (
            "left of",
            "right of",
            "above",
            "below",
            "under",
            "behind",
            "in front of",
            "on top of",
            "spatial",
            "position",
            "positioned",
            "layout",
        )
    )


def _looks_like_count_failure(record: Mapping[str, Any], text: str) -> bool:
    question_id = str(record.get("question_id") or "").lower()
    category = str(record.get("category") or record.get("type") or "").lower()
    lowered = text.lower()
    return (
        question_id.startswith("count:")
        or category in {"count", "wrong_count"}
        or "wrong_count" in category
        or "exactly" in lowered
        and any(token in lowered for token in ("observed", "visible", "extra", "fifth", "count"))
    )


def _looks_like_missing_object(text: str, error_type: str) -> bool:
    lowered = text.lower()
    return error_type == "missing_object" or any(
        needle in lowered
        for needle in ("missing", "absent", "not visible", "not present", "no visible")
    )


def _looks_like_attribute_failure(text: str, error_type: str) -> bool:
    lowered = text.lower()
    return error_type == "wrong_attribute" or any(
        needle in lowered for needle in ("color", "leakage", "bleed", "instead of")
    )


def _mark_subject_failure(
    observation: dict[str, Any],
    constraints: PromptConstraints,
    text: str,
) -> None:
    target = _matched_subject_from_text(text, constraints)
    if not target:
        return
    for item in observation["subjects"]:
        if _loose_name_match(target, item.get("name")):
            item["visible"] = False
            item["confidence"] = 0.9
            item["evidence"] = text
            return


def _mark_count_failure(
    observation: dict[str, Any],
    constraints: PromptConstraints,
    record: Mapping[str, Any],
    text: str,
) -> None:
    target = _matched_count_target(record, text, constraints)
    if not target:
        return
    observed_count = _count_value_from_record(record, text)
    for item in observation["subjects"]:
        if _loose_name_match(target, item.get("name")):
            item["visible"] = True
            if observed_count is not None:
                item["count"] = observed_count
            item["confidence"] = 0.9
            item["evidence"] = text
            return


def _matched_count_target(
    record: Mapping[str, Any],
    text: str,
    constraints: PromptConstraints,
) -> str:
    target = str(record.get("target") or record.get("prompt_span") or "").strip()
    question_id = str(record.get("question_id") or "").strip().lower()
    if not target and question_id.startswith("count:"):
        target = question_id.split(":", 1)[1]
    if target:
        for subject in constraints.subjects:
            if _loose_name_match(target, subject):
                return subject
    lowered = text.lower()
    for subject in sorted(constraints.subjects, key=len, reverse=True):
        if _loose_subject_or_head_mentioned(lowered, subject):
            return subject
    return ""


def _expected_count_for_subject(target: str, constraints: PromptConstraints) -> int | None:
    if constraints.intent_spec is None:
        return None
    for subject, count in constraints.intent_spec.counts.items():
        if _loose_name_match(target, subject):
            return int(count)
    return None


def _count_value_from_record(record: Mapping[str, Any], text: str) -> int | None:
    for key in ("observed", "observed_count", "count"):
        value = record.get(key)
        parsed = _parse_small_count(value)
        if parsed is not None:
            return parsed
    lowered = text.lower()
    for number, word in enumerate(
        ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
    ):
        if re.search(rf"\b(?:{number}|{word})\b", lowered):
            return number
    return None


def _parse_small_count(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
    }
    return words.get(text)


def _mark_attribute_failure(
    observation: dict[str, Any],
    constraints: PromptConstraints,
    record: Mapping[str, Any],
    text: str,
) -> None:
    target = _matched_color_target(record, text, constraints)
    if target:
        for item in observation["attributes"]:
            if _loose_name_match(target, item.get("object")):
                item["passed"] = False
                item["confidence"] = 0.9
                item["observed"] = str(record.get("observed") or "")
                item["evidence"] = text
                return
    if "leak" in text.lower() or "bleed" in text.lower():
        for item in observation["negative_constraints"]:
            item["passed"] = False
            item["confidence"] = 0.9
            item["evidence"] = text


def _mark_interaction_failure(
    observation: dict[str, Any],
    constraints: PromptConstraints,
    text: str,
) -> None:
    if constraints.intent_spec is None or not constraints.intent_spec.interaction_relations:
        return
    relation = constraints.intent_spec.interaction_relations[0]
    expected_subject = str(relation.get("subject") or "")
    expected_target = str(relation.get("object") or "")
    wrong_target = _confused_subject_from_text(
        text,
        constraints,
        expected_target,
        expected_subject=expected_subject,
    )
    for item in observation["interaction_relations"]:
        if not _loose_name_match(expected_target, item.get("object")):
            continue
        item["passed"] = False
        item["confidence"] = 0.9
        item["evidence"] = text
        item["confused_with"] = wrong_target or item.get("confused_with")
        return


def _mark_spatial_failure(
    observation: dict[str, Any],
    constraints: PromptConstraints,
    text: str,
) -> None:
    if constraints.intent_spec is None or not constraints.intent_spec.relations:
        return
    lowered = text.lower()
    matched = False
    for item in observation["spatial_relations"]:
        subject = str(item.get("subject") or "")
        phrase = str(item.get("phrase") or item.get("relation") or "")
        target = str(item.get("object") or "")
        if not phrase:
            continue
        if phrase.lower() not in lowered and not _spatial_relation_mentioned(lowered, phrase):
            continue
        if subject and not _loose_subject_or_head_mentioned(lowered, subject):
            if target and not _loose_subject_or_head_mentioned(lowered, target):
                continue
        item["passed"] = False
        item["confidence"] = 0.9
        item["evidence"] = text
        matched = True
    if matched:
        return
    for item in observation["spatial_relations"]:
        item["passed"] = False
        item["confidence"] = 0.9
        item["evidence"] = text


def _spatial_relation_mentioned(text: str, phrase: str) -> bool:
    phrase = phrase.lower().replace("_", " ").strip()
    aliases = {
        "right of": ("right", "to the right"),
        "left of": ("left", "to the left"),
        "under": ("under", "below", "beneath"),
        "above": ("above", "over"),
        "behind": ("behind",),
        "in front of": ("in front", "front of"),
        "next to": ("next to", "beside"),
        "on top of": ("on top", "top of"),
    }
    return any(alias in text for alias in aliases.get(phrase, (phrase,)))


def _loose_subject_or_head_mentioned(text: str, subject: str) -> bool:
    subject = str(subject or "").strip().lower()
    if not subject:
        return False
    if subject in text:
        return True
    head = subject.split()[-1]
    return len(head) > 2 and re.search(rf"\b{re.escape(head)}\b", text) is not None


def _matched_subject_from_text(text: str, constraints: PromptConstraints) -> str:
    lowered = text.lower()
    for subject in sorted(constraints.subjects, key=len, reverse=True):
        if subject.lower() in lowered:
            return subject
    return ""


def _matched_color_target(
    record: Mapping[str, Any],
    text: str,
    constraints: PromptConstraints,
) -> str:
    target = str(record.get("target") or record.get("prompt_span") or "").lower()
    for object_name in constraints.colors:
        if object_name.lower() in target:
            return object_name
    lowered = text.lower()
    color_supported = [
        object_name
        for object_name, color in constraints.colors.items()
        if object_name.lower() in lowered
        and color
        and re.search(rf"\b{re.escape(str(color).lower())}\b", lowered)
    ]
    if color_supported:
        return min(
            color_supported,
            key=lambda item: _object_color_mention_distance(
                lowered,
                item,
                str(constraints.colors[item]),
            ),
        )
    mentioned = [
        object_name
        for object_name in constraints.colors
        if object_name.lower() in lowered
    ]
    if mentioned:
        return min(mentioned, key=lambda item: lowered.find(item.lower()))
    return ""


def _object_color_mention_distance(text: str, object_name: str, color: str) -> tuple[int, int]:
    object_pos = text.find(object_name.lower())
    color_positions = [
        match.start()
        for match in re.finditer(rf"\b{re.escape(color.lower())}\b", text)
    ]
    if object_pos < 0:
        object_pos = 10**8
    if not color_positions:
        return (10**8, object_pos)
    return (min(abs(pos - object_pos) for pos in color_positions), object_pos)


def _confused_subject_from_text(
    text: str,
    constraints: PromptConstraints,
    expected_target: str,
    *,
    expected_subject: str = "",
) -> str:
    lowered = text.lower()
    expected_head = str(expected_target or "").split()[-1] if str(expected_target or "").split() else ""
    for subject in sorted(constraints.subjects, key=len, reverse=True):
        if _loose_name_match(subject, expected_target):
            continue
        if expected_subject and _loose_name_match(subject, expected_subject):
            continue
        if subject.lower() in lowered:
            if (
                expected_head
                and re.search(rf"\b{re.escape(expected_head.lower())}\b", lowered)
                and expected_head.lower() not in subject.lower()
            ):
                return f"{subject} {expected_head}"
            return subject
    return ""


def _loose_name_match(left: Any, right: Any) -> bool:
    left_norm = re.sub(r"[^a-z0-9]+", "_", str(left or "").strip().lower()).strip("_")
    right_norm = re.sub(r"[^a-z0-9]+", "_", str(right or "").strip().lower()).strip("_")
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True
    left_parts = left_norm.split("_")
    right_parts = right_norm.split("_")
    if len(left_parts) == 1:
        return left_norm in right_parts
    if len(right_parts) == 1:
        return right_norm in left_parts
    return left_norm.endswith(right_norm) or right_norm.endswith(left_norm)


def _question_level_prompt_errors(
    prompt: str,
    critique: Mapping[str, Any],
) -> list[dict[str, Any]]:
    constraint_check = critique.get("constraint_check")
    if not isinstance(constraint_check, Mapping):
        return []
    errors = _list_mapping_records(constraint_check.get("errors"))
    checks = _list_mapping_records(constraint_check.get("checks"))
    if not errors and not any(item.get("passed") is False for item in checks):
        return []
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in sorted(
        [*errors, *[check for check in checks if check.get("passed") is False]],
        key=_question_error_priority,
    ):
        error_type = _question_error_type(item)
        span = _question_failed_span(item)
        evidence = _question_error_evidence(item)
        key = (error_type, span.lower(), evidence.lower())
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "original_prompt": prompt,
                "failed_sentence": span or prompt,
                "error": evidence,
                "error_type": error_type,
                "source": "question_level_vqa",
            }
        )
    return records


def _question_error_priority(item: Mapping[str, Any]) -> tuple[int, str]:
    question_id = str(item.get("question_id") or "").lower()
    category = str(item.get("category") or item.get("type") or "").lower()
    error_type = normalize_error_type(item.get("type"))
    if question_id.startswith("existence:") or category in {"entity_existence", "subject"}:
        return (0, question_id)
    if question_id.startswith("count:") or error_type == "wrong_count" or category == "count":
        return (1, question_id)
    if question_id.startswith("color:") or category in {"color_binding", "color"}:
        return (2, question_id)
    if question_id.startswith("relation:") or question_id.startswith("action:"):
        return (3, question_id)
    if error_type == "wrong_relation" or category in {"action_relation", "spatial_relation"}:
        return (3, question_id)
    return (4, question_id)


def _question_error_type(item: Mapping[str, Any]) -> str:
    question_id = str(item.get("question_id") or "").lower()
    category = str(item.get("category") or item.get("type") or "").lower()
    if question_id.startswith("existence:") or category in {"entity_existence", "subject"}:
        return "missing_object"
    if question_id.startswith("count:") or category == "count":
        return "wrong_count"
    if question_id.startswith("relation:") or question_id.startswith("action:"):
        return "wrong_relation"
    if category in {"action_relation", "spatial_relation", "relation", "action"}:
        return "wrong_relation"
    return normalize_error_type(item.get("type"))


def _question_failed_span(item: Mapping[str, Any]) -> str:
    span = str(item.get("prompt_span") or "").strip()
    if span:
        return span
    target = str(item.get("target") or "").strip()
    expected = str(item.get("expected") or "").strip()
    if target and expected and expected.lower() not in {"yes", "no", "uncertain"}:
        return f"{expected} {target}".strip()
    return target


def _question_error_evidence(item: Mapping[str, Any]) -> str:
    question_id = str(item.get("question_id") or "").strip()
    expected = str(item.get("expected") or "").strip()
    observed = str(item.get("observed") or "").strip()
    description = str(item.get("evidence") or item.get("description") or "").strip()
    prefix = ""
    if question_id:
        if question_id.startswith("existence:"):
            prefix = "Required subject/object is missing or unclear."
        elif question_id.startswith("count:"):
            prefix = "Required count is not satisfied."
        elif question_id.startswith("color:"):
            prefix = "Required color binding is not satisfied."
        elif question_id.startswith(("relation:", "action:")):
            prefix = "Required action or relation is not satisfied."
    pieces = [piece for piece in (prefix, description) if piece]
    if expected and observed:
        pieces.append(f"Expected: {expected}; observed: {observed}.")
    return " ".join(pieces) or "Question-level hard constraint failed."


def _filter_positive_evaluation_errors(errors: Any) -> list[dict[str, Any]]:
    records = _list_mapping_records(errors)
    return [item for item in records if not _is_positive_evaluation_error(item)]


def _is_positive_evaluation_error(error: Mapping[str, Any]) -> bool:
    text = " ".join(
        str(error.get(key) or "").lower()
        for key in ("evidence", "description", "reason", "message")
    )
    if not text:
        return False
    positive_markers = (
        "matches",
        "match the prompt",
        "matches the prompt",
        "matches the original intent",
        "satisfies",
        "is correct",
        "are correct",
        "correctly",
        "as requested",
        "as specified",
        "no error",
        "not an error",
        "no issue",
        "no problem",
    )
    negative_relation_markers = (
        "not attached",
        "not connected",
        "not mounted",
        "not stuck",
        "not touching",
    )
    negative_markers = (
        "does not",
        "doesn't",
        "not as requested",
        "not as specified",
        "not visible",
        "no visible",
        "missing",
        "absent",
        "instead of",
        "rather than",
        "wrong",
        "mismatch",
        "fails",
        "failed",
    )
    has_negative_marker = any(marker in text for marker in negative_markers) or bool(
        re.search(r"\bnot\b.*\bas\s+(?:requested|specified)\b", text)
    ) or any(
        marker in text for marker in negative_relation_markers
    )
    return any(marker in text for marker in positive_markers) and not has_negative_marker


def _question_level_constraints_passed(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("source") != "question_level_vqa":
        return False
    if value.get("passed") is not True or value.get("errors"):
        return False
    score = _coerce_float(
        value.get("score", value.get("constraint_score")),
        default=1.0,
    )
    return score >= 0.85


def _partition_evaluation_errors(
    errors: Any,
    constraint_check: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = _list_mapping_records(errors)
    if not records:
        return [], []
    explicit_targets = _explicit_constraint_targets(constraint_check)
    hard: list[dict[str, Any]] = []
    soft: list[dict[str, Any]] = []
    for item in records:
        if _contradicts_passed_question_check(item, constraint_check):
            soft.append(item)
        elif _is_hard_evaluation_error(item, explicit_targets):
            hard.append(item)
        else:
            soft.append(item)
    return hard, soft


def _explicit_constraint_targets(constraint_check: Any) -> dict[str, set[str]]:
    targets = {
        "objects": set(),
        "colors": set(),
        "relations": set(),
        "passed_questions": set(),
    }
    if not isinstance(constraint_check, Mapping):
        return targets
    for item in _list_mapping_records(constraint_check.get("checks")):
        target = str(item.get("target") or "").strip().lower()
        expected = str(item.get("expected") or "").strip().lower()
        category = str(item.get("category") or item.get("type") or "").strip().lower()
        question_id = str(item.get("question_id") or "").strip().lower()
        passed = item.get("passed") is True
        if target:
            targets["objects"].add(target)
        if passed and target:
            targets["passed_questions"].add(target)
        if category in {"color_binding", "color"} and expected:
            targets["colors"].add(f"{target}:{expected}")
        if category in {"action_relation", "spatial_relation", "relation", "action"}:
            targets["relations"].add(target)
        if category == "negative_relation" or question_id.startswith("negative_relation:"):
            targets["relations"].add(target)
            if ":" in question_id:
                parts = [part for part in question_id.split(":")[1:] if part]
                if len(parts) >= 3:
                    targets["relations"].add(":".join(parts[:3]))
    return targets


def _contradicts_passed_question_check(
    error: Mapping[str, Any],
    constraint_check: Any,
) -> bool:
    if not isinstance(constraint_check, Mapping):
        return False
    error_text = _evaluation_error_text(error)
    error_type = _semantic_evaluation_error_type(error, error_text)
    if error_type not in {"wrong_attribute", "wrong_count", "wrong_relation"}:
        return False
    if not error_text:
        return False
    for check in _list_mapping_records(constraint_check.get("checks")):
        if check.get("passed") is not True:
            continue
        if not _same_constraint_family(error_type, check):
            continue
        if not _evaluation_error_matches_check_target(error_text, check):
            continue
        if error_type == "wrong_count" and _count_error_matches_passed_count(error_text, check):
            return True
        if error_type in {"wrong_attribute", "wrong_relation"}:
            return True
    return False


def _same_constraint_family(error_type: str, check: Mapping[str, Any]) -> bool:
    category = str(check.get("category") or check.get("type") or "").lower()
    question_id = str(check.get("question_id") or "").lower()
    if error_type == "wrong_count":
        return category == "count" or question_id.startswith("count:")
    if error_type == "wrong_attribute":
        return category in {"color_binding", "color"} or question_id.startswith("color:")
    if error_type == "wrong_relation":
        return category in {
            "action_relation",
            "spatial_relation",
            "relation",
            "action",
            "negative_relation",
        } or question_id.startswith(("relation:", "action:", "negative_relation:"))
    return False


def _evaluation_error_matches_check_target(
    error_text: str,
    check: Mapping[str, Any],
) -> bool:
    target = str(check.get("target") or "").strip().lower()
    expected = str(check.get("expected") or "").strip().lower()
    question_id = str(check.get("question_id") or "").strip().lower()
    terms = [target, expected]
    if ":" in question_id:
        for part in question_id.split(":")[1:]:
            if not part:
                continue
            terms.append(part)
            terms.append(part.replace("_", " "))
    return any(_target_matches_error_text(term, error_text) for term in terms)


def _count_error_matches_passed_count(error_text: str, check: Mapping[str, Any]) -> bool:
    if _looks_like_explicit_count_failure_text(error_text):
        return False
    expected = str(check.get("expected") or "").strip().lower()
    observed = str(check.get("observed") or "").strip().lower()
    values = {value for value in (expected, observed) if value}
    for value in list(values):
        if value.isdigit():
            values.add(_number_word(int(value)))
    return any(value and re.search(rf"\b{re.escape(value)}\b", error_text) for value in values)


def _number_word(value: int) -> str:
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    return words.get(value, str(value))


def _is_hard_evaluation_error(
    error: Mapping[str, Any],
    explicit_targets: Mapping[str, set[str]],
) -> bool:
    prompt_text = str(error.get("prompt_span") or "").lower()
    evidence_text = " ".join(
        str(error.get(key) or "").lower()
        for key in ("evidence", "description", "reason")
    )
    text = f"{prompt_text} {evidence_text}".strip()
    error_type = _semantic_evaluation_error_type(error, text)
    if error_type in {"style_mismatch", "artifact"}:
        return False
    if error_type in {"missing_object", "wrong_count", "forbidden_object_present"}:
        return True
    if error_type == "wrong_relation":
        return any(_target_matches_error_text(target, text) for target in explicit_targets.get("relations", set()))
    if error_type == "wrong_attribute":
        for item in explicit_targets.get("colors", set()):
            target, _, expected = item.partition(":")
            if (
                target
                and expected
                and _target_matches_error_text(target, text)
                and _explicit_attribute_contradiction(evidence_text, expected)
            ):
                return True
        return False
    return False


def _semantic_evaluation_error_type(
    error: Mapping[str, Any],
    text: str | None = None,
) -> str:
    """Infer hard evaluator error types from evidence, not only its label.

    VLM judges sometimes label count/layout/text failures as ``wrong_attribute``.
    Treating those as ordinary attribute disagreements lets a narrow color VQA
    pass incorrectly erase hard user-constraint failures.
    """

    normalized = normalize_error_type(error.get("type"))
    combined = (text if text is not None else _evaluation_error_text(error)).lower()
    if _looks_like_forbidden_object_present_text(combined):
        return "forbidden_object_present"
    if _looks_like_explicit_count_failure_text(combined):
        return "wrong_count"
    if _looks_like_symbol_text_failure_text(combined):
        return "wrong_symbol_text"
    if _looks_like_spatial_failure_text(combined):
        return "wrong_relation"
    return normalized


def _looks_like_forbidden_object_present_text(text: str) -> bool:
    text = str(text or "").lower()
    if not text:
        return False
    negative_terms = (
        "no ",
        "without ",
        "should not",
        "must not",
        "forbidden",
        "not supposed",
        "absence",
        "absent",
    )
    violation_terms = (
        "violat",
        "contains",
        "contain",
        "present",
        "visible",
        "appears",
        "shown",
        "depicted",
        "nearby",
        "extra",
    )
    if not any(term in text for term in negative_terms):
        return False
    return any(term in text for term in violation_terms)


def _looks_like_explicit_count_failure_text(text: str) -> bool:
    text = str(text or "").lower()
    if not text:
        return False
    patterns = (
        r"\bcontains?\s+(?:an?\s+|the\s+)?(?:extra|additional|too many)\b",
        r"\b(extra|additional|too many|duplicate|duplicated|repeated)\b",
        r"\bcontains?\s+(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
        r"\bvisible\s+(?:instead of|rather than)\b",
        r"\binstead\s+of\s+(?:the\s+)?(?:required|requested|specified)?\s*(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
        r"\b(?:not|isn't|are not|aren't)\s+(?:exactly\s+)?(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\b",
        r"\b(?:missing|absent)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def _looks_like_symbol_text_failure_text(text: str) -> bool:
    text = str(text or "").lower()
    if not text:
        return False
    return any(
        marker in text
        for marker in (
            "exact text",
            "text is",
            "text on",
            "should read",
            "reads",
            "letter",
            "symbol",
            "no text",
        )
    ) and any(
        marker in text
        for marker in (
            "wrong",
            "incorrect",
            "missing",
            "not",
            "should",
            "instead",
            "violat",
        )
    )


def _looks_like_spatial_failure_text(text: str) -> bool:
    text = str(text or "").lower()
    if not text:
        return False
    relation_terms = (
        "left of",
        "right of",
        "above",
        "below",
        "under",
        "behind",
        "in front of",
        "next to",
        "beside",
        "stacked",
    )
    failure_terms = ("wrong", "incorrect", "not ", "instead", "rather than", "violat")
    return any(term in text for term in relation_terms) and any(
        term in text for term in failure_terms
    )


def _evaluation_error_text(error: Mapping[str, Any]) -> str:
    return " ".join(
        str(error.get(key) or "").lower()
        for key in ("prompt_span", "evidence", "description", "reason", "message")
    ).strip()


def _explicit_attribute_contradiction(evidence_text: str, expected: str) -> bool:
    evidence_text = str(evidence_text or "").lower()
    expected = str(expected or "").strip().lower()
    if not evidence_text or not expected:
        return False
    if any(
        phrase in evidence_text
        for phrase in (
            "wrong color",
            "color mismatch",
            "attribute mismatch",
            "not the requested color",
            "not the specified color",
            "different from the requested color",
        )
    ):
        return True
    expected_pattern = re.escape(expected)
    patterns = (
        rf"\bnot\s+(?:clearly\s+|visibly\s+)?{expected_pattern}\b",
        rf"\binstead\s+of\s+{expected_pattern}\b",
        rf"\brather\s+than\s+{expected_pattern}\b",
        rf"\bshould\s+be\s+{expected_pattern}\b",
        rf"\bspecified\s+{expected_pattern}\b",
        rf"\brequested\s+{expected_pattern}\b",
        rf"\bexpected\s+{expected_pattern}\b",
    )
    return any(re.search(pattern, evidence_text) for pattern in patterns)


def _target_matches_error_text(target: str, text: str) -> bool:
    target = str(target or "").strip().lower()
    if not target:
        return False
    terms = [target, *target.split(":"), *target.split("-"), *target.split()]
    return any(term and len(term) > 2 and re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def _merge_repair_evaluation(
    critique: Mapping[str, Any],
    repair_evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(critique))
    result["pre_repair_feedback"] = _strip_large_prompt(result)
    eval_record = deepcopy(dict(repair_evaluation))
    result["evaluation"] = _strip_large_prompt(eval_record)
    if eval_record.get("failed"):
        result["repaired"] = False
        result.setdefault("warnings", [])
        if isinstance(result["warnings"], list):
            result["warnings"].append(
                f"repair evaluation failed: {eval_record.get('error', 'unknown error')}"
            )
        return result
    result["score"] = _coerce_float(eval_record.get("score"), default=0.0)
    result["errors"] = _dedupe_prompt_errors(list(eval_record.get("errors", []) or []))
    result["strengths"] = list(eval_record.get("strengths", []) or result.get("strengths", []))
    result["revision_hint"] = str(eval_record.get("revision_hint") or result.get("revision_hint") or "")
    result["repaired"] = True
    result["user_grounded"] = not bool(eval_record.get("passed", True))
    return result


def _merge_object_repair_check(
    critique: Mapping[str, Any],
    post_check: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(critique))
    check = deepcopy(dict(post_check))
    result["pre_object_repair_feedback"] = _strip_large_prompt(result)
    result["constraint_check"] = _strip_large_prompt(check)
    result["score"] = _coerce_float(check.get("score"), default=0.0)
    result["errors"] = _dedupe_prompt_errors(list(check.get("errors", []) or []))
    result["strengths"] = list(check.get("strengths", []) or result.get("strengths", []))
    result["revision_hint"] = str(check.get("revision_hint") or result.get("revision_hint") or "")
    result["object_repaired"] = True
    result["user_grounded"] = not bool(check.get("passed", True))
    return result


def _mark_accepted_local_edit(
    critique: Mapping[str, Any],
    repair: Mapping[str, Any],
) -> dict[str, Any]:
    """Clear stale pre-edit hard failures after an accepted post-checked edit."""

    result = deepcopy(dict(critique))
    result["accepted_local_edit"] = True
    result["accepted_local_edit_summary"] = {
        "route": repair.get("route"),
        "edited_image": repair.get("edited_image"),
        "repair_kind": repair.get("repair_kind"),
    }
    evaluation = result.get("evaluation")
    if isinstance(evaluation, Mapping) and evaluation.get("passed") is False:
        previous = deepcopy(dict(evaluation))
        score = max(_coerce_float(previous.get("score"), default=0.0), 1.0)
        result["pre_accepted_local_edit_evaluation"] = _strip_large_prompt(previous)
        result["evaluation"] = {
            **previous,
            "passed": True,
            "score": score,
            "errors": [],
            "revision_hint": "Accepted local edit passed original-constraint post-check.",
            "overridden_after_accepted_local_edit": True,
        }
        result["score"] = max(_coerce_float(result.get("score"), default=0.0), score)
    result["errors"] = []
    result["user_grounded"] = False
    result["revision_hint"] = "Accepted local edit passed original-constraint post-check."
    return result


def _mark_accepted_typed_action(
    critique: Mapping[str, Any],
    typed_action: Mapping[str, Any],
) -> dict[str, Any]:
    """Clear stale failures after a typed regeneration candidate passes hard VQA."""

    result = deepcopy(dict(critique))
    selected_check = typed_action.get("selected_constraint_check")
    selected_check = selected_check if isinstance(selected_check, Mapping) else {}
    score = _coerce_float(selected_check.get("score"), default=1.0)
    result["accepted_typed_action"] = True
    result["accepted_typed_action_summary"] = {
        "route": typed_action.get("route"),
        "selected_image": typed_action.get("selected_image"),
        "selected_prompt": typed_action.get("selected_prompt"),
        "selected_index": typed_action.get("selected_index"),
        "score": score,
    }
    result["constraint_check"] = _strip_large_prompt(deepcopy(dict(selected_check)))
    evaluation = result.get("evaluation")
    if isinstance(evaluation, Mapping) and evaluation.get("passed") is False:
        previous = deepcopy(dict(evaluation))
        result["pre_accepted_typed_action_evaluation"] = _strip_large_prompt(previous)
        result["evaluation"] = {
            **previous,
            "passed": True,
            "score": score,
            "errors": [],
            "revision_hint": "Accepted typed action candidate passed original-constraint post-check.",
            "overridden_after_accepted_typed_action": True,
        }
    result["score"] = max(_coerce_float(result.get("score"), default=0.0), score)
    result["errors"] = []
    result["user_grounded"] = False
    result["revision_hint"] = "Accepted typed action candidate passed original-constraint post-check."
    return result


def _merge_relation_repair_verification(
    critique: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    result = deepcopy(dict(critique))
    result["pre_relation_repair_feedback"] = _strip_large_prompt(result)
    verify_record = deepcopy(dict(verification))
    result["relation_repair_verification"] = _strip_large_prompt(verify_record)
    result["score"] = _coerce_float(verify_record.get("score"), default=0.0)
    result["errors"] = _dedupe_prompt_errors(list(verify_record.get("errors", []) or []))
    result["strengths"] = list(
        verify_record.get("strengths", []) or result.get("strengths", [])
    )
    result["revision_hint"] = str(
        verify_record.get("revision_hint") or result.get("revision_hint") or ""
    )
    result["relation_repaired"] = True
    result["user_grounded"] = not bool(verify_record.get("passed", True))
    return result


def _first_recolor_repair_plan(
    constraints: PromptConstraints,
    critique: Mapping[str, Any],
    *,
    preferred_object: str | None = None,
) -> dict[str, str] | None:
    if not constraints.colors:
        return None
    haystack = " ".join(_critique_reasons(critique)).lower()
    items = list(constraints.colors.items())
    preferred = str(preferred_object or "").strip().lower()
    if preferred:
        items = sorted(
            items,
            key=lambda item: 0
            if preferred in item[0].lower() or item[0].lower() in preferred
            else 1,
        )
    for object_name, target_color in items:
        if preferred and preferred not in object_name.lower() and object_name.lower() not in preferred:
            continue
        source_color = _source_color_from_structured_failures(
            critique,
            object_name=object_name,
            target_color=target_color,
            constraints=constraints,
        )
        if not source_color:
            source_color = _source_color_for_repair(
                haystack,
                object_name=object_name,
                target_color=target_color,
                constraints=constraints,
            )
        if not source_color:
            continue
        return {
            "target_name": object_name.split()[-1],
            "target_color_name": target_color,
            "target_color": _color_hex(target_color),
            "source_color": source_color,
            "prompt": f"repair the {object_name} so it is clearly {target_color}",
            "negative_prompt": f"{source_color} {object_name}, wrong {object_name} color",
            "target_region": _target_region_for_recolor(object_name),
            "subtract_other_objects": True,
        }
    return None


def _source_color_from_structured_failures(
    critique: Mapping[str, Any],
    *,
    object_name: str,
    target_color: str,
    constraints: PromptConstraints,
) -> str | None:
    object_terms = {object_name.lower(), object_name.split()[-1].lower()}
    colors = sorted(set(constraints.colors.values()) | _REPAIR_SOURCE_COLORS)
    matching_items: list[Mapping[str, Any]] = []
    for item in _iter_failure_records(critique):
        question_id = str(item.get("question_id") or "").lower()
        category = str(item.get("category") or item.get("type") or "").lower()
        target = str(item.get("target") or item.get("prompt_span") or "").lower()
        if "color:" not in question_id and "color" not in category and category != "wrong_attribute":
            continue
        if not any(term and term in " ".join([question_id, target]) for term in object_terms):
            continue
        matching_items.append(item)

    for item in matching_items:
        observed = str(item.get("observed") or "").strip().lower()
        if observed and observed not in {"no", "none", "uncertain", target_color}:
            normalized = _normalize_repair_color(observed)
            if normalized and normalized != target_color:
                return normalized

    for item in matching_items:
        text = " ".join(
            str(item.get(key) or "")
            for key in ("evidence", "description", "observed", "message")
        ).lower()
        for color in colors:
            if color == target_color:
                continue
            if any(
                _color_object_mismatch_mentioned(text, color, term)
                for term in object_terms
            ):
                return color
    return None


def _iter_failure_records(critique: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for key in ("constraint_check", "evaluation"):
        nested = critique.get(key)
        if not isinstance(nested, Mapping):
            continue
        for item in _as_mapping_records(nested.get("checks", [])):
            if item.get("passed") is False:
                records.append(item)
        records.extend(_as_mapping_records(nested.get("errors", [])))
    for value in (critique.get("checks", []),):
        for item in _as_mapping_records(value):
            if item.get("passed") is False:
                records.append(item)
    for value in (critique.get("errors", []),):
        records.extend(_as_mapping_records(value))
    return records


def _as_mapping_records(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if isinstance(item, Mapping)]
    return []


def _normalize_repair_color(value: str) -> str | None:
    text = str(value or "").strip().lower()
    aliases = {"grey": "gray", "golden": "gold"}
    text = aliases.get(text, text)
    if text in _REPAIR_SOURCE_COLORS:
        return text
    return None


def _source_color_for_repair(
    haystack: str,
    *,
    object_name: str,
    target_color: str,
    constraints: PromptConstraints,
) -> str | None:
    colors = sorted(set(constraints.colors.values()) | _REPAIR_SOURCE_COLORS)
    object_terms = [object_name, object_name.split()[-1]]
    low_saturation_terms = (
        "transparent",
        "translucent",
        "clear",
        "silver",
        "silvery",
        "gray",
        "grey",
        "white",
        "glass",
    )
    for color in colors:
        if color == target_color:
            continue
        if any(_color_object_mismatch_mentioned(haystack, color, term) for term in object_terms):
            return color
    if any(term in haystack for term in low_saturation_terms) and any(
        term and term in haystack for term in object_terms
    ):
        return "low_saturation"
    if not any(term and term in haystack for term in object_terms):
        return None
    if "wrong color" in haystack or "not " + target_color in haystack:
        for color in colors:
            if color != target_color:
                return color
    return None


def _color_object_mismatch_mentioned(text: str, color: str, object_term: str) -> bool:
    color = re.escape(color.lower())
    object_term = re.escape(object_term.lower())
    patterns = [
        rf"\b{color}\s+{object_term}\b",
        rf"\b{object_term}\s+(?:is|appears|looks|became|rendered|shown|visible as)\s+{color}\b",
        rf"\b{object_term}\b.{0,40}\b{color}\b",
        rf"\b{color}\b.{0,40}\b{object_term}\b",
    ]
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def _target_region_for_recolor(object_name: str) -> str:
    lowered = object_name.lower()
    if any(term in lowered for term in ("umbrella", "canopy", "parasol")):
        return "canopy"
    return "full"


_REPAIR_SOURCE_COLORS = {
    "red",
    "orange",
    "yellow",
    "green",
    "cyan",
    "blue",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "grey",
    "silver",
}


def _repair_acceptance_constraints(
    constraints: PromptConstraints,
    repair_plan: Mapping[str, Any],
) -> dict[str, Any]:
    repaired_target = str(repair_plan.get("target_name", "")).strip().lower()
    color_requirements = {
        object_name: color for object_name, color in constraints.colors.items()
    }
    return {
        "color_requirements": color_requirements,
        "must_preserve_non_target_colors": [
            f"{color} {object_name}"
            for object_name, color in color_requirements.items()
            if repaired_target not in object_name and object_name not in repaired_target
        ],
        "instruction": (
            "Accept the repair only if all original user color bindings still hold. "
            "Reject if fixing the target object's color changes any other user-specified object color."
        ),
    }


def _local_repair_target_evidence(
    repair_plan: Mapping[str, Any],
    detection: Mapping[str, Any],
    edit_result: Mapping[str, Any],
    constraints: PromptConstraints,
) -> dict[str, Any]:
    target_name = str(repair_plan.get("target_name") or "").strip()
    target_color = str(
        repair_plan.get("target_color_name") or repair_plan.get("target_color") or ""
    ).strip()
    protected_objects = _protected_objects_for_target(constraints, target_name)
    return {
        "evidence_type": "local_recolor_target",
        "target_object": target_name,
        "target_region": repair_plan.get("target_region", "full"),
        "target_attribute": "color",
        "target_value": target_color,
        "source": detection.get("method", "local_repair_detection"),
        "bbox": detection.get("detected_bbox") or detection.get("constrained_bbox"),
        "constrained_bbox": detection.get("constrained_bbox"),
        "layout_bbox_scaled": detection.get("layout_bbox_scaled"),
        "bbox_provenance": detection.get("bbox_provenance", {}),
        "mask_path": edit_result.get("mask_path"),
        "bbox_mask_path": edit_result.get("bbox_mask_path"),
        "precomputed_mask_path": detection.get("precomputed_mask_path"),
        "mask_refinement": _strip_large_prompt(detection.get("mask_refinement", {})),
        "target_localization": _strip_large_prompt(
            detection.get("target_localization", {})
        ),
        "protected_bboxes": detection.get("subtract_bboxes", []),
        "protected_objects": protected_objects,
        "geometry": detection.get("geometry"),
        "expected_post_edit_constraints": _expected_post_edit_constraints(
            constraints,
            action=f"recolor {target_name}",
        ),
    }


def _object_insertion_target_evidence(
    target_name: str,
    region: Mapping[str, Any],
    edit_result: Mapping[str, Any],
    constraints: PromptConstraints,
) -> dict[str, Any]:
    return {
        "evidence_type": "object_insertion_target",
        "target_object": target_name,
        "target_attribute": "existence_or_count",
        "source": "layout_or_repair_plan_region",
        "bbox": region.get("bbox"),
        "mask_path": edit_result.get("mask_path"),
        "protected_bboxes": [],
        "protected_objects": _protected_objects_for_target(constraints, target_name),
        "expected_post_edit_constraints": _expected_post_edit_constraints(
            constraints,
            action=f"insert {target_name}",
        ),
    }


def _object_insertion_region(
    layout_context: Mapping[str, Any] | None,
    target_name: str,
    repair_plan: Mapping[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    canvas_size: tuple[int, int],
) -> InpaintRegion:
    if layout_context:
        return plan_inpaint_region_from_layout(
            layout_context,
            target_name,
            prompt=prompt,
            negative_prompt=negative_prompt,
            expand=0.14,
        )
    if str(repair_plan.get("typed_route") or "") != "occlusion_object_insertion":
        raise ValueError("object insertion requires layout_context unless using typed occlusion repair")
    return _occlusion_inpaint_region_from_repair_plan(
        target_name,
        repair_plan,
        prompt=prompt,
        negative_prompt=negative_prompt,
        canvas_size=canvas_size,
    )


def _occlusion_inpaint_region_from_repair_plan(
    target_name: str,
    repair_plan: Mapping[str, Any],
    *,
    prompt: str,
    negative_prompt: str,
    canvas_size: tuple[int, int],
) -> InpaintRegion:
    width, height = _coerce_canvas_size(canvas_size)
    region = str(repair_plan.get("target_region") or "center").strip().lower()
    if region == "lower_half":
        bbox = [int(width * 0.22), int(height * 0.58), int(width * 0.56), int(height * 0.22)]
    elif region == "upper_half":
        bbox = [int(width * 0.22), int(height * 0.20), int(width * 0.56), int(height * 0.22)]
    elif region == "left_half":
        bbox = [int(width * 0.18), int(height * 0.26), int(width * 0.24), int(height * 0.50)]
    elif region == "right_half":
        bbox = [int(width * 0.58), int(height * 0.26), int(width * 0.24), int(height * 0.50)]
    else:
        bbox = [int(width * 0.25), int(height * 0.35), int(width * 0.50), int(height * 0.30)]
    bbox = [
        max(0, min(width - 1, int(bbox[0]))),
        max(0, min(height - 1, int(bbox[1]))),
        max(1, min(width - int(bbox[0]), int(bbox[2]))),
        max(1, min(height - int(bbox[1]), int(bbox[3]))),
    ]
    spec = repair_plan.get("occlusion_spec")
    if not isinstance(spec, Mapping):
        spec = {}
    hidden_part = str(spec.get("hidden_part") or region.replace("_", " "))
    target = str(spec.get("target") or "target object")
    visible_part = str(spec.get("visible_part") or "").strip()
    preserve = f"; preserve the visible {visible_part}" if visible_part else ""
    return InpaintRegion(
        name=target_name,
        bbox=bbox,
        prompt=(
            f"{prompt}; place the {target_name} as a foreground occluder over "
            f"the {hidden_part} of the {target}{preserve}"
        ),
        negative_prompt=negative_prompt,
        reason=(
            "typed occlusion repair fallback region from prompt statistics and "
            f"hidden_part={hidden_part}"
        ),
        canvas_size=[width, height],
    )


def _relation_repair_target_evidence(
    repair: Mapping[str, Any],
    constraints: PromptConstraints,
) -> dict[str, Any]:
    region = repair.get("region", {})
    detection = repair.get("detection", {})
    if not isinstance(region, Mapping):
        region = {}
    if not isinstance(detection, Mapping):
        detection = {}
    return {
        "evidence_type": "relation_action_contact",
        "target_object": "relation/action contact region",
        "target_part": "contact region",
        "source": detection.get("method", "relation_repair_region"),
        "bbox": region.get("bbox") or detection.get("detected_bbox"),
        "mask_path": _selected_candidate_mask_path(repair),
        "protected_bboxes": [],
        "protected_objects": _protected_objects_for_target(
            constraints,
            str(region.get("name") or ""),
        ),
        "expected_post_edit_constraints": _expected_post_edit_constraints(
            constraints,
            action="repair relation/action",
        ),
    }


def _expected_post_edit_constraints(
    constraints: PromptConstraints,
    *,
    action: str,
) -> dict[str, Any]:
    return {
        "action": action,
        "must_preserve_original_prompt": constraints.original_prompt,
        "color_requirements": dict(constraints.colors),
        "subjects": list(constraints.subjects),
        "actions": list(constraints.actions),
        "relations": list(constraints.relations),
        "protected_phrases": list(constraints.protected_phrases),
        "acceptance_rule": (
            "The edit is valid only if the original user constraints still pass "
            "after repair; fixing one attribute or relation cannot break another."
        ),
    }


def _protected_objects_for_target(
    constraints: PromptConstraints,
    target_name: str,
) -> list[dict[str, Any]]:
    target = str(target_name or "").lower()
    protected: list[dict[str, Any]] = []
    for object_name, color in constraints.colors.items():
        lowered = object_name.lower()
        if target and (target in lowered or lowered in target):
            continue
        protected.append(
            {
                "object": object_name,
                "protected_attribute": "color",
                "expected": color,
            }
        )
    return protected


def _selected_candidate_mask_path(repair: Mapping[str, Any]) -> str | None:
    selected_index = int(repair.get("selected_index", 0) or 0)
    candidates = repair.get("candidates", [])
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        for item in candidates:
            if not isinstance(item, Mapping):
                continue
            if int(item.get("index", -1) or -1) == selected_index:
                mask_path = str(item.get("mask_path") or "")
                return mask_path or None
    return None


def _local_repair_target_verification_context(
    repair_plan: Mapping[str, Any],
    detection: Mapping[str, Any],
    edit_result: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "instruction": (
            "Judge the edited image after local repair. Verify that the edited mask/bbox "
            "corresponds to the requested target object or target region, not a background "
            "object. The repaired image must satisfy the original user prompt."
        ),
        "target_name": repair_plan.get("target_name"),
        "target_region": repair_plan.get("target_region", "full"),
        "target_color_name": repair_plan.get("target_color_name"),
        "source_color": repair_plan.get("source_color"),
        "layout_bbox_scaled": detection.get("layout_bbox_scaled"),
        "constrained_bbox": detection.get("constrained_bbox"),
        "detected_bbox": detection.get("detected_bbox"),
        "selected_component": detection.get("selected_component"),
        "geometry": detection.get("geometry"),
        "target_localization": _strip_large_prompt(
            detection.get("target_localization", {})
        ),
        "mask_refinement": _strip_large_prompt(detection.get("mask_refinement", {})),
        "mask_path": edit_result.get("mask_path"),
        "bbox_mask_path": edit_result.get("bbox_mask_path"),
        "acceptance_rule": (
            "Return passed=false if the bbox/mask is on background signage, lights, "
            "reflections, unrelated clothing/body parts, or any non-target object, even "
            "when the edited pixels have the requested color."
        ),
    }


def _prompt_with_local_repair_context(
    selected_prompt: str,
    repair: Mapping[str, Any],
) -> str:
    repair_plan = repair.get("repair_plan", {})
    detection = repair.get("detection", {})
    edit_result = repair.get("edit_result", {})
    target_evidence = repair.get("target_evidence", {})
    if not isinstance(repair_plan, Mapping):
        repair_plan = {}
    if not isinstance(detection, Mapping):
        detection = {}
    if not isinstance(edit_result, Mapping):
        edit_result = {}
    if not isinstance(target_evidence, Mapping):
        target_evidence = {}
    mask_refinement = detection.get("mask_refinement", {})
    geometry_checks = (
        mask_refinement.get("geometry_checks", {})
        if isinstance(mask_refinement, Mapping)
        else {}
    )
    mask_result = (
        mask_refinement.get("result", {})
        if isinstance(mask_refinement, Mapping)
        else {}
    )
    context = {
        "post_repair_instruction": (
            "Evaluate the edited image against the original user prompt. Accept only "
            "if the full visible target object/part, not just the edited pixels, satisfies "
            "the requested attribute and original protected constraints remain."
        ),
        "target_name": repair_plan.get("target_name"),
        "target_region": repair_plan.get("target_region"),
        "target_color_name": repair_plan.get("target_color_name"),
        "source_color": repair_plan.get("source_color"),
        "detected_bbox": detection.get("detected_bbox"),
        "constrained_bbox": detection.get("constrained_bbox"),
        "bbox_source": (
            detection.get("bbox_provenance", {}).get("bbox_source")
            if isinstance(detection.get("bbox_provenance"), Mapping)
            else None
        ),
        "mask_method": mask_result.get("method") if isinstance(mask_result, Mapping) else None,
        "mask_geometry_passed": (
            geometry_checks.get("passed")
            if isinstance(geometry_checks, Mapping)
            else None
        ),
        "coverage_before": _coverage_summary(repair.get("coverage_before", {})),
        "coverage_after": _coverage_summary(repair.get("coverage_after", {})),
        "full_object_coverage_before": _coverage_summary(
            repair.get("full_object_coverage_before", {})
        ),
        "full_object_coverage_after": _coverage_summary(
            repair.get("full_object_coverage_after", {})
        ),
        "protected_objects": list(target_evidence.get("protected_objects", []) or [])[:4],
        "coverage_caution": (
            "Coverage numbers are diagnostic only. Re-inspect the edited image visually. "
            "Reject if the target object still has a visible wrong-color rim, wrong-color "
            "outer surface, hidden handle, weak grip, or protected-object damage."
        ),
        "reject_if": (
            "Reject if the target object is still the wrong color, if a protected "
            "object changed color, if the relation/action is still not clearly visible, "
            "or if the edit is on background/non-target content."
        ),
    }
    return _truncate_text(selected_prompt, max_chars=650) + "\nLocal repair verification context: " + json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
    )


def _coverage_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    keys = (
        "source_color",
        "source_coverage",
        "target_color",
        "target_coverage",
        "eligible_pixel_count",
    )
    return {key: value.get(key) for key in keys if key in value}


def _truncate_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def _prompt_with_object_repair_context(
    selected_prompt: str,
    repair: Mapping[str, Any],
) -> str:
    repair_plan = repair.get("repair_plan", {})
    region = repair.get("region", {})
    if not isinstance(repair_plan, Mapping):
        repair_plan = {}
    if not isinstance(region, Mapping):
        region = {}
    context = {
        "post_repair_instruction": (
            "This image is after a local object insertion edit. Evaluate the original "
            "user constraints on the edited image. The edit only counts as valid if "
            "the missing target object was added and all existing user-specified "
            "objects, colors, actions, and relations are preserved."
        ),
        "target_object": repair_plan.get("target_object"),
        "target_attribute": repair_plan.get("target_attribute"),
        "region": {
            "name": region.get("name"),
            "bbox": region.get("bbox"),
            "reason": region.get("reason"),
        },
        "reject_if": (
            "Reject if the insertion replaces the wrong object, changes another "
            "user-specified color, creates a duplicate wrong subject, or still leaves "
            "the requested subject/object missing."
        ),
    }
    return selected_prompt + "\nObject insertion repair verification context: " + json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
    )


def _prompt_with_relation_repair_context(
    selected_prompt: str,
    repair: Mapping[str, Any],
) -> str:
    repair_plan = repair.get("repair_plan", {})
    region = repair.get("region", {})
    verification = repair.get("verification", {})
    if not isinstance(repair_plan, Mapping):
        repair_plan = {}
    if not isinstance(region, Mapping):
        region = {}
    if not isinstance(verification, Mapping):
        verification = {}
    context = {
        "post_repair_instruction": (
            "This image is after a relation/action local edit. Evaluate every original "
            "user constraint on the edited final image, including user-specified colors, "
            "subjects, actions, and relations. The edit is invalid if it fixes contact "
            "but changes a required color or object attribute."
        ),
        "repair_strategy": repair_plan.get("strategy"),
        "edited_image": repair.get("edited_image"),
        "source_image": repair.get("source_image"),
        "region": {
            "name": region.get("name"),
            "bbox": region.get("bbox"),
            "reason": region.get("reason"),
        },
        "relation_verification": {
            "passed": verification.get("passed"),
            "score": verification.get("score"),
            "checks": verification.get("checks"),
        },
        "reject_if": (
            "Reject if any user color binding is wrong, for example a blue umbrella "
            "became red, even when the hand/handle relation looks repaired."
        ),
    }
    return selected_prompt + "\nRelation repair verification context: " + json.dumps(
        context,
        ensure_ascii=False,
        sort_keys=True,
    )


def _object_insertion_prompt(
    target_name: str,
    user_prompt: str,
    constraints: PromptConstraints,
) -> str:
    target = target_name.strip()
    color = ""
    for object_name, object_color in constraints.colors.items():
        if target.lower() in object_name.lower() or object_name.lower() in target.lower():
            color = object_color
            break
    colored_target = f"{color} {target}".strip()
    return (
        f"add the missing {colored_target}, match the original prompt: {user_prompt}, "
        "preserve existing objects, colors, lighting, camera, rain, and composition"
    )


def _object_insertion_negative_prompt(
    target_name: str,
    constraints: PromptConstraints,
) -> str:
    protected = ", ".join(
        f"changed {object_name} color, wrong {object_name} color"
        for object_name in constraints.colors
        if target_name.lower() not in object_name.lower()
    )
    base = (
        "duplicate subject, wrong object, changed existing object, changed background, "
        "distorted body, extra limbs, blurry insertion"
    )
    return ", ".join(part for part in (base, protected) if part)


def _efficient_repair_request_from_plan(
    route: str,
    repair_plan: Mapping[str, Any],
    selected_image: str,
    user_prompt: str,
    constraints: PromptConstraints,
    *,
    output_dir: Path,
    canvas_size: tuple[int, int],
) -> EfficientRepairRequest:
    bbox = _repair_plan_bbox(repair_plan, canvas_size)
    target_object = str(
        repair_plan.get("target_object")
        or repair_plan.get("target_name")
        or _default_efficient_repair_target(route)
    )
    prompt = str(repair_plan.get("edit_prompt") or repair_plan.get("reason") or user_prompt)
    if route in {"shape_overlay", "bbox_shape_inpaint"} and str(repair_plan.get("typed_route") or "") == "occlusion_object_insertion":
        region = _occlusion_inpaint_region_from_repair_plan(
            target_object,
            repair_plan,
            prompt=prompt,
            negative_prompt=_object_insertion_negative_prompt(target_object, constraints),
            canvas_size=canvas_size,
        )
        bbox = list(region.bbox)
        prompt = region.prompt
    return EfficientRepairRequest(
        repair_kind=route,
        image_path=selected_image,
        output_dir=output_dir,
        bbox=bbox,
        target_object=target_object,
        prompt=prompt,
        text=_exact_text_from_prompt_or_plan(user_prompt, repair_plan),
        symbol=_symbol_from_prompt_or_plan(user_prompt, repair_plan),
        fill_color=_fill_color_from_prompt_or_plan(user_prompt, repair_plan, constraints),
        text_color=_text_color_from_prompt_or_plan(user_prompt, repair_plan),
        negative_prompt=str(
            repair_plan.get("negative_prompt")
            or (
                _object_insertion_negative_prompt(target_object, constraints)
                if route == "bbox_shape_inpaint"
                else ""
            )
        ),
        reason=str(repair_plan.get("reason") or "efficient deterministic repair"),
        canvas_size=[canvas_size[0], canvas_size[1]],
    )


def _default_efficient_repair_target(route: str) -> str:
    route = str(route or "").strip().lower()
    if route == "text_overlay":
        return "text region"
    if route == "symbol_overlay":
        return "symbol region"
    if route == "shape_overlay":
        return "shape region"
    return "local repair region"


def _target_for_efficient_bbox_localization(
    repair_plan: Mapping[str, Any],
    constraints: PromptConstraints,
) -> str:
    target = str(
        repair_plan.get("target_object")
        or repair_plan.get("target_name")
        or ""
    ).strip()
    target = _clean_forbidden_localization_target(target)
    if target:
        return target
    typed_route = str(repair_plan.get("typed_route") or "")
    if typed_route.startswith("forbidden_") and constraints.intent_spec is not None:
        negatives = getattr(constraints.intent_spec, "negative_constraints", []) or []
        for item in negatives:
            target = _clean_forbidden_localization_target(str(item))
            if target:
                return target
    return ""


def _clean_forbidden_localization_target(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"\bno\s+(?:visible\s+)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwithout\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bmust\s+not\s+(?:show|contain|have)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bshould\s+not\s+(?:show|contain|have)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bis\s+visible\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bare\s+visible\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpresent\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bnearby\b", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"\b([a-z0-9-]+)\s+(?:reads?|says?|has|have)\b.*$",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b(?:lower|upper|left|right|bottom|top)\s+half\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:lower|upper|left|right|bottom|top)\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" .,:;")
    return text


def _efficient_repair_gate(
    route: str,
    repair_plan: Mapping[str, Any],
    agent: EfficientRepairAgent,
    *,
    canvas_size: tuple[int, int],
) -> dict[str, Any]:
    """Return whether an efficient repair route should execute in the main loop."""

    route = str(route or "").strip().lower()
    if route in {"text_overlay", "symbol_overlay"}:
        return {"allowed": True, "reason": "deterministic overlay route"}
    if route == "shape_overlay":
        low_editability = _low_editability_repair_plan(repair_plan)
        if low_editability:
            return {
                "allowed": False,
                "reason": low_editability,
            }
        bbox = _repair_plan_bbox_or_none(repair_plan)
        if bbox is None and str(repair_plan.get("typed_route") or "") == "occlusion_object_insertion":
            bbox = _occlusion_bbox_from_repair_plan(repair_plan, canvas_size)
        if bbox is None:
            return {
                "allowed": False,
                "reason": "shape overlay requires an explicit localized bbox",
            }
        bbox_confidence = _optional_float(repair_plan.get("bbox_confidence"))
        if bbox_confidence is not None and bbox_confidence < 0.50:
            return {
                "allowed": False,
                "reason": "localized bbox confidence is too low for automatic shape overlay",
                "bbox_confidence": bbox_confidence,
            }
        width, height = _coerce_canvas_size(canvas_size)
        area_ratio = (bbox[2] * bbox[3]) / max(1, width * height)
        if area_ratio > 0.42:
            return {
                "allowed": False,
                "reason": "localized bbox is too large for safe automatic shape overlay",
                "bbox": bbox,
                "area_ratio": round(area_ratio, 4),
                "max_area_ratio": 0.42,
            }
        return {
            "allowed": True,
            "reason": "localized deterministic shape overlay route passed safety gate",
            "bbox": bbox,
            "bbox_confidence": bbox_confidence,
            "area_ratio": round(area_ratio, 4),
        }
    if route not in {"bbox_shape_inpaint", "existing_object_inpaint"}:
        return {"allowed": False, "reason": f"unsupported efficient repair route: {route}"}
    if agent.inpaint_agent is None:
        return {
            "allowed": False,
            "reason": "efficient inpaint route requires an inpaint backend",
        }
    low_editability = _low_editability_repair_plan(repair_plan)
    if low_editability:
        return {
            "allowed": False,
            "reason": low_editability,
        }
    bbox = _repair_plan_bbox_or_none(repair_plan)
    if bbox is None and str(repair_plan.get("typed_route") or "") == "occlusion_object_insertion":
        bbox = _occlusion_bbox_from_repair_plan(repair_plan, canvas_size)
    if bbox is None:
        return {
            "allowed": False,
            "reason": "efficient inpaint route requires an explicit localized bbox",
        }
    bbox_confidence = _optional_float(repair_plan.get("bbox_confidence"))
    if bbox_confidence is not None and bbox_confidence < 0.50:
        return {
            "allowed": False,
            "reason": "localized bbox confidence is too low for automatic editing",
            "bbox_confidence": bbox_confidence,
        }
    width, height = _coerce_canvas_size(canvas_size)
    area_ratio = (bbox[2] * bbox[3]) / max(1, width * height)
    max_area = 0.42 if route == "bbox_shape_inpaint" else 0.35
    if area_ratio > max_area:
        return {
            "allowed": False,
            "reason": "localized bbox is too large for safe automatic editing",
            "bbox": bbox,
            "area_ratio": round(area_ratio, 4),
            "max_area_ratio": max_area,
        }
    target = str(repair_plan.get("target_object") or repair_plan.get("target_name") or "").strip()
    if not target:
        return {
            "allowed": False,
            "reason": "efficient inpaint route requires a target_object/target_name",
        }
    return {
        "allowed": True,
        "reason": "localized efficient inpaint route passed safety gate",
        "bbox": bbox,
        "bbox_confidence": bbox_confidence,
        "area_ratio": round(area_ratio, 4),
    }


def _repair_plan_bbox_or_none(repair_plan: Mapping[str, Any]) -> list[int] | None:
    for key in (
        "bbox",
        "target_bbox",
        "edit_bbox",
        "layout_bbox",
        "layout_bbox_scaled",
        "constrained_bbox",
        "detected_bbox",
    ):
        value = repair_plan.get(key)
        if _is_bbox(value):
            return [int(float(item)) for item in value]
    return None


def _occlusion_bbox_from_repair_plan(
    repair_plan: Mapping[str, Any],
    canvas_size: tuple[int, int],
) -> list[int]:
    target_name = str(repair_plan.get("target_object") or "occluder")
    region = _occlusion_inpaint_region_from_repair_plan(
        target_name,
        repair_plan,
        prompt=str(repair_plan.get("edit_prompt") or repair_plan.get("reason") or "typed occlusion repair"),
        negative_prompt="",
        canvas_size=canvas_size,
    )
    return list(region.bbox)


def _low_editability_repair_plan(repair_plan: Mapping[str, Any]) -> str | None:
    route = str(
        repair_plan.get("typed_route")
        or repair_plan.get("repair_kind")
        or repair_plan.get("primary_action")
        or ""
    ).lower()
    attr = str(repair_plan.get("target_attribute") or "").lower()
    error_type = str(repair_plan.get("error_type") or "").lower()
    editability = str(repair_plan.get("editability") or "").lower()
    combined = " ".join([route, attr, error_type, editability])
    if any(token in combined for token in ("layout", "spatial", "position", "left", "right")):
        return "spatial/layout failures should regenerate or use layout guidance, not local inpaint"
    if "count" in combined:
        return "count failures should rerank or regenerate, not local inpaint"
    if editability in {"low", "none", "regenerate", "rerank"}:
        return f"repair plan editability is {editability}"
    return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _repair_plan_bbox(
    repair_plan: Mapping[str, Any],
    canvas_size: tuple[int, int],
) -> list[int]:
    for key in (
        "bbox",
        "target_bbox",
        "edit_bbox",
        "layout_bbox",
        "layout_bbox_scaled",
        "constrained_bbox",
        "detected_bbox",
    ):
        value = repair_plan.get(key)
        if _is_bbox(value):
            return [int(item) for item in value]
    width, height = _coerce_canvas_size(canvas_size)
    return [int(width * 0.20), int(height * 0.10), int(width * 0.60), int(height * 0.26)]


def _exact_text_from_prompt_or_plan(
    user_prompt: str,
    repair_plan: Mapping[str, Any],
) -> str:
    for key in ("text", "exact_text", "target_text", "expected_text", "expected"):
        value = str(repair_plan.get(key) or "").strip()
        if value:
            quoted = re.search(r"['\"]([^'\"]+)['\"]", value)
            return (quoted.group(1) if quoted else value).strip("'\"")
    match = re.search(r"['\"]([^'\"]+)['\"]", user_prompt)
    if match:
        return match.group(1)
    lowered = user_prompt.lower()
    for token in ("no", "go", "on", "off", "stop"):
        if f" text {token!r}" in lowered:
            return token.upper()
    return ""


def _symbol_from_prompt_or_plan(
    user_prompt: str,
    repair_plan: Mapping[str, Any],
) -> str:
    for key in ("symbol", "target_symbol", "expected_symbol"):
        value = str(repair_plan.get(key) or "").strip()
        if value:
            return value
    lowered = user_prompt.lower()
    for symbol in ("triangle", "moon", "crescent", "star", "circle", "square"):
        if symbol in lowered:
            return symbol
    return ""


def _fill_color_from_prompt_or_plan(
    user_prompt: str,
    repair_plan: Mapping[str, Any],
    constraints: PromptConstraints,
) -> str:
    for key in ("fill_color", "background_color", "target_color", "target_color_name"):
        value = str(repair_plan.get(key) or "").strip()
        if value:
            return value
    target = str(repair_plan.get("target_object") or repair_plan.get("target_name") or "").lower()
    for object_name, color in constraints.colors.items():
        if target and (target in object_name.lower() or object_name.lower() in target):
            return str(color)
    sign_match = re.search(
        r"\b(black|white|red|blue|green|yellow|orange|purple|pink|cyan|turquoise|silver|gray|grey)\s+(?:sign|folder|lunchbox|notebook|label)",
        user_prompt,
        flags=re.IGNORECASE,
    )
    if sign_match:
        return sign_match.group(1)
    return "black"


def _text_color_from_prompt_or_plan(
    user_prompt: str,
    repair_plan: Mapping[str, Any],
) -> str:
    for key in ("text_color", "symbol_color", "foreground_color"):
        value = str(repair_plan.get(key) or "").strip()
        if value:
            return value
    match = re.search(
        r"\b(black|white|red|blue|green|yellow|orange|purple|pink|cyan|turquoise|silver|gray|grey)\s+(?:text|symbol|letter|triangle|moon|star|circle|square)",
        user_prompt,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1)
    return "yellow"


def _post_repair_constraint_failures(check: Mapping[str, Any]) -> list[dict[str, Any]]:
    if check.get("failed"):
        return [
            {
                "type": "post_repair_constraint_check_failed",
                "message": str(check.get("error") or "post-repair constraint check failed"),
            }
        ]
    if check.get("passed") is False:
        return [
            {
                "type": "post_repair_constraint_failed",
                "score": check.get("score"),
                "error_count": len(check.get("errors", []) or []),
                "errors": list(check.get("errors", []) or [])[:4],
                "message": "post-repair image still violates original user constraints",
            }
        ]
    if check.get("passed") is None:
        return [
            {
                "type": "post_repair_constraint_unknown",
                "message": "post-repair constraint check did not return a pass/fail decision",
            }
        ]
    return []


def _ocr_repair_failures(repair: Mapping[str, Any]) -> list[dict[str, Any]]:
    ocr = repair.get("ocr_verification")
    if not isinstance(ocr, Mapping):
        return []
    if ocr.get("available") is not True:
        return []
    if ocr.get("passed") is False:
        return [
            {
                "type": "ocr_text_mismatch",
                "expected": ocr.get("expected"),
                "recognized": ocr.get("recognized"),
                "similarity": ocr.get("similarity"),
                "message": "deterministic text repair failed OCR verification",
            }
        ]
    return []


def _efficient_repair_attempted_editing_backend(
    repair: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(repair, Mapping):
        return False
    route = str(repair.get("route") or "").strip().lower()
    if route not in {"shape_overlay", "bbox_shape_inpaint", "existing_object_inpaint"}:
        return False
    # A returned efficient local-edit record means the gate passed and this
    # route was selected for the current plan. Even backend/path/OOM/post-check
    # failures should consume the local-edit attempt so the loop does not
    # immediately launch a second object-repair subprocess for the same plan.
    return True


def _blocking_post_repair_constraint_failures(
    failures: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return post-repair failures that should roll back the edited image."""

    blocking_types = {"post_repair_constraint_failed"}
    return [
        dict(item)
        for item in failures
        if str(item.get("type") or "") in blocking_types
    ]


def _apply_local_repair_post_check(
    repair: dict[str, Any],
    post_check: Mapping[str, Any],
) -> None:
    """Attach post-check results without discarding useful edits on API failure."""

    repair["post_repair_constraint_check"] = dict(post_check)
    post_failures = _post_repair_constraint_failures(post_check)
    if not post_failures:
        repair.setdefault("acceptance", {})["post_repair_constraint_failures"] = []
        repair["acceptance"]["verification_unavailable"] = False
        return
    blocking_failures = _blocking_post_repair_constraint_failures(post_failures)
    repair.setdefault("acceptance", {})["post_repair_constraint_failures"] = post_failures
    repair["acceptance"]["verification_unavailable"] = not blocking_failures
    if blocking_failures:
        repair["acceptance"]["hard_gate_failures"] = [
            *list(repair["acceptance"].get("hard_gate_failures", [])),
            *blocking_failures,
        ]
        repair["acceptance"]["accepted"] = False
        repair["accepted"] = False


def _object_insertion_hard_gate_failures(
    region: Any,
    edit_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    scaled_bbox = edit_result.get("scaled_bbox")
    image_size = edit_result.get("source_image_size")
    if (
        not isinstance(scaled_bbox, Sequence)
        or isinstance(scaled_bbox, (str, bytes))
        or len(scaled_bbox) != 4
        or not isinstance(image_size, Sequence)
        or isinstance(image_size, (str, bytes))
        or len(image_size) != 2
    ):
        bbox = getattr(region, "bbox", [])
        canvas_size = getattr(region, "canvas_size", [1024, 1024])
        scaled_bbox = bbox
        image_size = canvas_size
    try:
        x, y, width, height = [int(value) for value in scaled_bbox]
        image_width, image_height = [int(value) for value in image_size]
    except (TypeError, ValueError):
        return [
            {
                "type": "object_insertion_region_unknown",
                "message": "object insertion edit region could not be measured",
            }
        ]
    del x, y
    image_area = max(1, image_width * image_height)
    area_ratio = max(0, width) * max(0, height) / image_area
    width_ratio = max(0, width) / max(1, image_width)
    height_ratio = max(0, height) / max(1, image_height)
    max_area_ratio = 0.20
    if area_ratio > max_area_ratio:
        failures.append(
            {
                "type": "object_insertion_region_too_large",
                "value": round(area_ratio, 6),
                "threshold": max_area_ratio,
                "bbox": [width, height],
                "message": "object insertion mask is too large for a local edit",
            }
        )
    if width_ratio > 0.65 or height_ratio > 0.65:
        failures.append(
            {
                "type": "object_insertion_region_side_too_large",
                "value": [round(width_ratio, 6), round(height_ratio, 6)],
                "threshold": [0.65, 0.65],
                "message": "object insertion mask spans too much of the image",
            }
        )
    return failures


def _local_repair_acceptance(
    repaired_eval: Mapping[str, Any],
    constraints: PromptConstraints,
    repair_plan: Mapping[str, Any],
    *,
    old_score: float,
    new_score: float,
    detection: Mapping[str, Any],
    coverage_before: Mapping[str, Any],
    coverage_after: Mapping[str, Any],
    component_coverage_before: Mapping[str, Any] | None = None,
    component_coverage_after: Mapping[str, Any] | None = None,
    full_object_coverage_before: Mapping[str, Any] | None = None,
    full_object_coverage_after: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    color_errors = _user_color_preservation_errors(
        repaired_eval,
        constraints,
        repair_plan,
    )
    score_improved = new_score >= old_score
    hard_gate_failures = _local_recolor_hard_gate_failures(
        detection,
        coverage_before,
        coverage_after,
        component_coverage_before=component_coverage_before,
        component_coverage_after=component_coverage_after,
        full_object_coverage_before=full_object_coverage_before,
        full_object_coverage_after=full_object_coverage_after,
    )
    accepted = score_improved and not color_errors and not hard_gate_failures
    return {
        "accepted": accepted,
        "score_improved": score_improved,
        "old_score": old_score,
        "new_score": new_score,
        "color_preservation_errors": color_errors,
        "hard_gate_failures": hard_gate_failures,
        "coverage_before": deepcopy(dict(coverage_before)),
        "coverage_after": deepcopy(dict(coverage_after)),
        "component_coverage_before": deepcopy(dict(component_coverage_before or {})),
        "component_coverage_after": deepcopy(dict(component_coverage_after or {})),
        "full_object_coverage_before": deepcopy(dict(full_object_coverage_before or {})),
        "full_object_coverage_after": deepcopy(dict(full_object_coverage_after or {})),
    }


def _mask_refinement_prompt_bbox(
    detection: Mapping[str, Any],
    region: Any,
) -> list[int]:
    for key in ("detected_bbox", "constrained_bbox", "layout_bbox_scaled"):
        bbox = detection.get(key)
        if _is_bbox(bbox):
            return [int(value) for value in bbox]
    bbox = getattr(region, "bbox", None)
    if _is_bbox(bbox):
        return [int(value) for value in bbox]
    raise ValueError("cannot refine mask without a valid target bbox")


def _is_bbox(value: Any) -> bool:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return False
    if len(value) != 4:
        return False
    try:
        x, y, width, height = [int(float(part)) for part in value]
    except (TypeError, ValueError):
        return False
    return x >= 0 and y >= 0 and width > 0 and height > 0


def _mask_refinement_source(detection: Mapping[str, Any]) -> str:
    locator = detection.get("target_localization", {})
    if isinstance(locator, Mapping) and locator.get("applied"):
        return "vlm_target_region_locator"
    return str(detection.get("method") or "local_repair_detection")


def _full_object_repair_mask_path(detection: Mapping[str, Any]) -> str | None:
    """Return a broader target-object/part mask for post-edit residual checks."""

    refinement = detection.get("mask_refinement", {})
    result = (
        refinement.get("result", {})
        if isinstance(refinement, Mapping)
        else {}
    )
    prior = (
        result.get("prior_constraint", {})
        if isinstance(result, Mapping) and isinstance(result.get("prior_constraint"), Mapping)
        else {}
    )
    prior_path = prior.get("prior_mask_path") if isinstance(prior, Mapping) else None
    if prior_path:
        return str(prior_path)
    mask_path = detection.get("object_region_mask_path") or detection.get("precomputed_mask_path")
    return str(mask_path) if mask_path else None


def _mask_refinement_geometry_checks(result: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    method = str(result.get("method") or "").strip().lower()
    fallback_used = bool(result.get("fallback_used"))
    selected_pixels = int(result.get("selected_pixel_count") or 0)
    area_ratio = _coerce_float(result.get("area_ratio"), default=0.0)
    geometry = result.get("geometry", {}) if isinstance(result.get("geometry"), Mapping) else {}
    mask_to_bbox = _coerce_float(geometry.get("mask_to_bbox_ratio"), default=0.0)
    protected = result.get("protected_overlap", {})
    protected_ratio = (
        _coerce_float(protected.get("overlap_ratio"), default=0.0)
        if isinstance(protected, Mapping)
        else 0.0
    )
    prior_constraint = (
        result.get("prior_constraint", {})
        if isinstance(result.get("prior_constraint"), Mapping)
        else {}
    )
    if prior_constraint.get("applied") is True:
        constrained_pixels = int(prior_constraint.get("constrained_pixel_count") or 0)
        if constrained_pixels <= 0:
            failures.append(
                {
                    "type": "target_prior_constrained_mask_empty",
                    "message": (
                        "mask refiner output has no overlap with the target-object "
                        "prior after protected regions are removed"
                    ),
                    "prior_mask_path": prior_constraint.get("prior_mask_path"),
                    "raw_mask_path": prior_constraint.get("raw_mask_path"),
                }
            )
    if selected_pixels <= 0:
        failures.append(
            {
                "type": "empty_refined_mask",
                "message": "mask refiner returned no editable pixels",
            }
        )
    if _uses_rectangular_bbox_mask(method, fallback_used) and area_ratio > 0.12:
        failures.append(
            {
                "type": "bbox_fallback_mask_too_large",
                "value": area_ratio,
                "threshold": 0.12,
                "message": (
                    "bbox fallback mask covers too much of the full image; use "
                    "SAM or a tighter image-grounded mask instead of rectangular edit"
                ),
            }
        )
    if _uses_shape_refined_mask(method, fallback_used):
        if mask_to_bbox > 0.92 and area_ratio > 0.12:
            failures.append(
                {
                    "type": "shape_refined_mask_degenerated_to_bbox",
                    "value": mask_to_bbox,
                    "threshold": 0.92,
                    "area_ratio": area_ratio,
                    "message": (
                        "shape-refined mask is almost the entire prompt bbox; "
                        "treat it like an unsafe rectangular edit"
                    ),
                }
            )
        if protected_ratio > 0.08:
            failures.append(
                {
                    "type": "shape_refined_mask_overlaps_protected_region",
                    "value": protected_ratio,
                    "threshold": 0.08,
                    "message": "shape-refined mask overlaps too much protected content",
                }
            )
    if mask_to_bbox < 0.03:
        failures.append(
            {
                "type": "refined_mask_low_bbox_coverage",
                "value": mask_to_bbox,
                "threshold": 0.03,
                "message": "refined mask covers too little of the prompted bbox",
            }
        )
    return {
        "passed": not failures,
        "failures": failures,
        "method": method,
        "fallback_used": fallback_used,
        "selected_pixel_count": selected_pixels,
        "area_ratio": area_ratio,
        "mask_to_bbox_ratio": mask_to_bbox,
        "protected_overlap_ratio": protected_ratio,
        "prior_constraint": deepcopy(dict(prior_constraint))
        if isinstance(prior_constraint, Mapping)
        else {},
    }


def _uses_rectangular_bbox_mask(method: str, fallback_used: bool) -> bool:
    return fallback_used or method in {
        "bbox_fallback",
        "mock_mask_refiner",
    } or method.startswith("bbox_")


def _uses_shape_refined_mask(method: str, fallback_used: bool) -> bool:
    return not fallback_used and method in {
        "sam_v1_bbox_prompt",
    }


def _local_repair_evidence_gate_failures(
    detection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    localization = detection.get("target_localization", {})
    if isinstance(localization, Mapping) and localization:
        if not localization.get("applied"):
            failures.append(
                {
                    "type": "target_locator_not_applied",
                    "message": (
                        "VLM target locator was requested but did not produce an "
                        "image-grounded target bbox, so layout-only repair is rejected"
                    ),
                    "reason": localization.get("reason") or localization.get("error"),
                }
            )
    refinement = detection.get("mask_refinement", {})
    if isinstance(refinement, Mapping) and refinement:
        geometry = refinement.get("geometry_checks", {})
        if isinstance(geometry, Mapping) and geometry.get("passed") is False:
            for failure in geometry.get("failures", []) or []:
                if isinstance(failure, Mapping):
                    failures.append({"type": "mask_refinement_failed", **dict(failure)})
    return failures


def _local_repair_bbox_provenance(
    detection: Mapping[str, Any],
    coverage_before: Mapping[str, Any],
    repair_plan: Mapping[str, Any],
) -> dict[str, Any]:
    localization = detection.get("target_localization", {})
    locator_applied = (
        isinstance(localization, Mapping) and localization.get("applied") is True
    )
    method = str(detection.get("method") or "")
    before_source = _coerce_float(coverage_before.get("source_coverage"), default=0.0)
    selected_bbox = (
        detection.get("detected_bbox")
        or detection.get("constrained_bbox")
        or detection.get("layout_bbox_scaled")
    )
    if locator_applied:
        bbox_source = "vlm_target_locator"
        image_grounded = True
        reason = "target bbox was localized on the generated image by VLM"
    elif method == "color_component_near_layout":
        bbox_source = "color_component_near_layout"
        image_grounded = True
        reason = "target bbox came from a source-color component in the generated image"
    elif before_source >= 0.03:
        bbox_source = "layout_prior_with_source_color_evidence"
        image_grounded = True
        reason = (
            "layout prior overlaps enough source-color pixels to be treated as "
            "image-supported for recolor"
        )
    else:
        bbox_source = "layout_prior"
        image_grounded = False
        reason = "bbox is only a layout prior and lacks generated-image target evidence"
    return {
        "bbox_source": bbox_source,
        "image_grounded": image_grounded,
        "reason": reason,
        "selected_bbox": selected_bbox,
        "layout_prior_bbox": detection.get("layout_bbox_scaled"),
        "constrained_bbox": detection.get("constrained_bbox"),
        "source_coverage_before": before_source,
        "target_name": repair_plan.get("target_name"),
        "source_color": repair_plan.get("source_color"),
        "target_color": repair_plan.get("target_color"),
    }


def _local_repair_pre_edit_gate_failures(
    detection: Mapping[str, Any],
    coverage_before: Mapping[str, Any],
    repair_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Reject local recolor before editing when the target bbox is only a prior."""

    failures: list[dict[str, Any]] = []
    source_color = str(repair_plan.get("source_color") or "").strip().lower()
    if source_color == "":
        return failures
    provenance = detection.get("bbox_provenance")
    if not isinstance(provenance, Mapping):
        provenance = _local_repair_bbox_provenance(
            detection,
            coverage_before,
            repair_plan,
        )
    before_source = _coerce_float(coverage_before.get("source_coverage"), default=0.0)
    localization = detection.get("target_localization", {})
    if (
        isinstance(localization, Mapping)
        and localization
        and localization.get("applied") is not True
    ):
        failures.append(
            {
                "type": "target_locator_not_applied_pre_edit",
                "reason": localization.get("reason") or localization.get("error"),
                "bbox_provenance": dict(provenance),
                "message": (
                    "VLM target locator was enabled but did not produce an "
                    "image-grounded bbox; local repair is skipped before editing"
                ),
            }
        )
    if provenance.get("image_grounded") is not True and before_source < 0.03:
        failures.append(
            {
                "type": "layout_only_bbox_not_on_source_target",
                "source_coverage": before_source,
                "threshold": 0.03,
                "target_name": repair_plan.get("target_name"),
                "source_color": source_color,
                "bbox_provenance": dict(provenance),
                "message": (
                    "layout-only recolor bbox contains too little of the source "
                    "color, so it is likely not on the generated target object"
                ),
            }
        )
    refinement = detection.get("mask_refinement", {})
    geometry = (
        refinement.get("geometry_checks", {})
        if isinstance(refinement, Mapping)
        else {}
    )
    if isinstance(geometry, Mapping) and geometry.get("passed") is False:
        for failure in geometry.get("failures", []) or []:
            if isinstance(failure, Mapping):
                failures.append(
                    {"type": "mask_refinement_failed_pre_edit", **dict(failure)}
                )
    return failures


def _local_recolor_hard_gate_failures(
    detection: Mapping[str, Any],
    coverage_before: Mapping[str, Any],
    coverage_after: Mapping[str, Any],
    *,
    component_coverage_before: Mapping[str, Any] | None = None,
    component_coverage_after: Mapping[str, Any] | None = None,
    full_object_coverage_before: Mapping[str, Any] | None = None,
    full_object_coverage_after: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    failures.extend(_local_repair_evidence_gate_failures(detection))
    for failure in detection.get("geometry_failures", []) or []:
        if isinstance(failure, Mapping):
            failures.append({"type": "mask_geometry_failed", **dict(failure)})
    overlap_ratio = _coerce_float(detection.get("selected_overlap_ratio"), default=0.0)
    if overlap_ratio < 0.05:
        failures.append(
            {
                "type": "low_layout_overlap",
                "value": overlap_ratio,
                "threshold": 0.05,
                "message": "selected color component barely overlaps the target layout region",
            }
        )
    before_target = _coerce_float(coverage_before.get("target_coverage"), default=0.0)
    after_target = _coerce_float(coverage_after.get("target_coverage"), default=0.0)
    before_source = _coerce_float(coverage_before.get("source_coverage"), default=0.0)
    after_source = _coerce_float(coverage_after.get("source_coverage"), default=0.0)
    target_gain = after_target - before_target
    source_drop = before_source - after_source
    if target_gain < 0.08:
        failures.append(
            {
                "type": "low_target_color_gain",
                "value": round(target_gain, 6),
                "threshold": 0.08,
                "before": before_target,
                "after": after_target,
                "message": "target color coverage did not increase enough",
            }
        )
    if before_source >= 0.08 and source_drop < 0.08:
        failures.append(
            {
                "type": "low_source_color_reduction",
                "value": round(source_drop, 6),
                "threshold": 0.08,
                "before": before_source,
                "after": after_source,
                "message": "source color remains too similar after recolor",
            }
        )
    if after_source > max(0.18, before_source * 0.7):
        failures.append(
            {
                "type": "high_source_color_remaining",
                "value": after_source,
                "threshold": max(0.18, round(before_source * 0.7, 6)),
                "before": before_source,
                "message": "too much source color remains in target region",
            }
        )
    if component_coverage_before and component_coverage_after:
        before_component_target = _coerce_float(
            component_coverage_before.get("target_coverage"), default=0.0
        )
        after_component_target = _coerce_float(
            component_coverage_after.get("target_coverage"), default=0.0
        )
        before_component_source = _coerce_float(
            component_coverage_before.get("source_coverage"), default=0.0
        )
        after_component_source = _coerce_float(
            component_coverage_after.get("source_coverage"), default=0.0
        )
        component_target_gain = after_component_target - before_component_target
        component_source_drop = before_component_source - after_component_source
        if component_target_gain < 0.16:
            failures.append(
                {
                    "type": "low_component_target_color_gain",
                    "value": round(component_target_gain, 6),
                    "threshold": 0.16,
                    "before": before_component_target,
                    "after": after_component_target,
                    "message": "target color did not increase enough inside the detected repair component",
                }
            )
        if before_component_source >= 0.08 and component_source_drop < 0.16:
            failures.append(
                {
                    "type": "low_component_source_color_reduction",
                    "value": round(component_source_drop, 6),
                    "threshold": 0.16,
                    "before": before_component_source,
                    "after": after_component_source,
                    "message": "source color did not reduce enough inside the detected repair component",
                }
            )
    if full_object_coverage_before and full_object_coverage_after:
        before_full_target = _coerce_float(
            full_object_coverage_before.get("target_coverage"), default=0.0
        )
        after_full_target = _coerce_float(
            full_object_coverage_after.get("target_coverage"), default=0.0
        )
        before_full_source = _coerce_float(
            full_object_coverage_before.get("source_coverage"), default=0.0
        )
        after_full_source = _coerce_float(
            full_object_coverage_after.get("source_coverage"), default=0.0
        )
        full_target_gain = after_full_target - before_full_target
        full_source_drop = before_full_source - after_full_source
        if after_full_target < 0.62:
            failures.append(
                {
                    "type": "low_full_object_target_color_coverage",
                    "value": after_full_target,
                    "threshold": 0.62,
                    "gain": round(full_target_gain, 6),
                    "message": (
                        "target color does not dominate the full target object/part, "
                        "not just the edited mask"
                    ),
                }
            )
        if before_full_source >= 0.12 and after_full_source > max(0.22, before_full_source * 0.45):
            failures.append(
                {
                    "type": "high_full_object_source_color_residual",
                    "value": after_full_source,
                    "threshold": max(0.22, round(before_full_source * 0.45, 6)),
                    "drop": round(full_source_drop, 6),
                    "before": before_full_source,
                    "message": (
                        "too much wrong source color remains on the full target "
                        "object/part after local recolor"
                    ),
                }
            )
    return failures


def _user_color_preservation_errors(
    evaluation: Mapping[str, Any],
    constraints: PromptConstraints,
    repair_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    repaired_target = str(repair_plan.get("target_name", "")).strip().lower()
    reasons = _critique_reasons(evaluation)
    checks = _collect_constraint_checks(evaluation)
    errors: list[dict[str, Any]] = []
    for object_name, expected_color in constraints.colors.items():
        if repaired_target and (
            repaired_target in object_name.lower()
            or object_name.lower() in repaired_target
        ):
            continue
        phrase = f"{expected_color} {object_name}".lower()
        object_terms = [object_name.lower(), object_name.lower().split()[-1]]
        wrong_color_reasons = [
            reason
            for reason in reasons
            if any(term in reason.lower() for term in object_terms)
            and (
                "wrong color" in reason.lower()
                or f"not {expected_color}" in reason.lower()
                or f"became {repair_plan.get('target_color_name', '')}" in reason.lower()
                or f"{repair_plan.get('target_color_name', '')} {object_terms[-1]}" in reason.lower()
            )
        ]
        failed_checks = [
            check
            for check in checks
            if check.get("passed") is False
            and any(term in str(check.get("target", "")).lower() for term in object_terms)
            and expected_color in str(check.get("expected", "")).lower()
        ]
        if wrong_color_reasons or failed_checks:
            errors.append(
                {
                    "object": object_name,
                    "expected": phrase,
                    "reasons": wrong_color_reasons[:3],
                    "failed_checks": failed_checks[:3],
                }
            )
    return errors


def _collect_constraint_checks(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    raw_checks = record.get("checks", [])
    if isinstance(raw_checks, Mapping):
        raw_checks = [raw_checks]
    if isinstance(raw_checks, list):
        checks.extend(dict(item) for item in raw_checks if isinstance(item, Mapping))
    context = record.get("context")
    if isinstance(context, Mapping):
        critique = context.get("critique")
        if isinstance(critique, Mapping):
            constraint_check = critique.get("constraint_check")
            if isinstance(constraint_check, Mapping):
                checks.extend(_collect_constraint_checks(constraint_check))
    nested = record.get("constraint_check")
    if isinstance(nested, Mapping):
        checks.extend(_collect_constraint_checks(nested))
    return checks


def _color_hex(color_name: str) -> str:
    return {
        "red": "#d62828",
        "blue": "#1d63d9",
        "green": "#2ca25f",
        "yellow": "#ffd43b",
        "black": "#111111",
        "white": "#f5f5f5",
        "orange": "#f97316",
        "purple": "#7c3aed",
        "pink": "#ec4899",
    }.get(color_name.lower(), "#1d63d9")


def _critique_reasons(critique: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    for item in critique.get("errors", []) or []:
        if isinstance(item, Mapping):
            reasons.extend(
                str(item.get(key, "")).strip()
                for key in ("evidence", "description", "reason", "prompt_span")
                if str(item.get(key, "")).strip()
            )
        else:
            reasons.append(str(item))
    hint = str(critique.get("revision_hint", "")).strip()
    if hint:
        reasons.append(hint)
    evaluation = critique.get("evaluation")
    if isinstance(evaluation, Mapping):
        reasons.extend(_critique_reasons(evaluation))
    return reasons


def _dedupe_prompt_errors(errors: Sequence[Any]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in errors:
        if not isinstance(item, Mapping):
            item = {"type": "wrong_attribute", "evidence": str(item), "prompt_span": ""}
        normalized = {
            "type": str(item.get("type", "wrong_attribute")),
            "evidence": str(item.get("evidence", "")),
            "prompt_span": str(item.get("prompt_span", "")),
        }
        key = (
            normalized["type"].lower(),
            normalized["evidence"].lower(),
            normalized["prompt_span"].lower(),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(normalized)
    return deduped


def _generator_negative_prompt(image_generator: ImageGenerator) -> str | None:
    return _clean_optional_text(getattr(image_generator, "negative_prompt", None))


def _generator_generation_metadata(image_generator: ImageGenerator) -> list[dict[str, Any]]:
    raw = getattr(image_generator, "last_metadata", None)
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    metadata: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, Mapping):
            metadata.append(deepcopy(dict(item)))
    return metadata


def _expand_image_prompts(
    prompts: Sequence[str],
    image_paths: Sequence[str],
    generator_metadata: Sequence[Mapping[str, Any]],
) -> list[str]:
    if len(prompts) == len(image_paths):
        return [str(item) for item in prompts]
    if len(generator_metadata) == len(image_paths):
        metadata_prompts = [
            str(item.get("prompt") or "").strip() for item in generator_metadata
        ]
        if all(metadata_prompts):
            return metadata_prompts
    if len(prompts) == 1:
        return [str(prompts[0]) for _ in image_paths]
    return [
        str(prompts[min(index, len(prompts) - 1)])
        for index, _ in enumerate(image_paths)
    ]


def _normalize_repair_plan_contract(
    plan: dict[str, Any],
    critique: Mapping[str, Any],
    *,
    can_regenerate: bool,
) -> None:
    selected_action = str(plan.get("selected_action") or plan.get("primary_action") or "none")
    plan["selected_action"] = selected_action
    plan.setdefault("primary_action", selected_action)
    plan["error_type"] = str(plan.get("error_type") or _repair_error_type(critique))
    if not plan.get("fallback_action"):
        plan["fallback_action"] = "regenerate" if can_regenerate and selected_action != "regenerate" else "none"
    if not plan.get("reason"):
        plan["reason"] = "rule-based repair planner selected action from evaluator feedback"


def _repair_plan_needs_clarification(repair_plan: Mapping[str, Any] | None) -> bool:
    if not isinstance(repair_plan, Mapping):
        return False
    route = str(repair_plan.get("typed_route") or "").strip()
    if route == "unverifiable_rare_word_or_clarify":
        return True
    preconditions = repair_plan.get("preconditions")
    return isinstance(preconditions, Mapping) and bool(preconditions.get("needs_clarification"))


def _repair_error_type(critique: Mapping[str, Any]) -> str:
    for key in ("constraint_check", "evaluation"):
        nested = critique.get(key)
        if isinstance(nested, Mapping):
            error_type = _repair_error_type(nested)
            if error_type != "unknown":
                return error_type
    errors = _list_mapping_records(critique.get("errors"))
    if not errors:
        return "unknown"
    for item in errors:
        raw = str(item.get("error_type") or item.get("type") or "").lower()
        if raw:
            return normalize_error_type(raw)
    return "unknown"


def _strip_layout_runtime(layout: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in dict(layout).items()
        if key not in {"request", "raw_response"}
    }


def _prompt_has_layout_guidance(prompt: str) -> bool:
    lowered = prompt.lower()
    return "cinematic composition" in lowered or "layout-guided composition" in lowered


def _coerce_canvas_size(value: tuple[int, int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("layout_canvas_size must be (width, height)")
    width, height = int(value[0]), int(value[1])
    if width <= 0 or height <= 0:
        raise ValueError("layout_canvas_size values must be positive")
    return width, height


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strip_large_prompt(record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(record))
    for key in ("request", "raw_response"):
        if key in result and isinstance(result[key], str) and len(result[key]) > 300:
            result[key] = result[key][:300] + "..."
    return result


def _last_score(round_records: Sequence[Mapping[str, Any]]) -> float | None:
    if not round_records:
        return None
    feedback = round_records[-1].get("feedback")
    if not isinstance(feedback, Mapping):
        return None
    try:
        return float(feedback.get("score"))
    except (TypeError, ValueError):
        return None


def _last_selected_image(round_records: Sequence[Mapping[str, Any]]) -> str | None:
    if not round_records:
        return None
    selected = round_records[-1].get("selected_image")
    return str(selected) if selected else None


def _best_final_selection(
    round_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not round_records:
        return None
    ranked: list[dict[str, Any]] = []
    for fallback_index, record in enumerate(round_records):
        if not isinstance(record, Mapping):
            continue
        selected_image = str(record.get("selected_image") or "").strip()
        if not selected_image:
            continue
        feedback = record.get("feedback")
        feedback = feedback if isinstance(feedback, Mapping) else {}
        gate = feedback.get("completion_gate")
        gate = gate if isinstance(gate, Mapping) else {}
        constraint_check = feedback.get("constraint_check")
        constraint_check = constraint_check if isinstance(constraint_check, Mapping) else {}
        score = _coerce_float(
            gate.get("score"),
            default=_coerce_float(feedback.get("score"), default=0.0),
        )
        hard_pass = _question_level_constraints_passed(constraint_check)
        completion_pass = gate.get("passed") is True
        ranked.append(
            {
                "round": int(record.get("round", fallback_index)),
                "prompt": str(record.get("prompt") or ""),
                "selected_image": selected_image,
                "score": score,
                "completion_passed": completion_pass,
                "constraint_passed": constraint_check.get("passed"),
                "constraint_score": constraint_check.get("score"),
                "hard_pass": hard_pass,
                "reason": (
                    "best question-level hard-constraint pass"
                    if hard_pass
                    else "best completion-gate pass"
                    if completion_pass
                    else "highest available completion score"
                ),
            }
        )
    if not ranked:
        return None
    return sorted(
        ranked,
        key=lambda item: (
            1 if item["completion_passed"] else 0,
            1 if item["hard_pass"] else 0,
            float(item["score"]),
            -int(item["round"]),
        ),
        reverse=True,
    )[0]


def _clean_mode(value: str) -> str:
    value = str(value or "").strip().lower()
    if value not in {"mock", "api", "local"}:
        raise ValueError("mode must be one of: mock, api, local")
    return value


def _coerce_score_threshold(value: float) -> float:
    threshold = float(value)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("score_threshold must be between 0 and 1")
    return threshold
