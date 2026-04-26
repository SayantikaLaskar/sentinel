"""LLM Agent wrapper for SENTINEL."""
from __future__ import annotations

import logging
from typing import Any

from sentinel.training.action_parser import parse_llm_action, parse_llm_action_result
from sentinel.training.prompt_builder import build_messages, build_prompt

logger = logging.getLogger(__name__)

_DEFAULT_GEN_KWARGS: dict[str, Any] = {
    "max_new_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": 0,
}


class LLMAgent:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        agent_role: str = "holmes",
        device: str = "cuda",
        use_chat_template: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.agent_role = agent_role
        self.device = device
        self.use_chat_template = use_chat_template
        self._step_counter = 0

        gen_kwargs = dict(_DEFAULT_GEN_KWARGS)
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            gen_kwargs["pad_token_id"] = tokenizer.eos_token_id
        self._gen_kwargs = gen_kwargs

    def act(self, obs: dict[str, Any], step: int = 0) -> dict[str, Any]:
        self._step_counter = step
        try:
            raw_output = self._generate(obs, step)
            action, parsed_ok = parse_llm_action_result(
                raw_output,
                fallback_agent=self.agent_role,
            )
            action["_llm_completion"] = raw_output
            action["_parse_failed"] = not parsed_ok
            return action
        except Exception as exc:
            logger.warning(
                "[LLMAgent/%s] Inference error at step %d: %s - using fallback.",
                self.agent_role,
                step,
                exc,
            )
            fallback = self._fallback_action()
            fallback["_parse_failed"] = True
            return fallback

    def reset(self) -> None:
        self._step_counter = 0

    def _generate(self, obs: dict[str, Any], step: int) -> str:
        import torch

        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            messages = build_messages(obs, self.agent_role, step)
            try:
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    continue_final_message=True,
                )
            except TypeError:
                input_text = self.tokenizer.apply_chat_template(
                    messages[:-1],
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            input_text = build_prompt(obs, self.agent_role, step)

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                **self._gen_kwargs,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        if self.use_chat_template:
            raw = "{" + raw

        return _truncate_at_first_object(raw)

    def _fallback_action(self) -> dict[str, Any]:
        if self.agent_role == "forge":
            return {
                "agent": "forge",
                "category": "remediation",
                "name": "ScaleService",
                "params": {"service": "api-gateway", "replicas": 2},
            }
        if self.agent_role == "hermes":
            return {
                "agent": "hermes",
                "category": "deployment",
                "name": "Rollback",
                "params": {"service": "api-gateway"},
            }
        if self.agent_role == "oracle":
            return {
                "agent": "oracle",
                "category": "meta",
                "name": "EscalateToHuman",
                "params": {"reason": "Unable to produce valid meta action"},
            }
        return {
            "agent": "holmes",
            "category": "investigative",
            "name": "QueryLogs",
            "params": {"service": "api-gateway", "time_range": [0, 300]},
        }


def build_llm_agent(
    model: Any,
    tokenizer: Any,
    agent_role: str = "holmes",
) -> LLMAgent | None:
    if model is None or tokenizer is None:
        return None
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"
    return LLMAgent(model, tokenizer, agent_role=agent_role, device=device)


def make_grpo_reward_fn(env: Any) -> Any:
    def _reward_fn(
        prompts: list[str],
        completions: list[str],
        agent_role: str = "holmes",
        obs: dict | None = None,
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for completion in completions:
            try:
                from sentinel.models import Action

                action = Action(**parse_llm_action(completion, fallback_agent=agent_role))
                if obs is not None and hasattr(env, "reward_function") and env._incident_state is not None:
                    rf = env.reward_function
                    inc = env._incident_state
                    step_reward = rf.compute_step_reward(
                        action,
                        env.world_state,
                        inc,
                        previous_blast_radius=set(inc.current_blast_radius),
                        current_blast_radius=set(inc.current_blast_radius),
                    )
                    rewards.append(float(step_reward))
                else:
                    rewards.append(0.0)
            except Exception as exc:
                logger.debug("GRPO reward fn error: %s", exc)
                rewards.append(-0.1)
        return rewards

    return _reward_fn


def _truncate_at_first_object(raw: str) -> str:
    """Return the substring up to and including the first balanced '}'."""
    start = raw.find("{")
    if start == -1:
        return raw

    depth = 0
    in_string = False
    string_quote = ""
    escape = False

    for i in range(start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch in ('"', "'"):
            if not in_string:
                in_string = True
                string_quote = ch
                continue
            if ch == string_quote:
                in_string = False
                string_quote = ""
                continue
        if in_string:
            continue
        
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[: i + 1]

    return raw
