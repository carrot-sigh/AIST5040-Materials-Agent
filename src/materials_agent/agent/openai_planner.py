"""OpenAI Responses API planner with strict output validation and bounded retries."""
from __future__ import annotations

import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from materials_agent.contracts import Action, ActionValidationError, State, validate_action

from .prompts import SYSTEM_PROMPT, request_fingerprint, serialize_state

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action_type": {"type": "string", "enum": ["relax", "stop"]},
        "candidate_id": {"type": ["string", "null"]},
        "justification": {"type": "string"},
    },
    "required": ["action_type", "candidate_id", "justification"],
    "additionalProperties": False,
}


class OpenAIPlannerError(RuntimeError):
    """The planner could not obtain and validate a response within its retry bound."""


@dataclass(frozen=True)
class PlannerConfig:
    model: str
    proxy_url: str = "http://127.0.0.1:17897"
    max_attempts: int = 3
    max_output_tokens: int = 500
    temperature: float | None = None
    reasoning_effort: str | None = None
    input_price_per_million: float | None = None
    output_price_per_million: float | None = None
    cached_input_price_per_million: float | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.max_attempts < 1 or self.max_attempts > 5:
            raise ValueError("max_attempts must be between 1 and 5")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        for price in (self.input_price_per_million, self.output_price_per_million, self.cached_input_price_per_million):
            if price is not None and (not math.isfinite(price) or price < 0):
                raise ValueError("configured token prices must be non-negative")
        if self.temperature is not None and (not math.isfinite(self.temperature) or self.temperature < 0):
            raise ValueError("temperature must be finite and non-negative")

    @classmethod
    def from_env(cls) -> "PlannerConfig":
        model = os.getenv("OPENAI_MODEL")
        if not model:
            raise ValueError("OPENAI_MODEL is required; configure a model enabled for the API account")

        def optional_float(name: str) -> float | None:
            value = os.getenv(name)
            return float(value) if value not in (None, "") else None

        return cls(
            model=model,
            proxy_url=os.getenv("OPENAI_PROXY_URL", "http://127.0.0.1:17897"),
            max_attempts=int(os.getenv("OPENAI_MAX_ATTEMPTS", "3")),
            max_output_tokens=int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "500")),
            temperature=optional_float("OPENAI_TEMPERATURE"),
            reasoning_effort=os.getenv("OPENAI_REASONING_EFFORT") or None,
            input_price_per_million=optional_float("OPENAI_INPUT_PRICE_PER_1M"),
            output_price_per_million=optional_float("OPENAI_OUTPUT_PRICE_PER_1M"),
            cached_input_price_per_million=optional_float("OPENAI_CACHED_INPUT_PRICE_PER_1M"),
        )

    def public_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "proxy_url": self.proxy_url,
            "max_attempts": self.max_attempts,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million,
            "cached_input_price_per_million": self.cached_input_price_per_million,
        }


class OpenAIPlanner:
    """A one-response-at-a-time policy implementing ``Planner.select_action``."""

    def __init__(
        self,
        config: PlannerConfig | None = None,
        *,
        client: Any | None = None,
        request_log_path: Path | None = None,
    ):
        self.config = config or PlannerConfig.from_env()
        self.last_rationale = ""
        self.round_number = 0
        self.request_log_path = request_log_path
        self.request_records: list[dict[str, Any]] = []
        self._http_client = None
        if client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required to initialize OpenAIPlanner")
            try:
                import httpx
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError("Install the optional LLM dependencies with pip install -e '.[llm]'") from exc
            self._http_client = httpx.Client(http2=False, proxy=self.config.proxy_url)
            client = OpenAI(api_key=api_key, http_client=self._http_client, max_retries=0)
        self.client = client

    def close(self) -> None:
        if self._http_client is not None:
            self._http_client.close()
            self._http_client = None

    def __enter__(self) -> "OpenAIPlanner":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _parameters(self, available_ids: list[str]) -> dict[str, Any]:
        schema = deepcopy(ACTION_SCHEMA)
        schema["properties"]["candidate_id"]["enum"] = [*available_ids, None]
        params: dict[str, Any] = {
            "model": self.config.model,
            "instructions": SYSTEM_PROMPT,
            "text": {"format": {
                "type": "json_schema",
                "name": "materials_candidate_action",
                "strict": True,
                "schema": schema,
            }},
            "max_output_tokens": self.config.max_output_tokens,
        }
        if self.config.temperature is not None:
            params["temperature"] = self.config.temperature
        if self.config.reasoning_effort is not None:
            params["reasoning"] = {"effort": self.config.reasoning_effort}
        return params

    @staticmethod
    def _response_id(response: Any) -> str | None:
        value = getattr(response, "id", None)
        return str(value) if value is not None else None

    def _record(self, record: dict[str, Any]) -> None:
        self.request_records.append(record)
        if self.request_log_path is not None:
            self.request_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.request_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    def _usage_record(self, response: Any) -> dict[str, Any]:
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", None) if usage else None
        output_tokens = getattr(usage, "output_tokens", None) if usage else None
        details = getattr(usage, "input_tokens_details", None)
        cached = getattr(details, "cached_tokens", None)
        record: dict[str, Any] = {
            "cached_input_tokens": cached,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            "input_cost_usd": None,
            "output_cost_usd": None,
            "total_cost_usd": None,
            "pricing_configured": (
                self.config.input_price_per_million is not None
                and self.config.output_price_per_million is not None
            ),
        }
        if record["pricing_configured"] and input_tokens is not None and output_tokens is not None:
            cached_rate = self.config.cached_input_price_per_million
            discounted = (cached or 0) if cached_rate is not None else 0
            record["input_cost_usd"] = (
                (input_tokens - discounted) * self.config.input_price_per_million
                + discounted * (cached_rate or 0)
            ) / 1_000_000
            record["output_cost_usd"] = output_tokens * self.config.output_price_per_million / 1_000_000
            record["total_cost_usd"] = record["input_cost_usd"] + record["output_cost_usd"]
        return record

    def select_action(self, state: State) -> Action:
        self.round_number += 1
        self.last_rationale = ""
        evaluated = state.evaluated_candidate_ids
        available_ids = [c.candidate_id for c in state.candidates if c.candidate_id not in evaluated]
        state_json = serialize_state(state)
        params = self._parameters(available_ids)
        state_payload = json.loads(state_json)
        params["input"] = state_json
        fingerprint = request_fingerprint(SYSTEM_PROMPT, state_json)
        errors = []

        for attempt in range(1, self.config.max_attempts + 1):
            record: dict[str, Any] = {
                "round": self.round_number,
                "attempt": attempt,
                "model": self.config.model,
                "parameters": {key: value for key, value in params.items() if key != "input"},
                "request_sha256": fingerprint,
                "input_bytes": len(state_json.encode("utf-8")),
                "candidate_count": len(available_ids),
                "geometry_candidate_count": sum(
                    c.get("features", {}).get("structure_feature_version") == "geometry-v1"
                    for c in state_payload["available_candidates"]
                ),
                "history_count": len(state.history),
            }
            response = None
            started = perf_counter()
            try:
                response = self.client.responses.create(**params)
                record["response_model"] = getattr(response, "model", None)
                record["response_status"] = getattr(response, "status", None)
                record["response_id"] = self._response_id(response)
                request_id = getattr(response, "_request_id", None)
                record["request_id"] = str(request_id) if request_id else None
                record.update(self._usage_record(response))
                if record["response_status"] not in (None, "completed"):
                    raise ValueError(f'response status: {record["response_status"]}')
                parsed = json.loads(response.output_text)
                if not isinstance(parsed, dict) or set(parsed) != {"action_type", "candidate_id", "justification"}:
                    raise ValueError("response must contain exactly action_type, candidate_id, and justification")
                if not isinstance(parsed["justification"], str) or not parsed["justification"].strip():
                    raise ValueError("justification must be a non-empty string")
                action = Action(parsed["action_type"], parsed["candidate_id"])
                validate_action(state, action)
                rationale = " ".join(parsed["justification"].split())[:1000]
                self.last_rationale = rationale
                record.update(outcome="accepted", action=action.to_dict(), rationale=rationale)
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    error_text = error_text.replace(api_key, "[REDACTED]")
                errors.append(error_text)
                record["outcome"] = "rejected_or_api_error"
                record["error"] = errors[-1]
                if isinstance(exc, ActionValidationError):
                    record["error_category"] = "candidate_constraint"
                    record["proposed_action"] = action.to_dict()
                else:
                    record["error_category"] = "api_error" if response is None else "response_validation"
                cause = exc.__cause__
                if cause is not None:
                    record["cause_type"] = type(cause).__name__
                if response is not None:
                    record.setdefault("response_id", self._response_id(response))
                    record.update(self._usage_record(response))
            record["elapsed_seconds"] = perf_counter() - started
            self._record(record)
            if record["outcome"] == "accepted":
                return action

        raise OpenAIPlannerError(
            f"OpenAI planner failed after {self.config.max_attempts} attempt(s): " + "; ".join(errors)
        )
