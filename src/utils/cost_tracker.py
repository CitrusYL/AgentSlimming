from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from src.utils.logs import logger


class LLMCostTracker(BaseModel):
    total_cost: float = Field(default=0.0)
    call_count: int = Field(default=0)
    token_usage: Dict[str, int] = Field(default_factory=dict)
    cost_details: Dict[str, float] = Field(default_factory=dict)
    usage_history: list[Dict[str, Any]] = Field(default_factory=list)

    def add_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        node_id: str | None = None,
        input_price: Optional[float] = None,
        output_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        if input_price is None or output_price is None:
            raise ValueError(
                f"Missing pricing for model {model_name}. "
                "Set input_price and output_price in config/config.yaml."
            )

        input_cost = (input_tokens / 1000) * input_price
        output_cost = (output_tokens / 1000) * output_price
        total_cost = input_cost + output_cost

        record = {
            "model": model_name,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "prices": {
                "input_price": input_price,
                "output_price": output_price,
            },
        }
        if node_id is not None:
            record["node_id"] = node_id
            self.cost_details[node_id] = self.cost_details.get(node_id, 0.0) + total_cost

        self.total_cost += total_cost
        self.call_count += 1
        self.token_usage["input_tokens"] = self.token_usage.get("input_tokens", 0) + input_tokens
        self.token_usage["output_tokens"] = self.token_usage.get("output_tokens", 0) + output_tokens
        self.usage_history.append(record)
        return record

    def add_llm_call(
        self,
        node_id: str,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        input_price: Optional[float] = None,
        output_price: Optional[float] = None,
    ) -> float:
        record = self.add_usage(
            model_name,
            input_tokens,
            output_tokens,
            node_id=node_id,
            input_price=input_price,
            output_price=output_price,
        )
        logger.debug(
            f"[COST] Node {node_id}: ${record['total_cost']:.6f} "
            f"({model_name}, {input_tokens}+{output_tokens} tokens)"
        )
        return record["total_cost"]

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_input_tokens": self.token_usage.get("input_tokens", 0),
            "total_output_tokens": self.token_usage.get("output_tokens", 0),
            "total_tokens": (
                self.token_usage.get("input_tokens", 0)
                + self.token_usage.get("output_tokens", 0)
            ),
            "total_cost": self.total_cost,
            "call_count": self.call_count,
            "history": self.usage_history,
        }

    def get_cost_summary(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_calls": self.call_count,
            "total_tokens": (
                self.token_usage.get("input_tokens", 0)
                + self.token_usage.get("output_tokens", 0)
            ),
            "input_tokens": self.token_usage.get("input_tokens", 0),
            "output_tokens": self.token_usage.get("output_tokens", 0),
            "cost_by_node": {
                key: round(value, 6) for key, value in self.cost_details.items()
            },
            "average_cost_per_call": round(self.total_cost / max(self.call_count, 1), 6),
        }
