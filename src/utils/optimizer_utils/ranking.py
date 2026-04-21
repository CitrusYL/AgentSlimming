from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class MetricRanking:
    name: str
    scores: Mapping[str, float]
    reverse: bool = True
    missing_value: float = float("-inf")


@dataclass(frozen=True)
class RankingResult:
    fused: list[tuple[str, float]]
    rankings: dict[str, list[str]]
    ranks: dict[str, dict[str, int]]
    scores: dict[str, dict[str, float]]

    @property
    def candidates(self) -> list[str]:
        return [node_id for node_id, _ in self.fused]

    def detail(self, candidate: str | None = None, tried_candidates: list[str] | None = None) -> dict:
        payload = {
            "candidate": candidate,
            "rrf_score": self.score_for(candidate) if candidate else None,
            "tried_candidates": tried_candidates or [],
            "rrf_fused_top10": [
                {"node": node_id, "score": score}
                for node_id, score in self.fused[:10]
            ],
        }
        for name, ranking in self.rankings.items():
            payload[f"{name}_scores"] = self.scores[name]
            payload[f"{name}_ranking"] = [
                {"node": node_id, "rank": index + 1}
                for index, node_id in enumerate(ranking)
            ]
        return payload

    def score_for(self, node_id: str | None) -> float | None:
        if node_id is None:
            return None
        return dict(self.fused).get(node_id)


def reciprocal_rank_fusion(
    metrics: list[MetricRanking],
    weights: Mapping[str, float],
    excluded: set[str] | None = None,
    k: int | None = None,
) -> RankingResult:
    excluded = excluded or set()
    candidate_ids = set().union(*(set(metric.scores) for metric in metrics)) - excluded
    if not candidate_ids:
        return RankingResult(fused=[], rankings={}, ranks={}, scores={})

    k = k or max(10, len(candidate_ids))
    rankings = {}
    ranks = {}
    scores = {}
    for metric in metrics:
        ordered = sorted(
            candidate_ids,
            key=lambda node_id: metric.scores.get(node_id, metric.missing_value),
            reverse=metric.reverse,
        )
        rankings[metric.name] = ordered
        ranks[metric.name] = {node_id: index + 1 for index, node_id in enumerate(ordered)}
        scores[metric.name] = {
            node_id: metric.scores.get(node_id)
            for node_id in ordered
        }

    fused_scores = {}
    for node_id in candidate_ids:
        fused_scores[node_id] = sum(
            weights.get(name, 0.0) / (k + metric_ranks[node_id])
            for name, metric_ranks in ranks.items()
        )

    fused = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return RankingResult(fused=fused, rankings=rankings, ranks=ranks, scores=scores)
