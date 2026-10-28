"""Chronological prequential comparisons on explicitly labeled feedback streams."""

import math
from dataclasses import dataclass
from datetime import datetime

from river import metrics

from resumeanalysis.recommendation.learner import LearnerConfig, PreferenceLearner


@dataclass
class FeedbackRound:
    timestamp: datetime
    user_id: str
    profile: object
    vacancies: list
    interested: list[bool]
    observed_index: int
    phase: str


def ndcg(labels, scores, k=3, identifiers=None):
    def dcg(order):
        return sum(
            float(labels[index]) / math.log2(rank + 2) for rank, index in enumerate(order[:k])
        )

    ideal = dcg(sorted(range(len(labels)), key=lambda i: -labels[i]))
    return (
        dcg(
            sorted(
                range(len(scores)), key=lambda i: (-scores[i], identifiers[i] if identifiers else i)
            )
        )
        / ideal
        if ideal
        else None
    )


def prequential_evaluate(rounds, builders, *, config=None):
    if not builders or {"non_learning", "content"} & set(builders):
        raise ValueError("Supply at least one uniquely named online feature builder")
    config = config or LearnerConfig()
    names = ["non_learning", "content", *builders]
    learners = {name: {} for name in builders}
    histories = {name: {} for name in builders}
    reports = {
        name: {
            "loss": metrics.LogLoss(),
            "accuracy": metrics.Accuracy(),
            "brier": 0.0,
            "ranking": [],
        }
        for name in names
    }
    traces = []
    previous = None
    observed_pairs = set()
    for step, row in enumerate(rounds):
        if row.timestamp.tzinfo is None or (previous is not None and row.timestamp <= previous):
            raise ValueError(
                "Feedback rounds must have strictly increasing timezone-aware timestamps"
            )
        previous = row.timestamp
        if row.user_id != row.profile.user_id or len(row.vacancies) != len(row.interested):
            raise ValueError("Feedback round user/pool mismatch")
        if not 0 <= row.observed_index < len(row.vacancies) or any(
            type(x) is not bool for x in row.interested
        ):
            raise ValueError("Feedback rounds need explicit labels and a valid observed index")
        pair = (row.user_id, row.vacancies[row.observed_index].vacancy_id)
        if pair in observed_pairs:
            raise ValueError(
                "Duplicate user/vacancy feedback in evaluation; resolve corrections explicitly"
            )
        observed_pairs.add(pair)
        features = {
            name: [builder.build(row.profile, vacancy) for vacancy in row.vacancies]
            for name, builder in builders.items()
        }
        if any(
            item.constraints.status == "blocked" for values in features.values() for item in values
        ):
            raise ValueError("Evaluation pool must respect independent hard constraints")
        base_features = next(iter(features.values()))
        scores = {
            "non_learning": [0.5] * len(row.vacancies),
            "content": [f.snapshot.content_score for f in base_features],
        }
        raw = {}
        for name, values in features.items():
            learner = learners[name].setdefault(row.user_id, PreferenceLearner(config))
            weight = config.preference_weight * min(1.0, learner.count / config.warmup_feedback)
            raw[name] = [learner.predict_proba_one(f.snapshot.values) for f in values]
            scores[name] = [
                (1 - weight) * f.snapshot.content_score + weight * p
                for f, p in zip(values, raw[name])
            ]
        # Every method predicts/ranks this pool before the one observed label is revealed.
        label, observed = row.interested[row.observed_index], row.observed_index
        trace = {
            "step": step,
            "timestamp": row.timestamp.isoformat(),
            "user_id": row.user_id,
            "phase": row.phase,
            "vacancy_id": row.vacancies[observed].vacancy_id,
            "observed_label": label,
            "scores_before_update": {name: s[observed] for name, s in scores.items()},
            "raw_preference_before_update": {name: s[observed] for name, s in raw.items()},
            "ndcg_at_3": {
                name: ndcg(row.interested, s, identifiers=[v.vacancy_id for v in row.vacancies])
                for name, s in scores.items()
            },
        }
        traces.append(trace)
        for name, values in scores.items():
            p = values[observed]
            reports[name]["loss"].update(label, {True: p, False: 1 - p})
            reports[name]["accuracy"].update(label, p >= 0.5)
            reports[name]["brier"] += (float(label) - p) ** 2
            if trace["ndcg_at_3"][name] is not None:
                reports[name]["ranking"].append(trace["ndcg_at_3"][name])
        for name, values in features.items():
            history = histories[name].setdefault(row.user_id, [])
            history.append((values[observed].snapshot.values, label))
            if len(history) > config.history_window:
                learner = PreferenceLearner(config)
                for x, y in history[-config.history_window :]:
                    learner.learn_one(x, y)
                learners[name][row.user_id] = learner
            else:
                learners[name][row.user_id].learn_one(values[observed].snapshot.values, label)
    if not traces:
        raise ValueError("Feedback evaluation stream is empty")
    output = {}
    for name, report in reports.items():
        output[name] = {
            "count": len(traces),
            "log_loss": report["loss"].get(),
            "accuracy_at_0_5": report["accuracy"].get(),
            "brier": report["brier"] / len(traces),
            "ndcg_at_3": sum(report["ranking"]) / len(report["ranking"])
            if report["ranking"]
            else None,
        }
        output[name]["phases"] = {}
        for phase in sorted({t["phase"] for t in traces}):
            values = [
                t["ndcg_at_3"][name]
                for t in traces
                if t["phase"] == phase and t["ndcg_at_3"][name] is not None
            ]
            output[name]["phases"][phase] = {
                "ndcg_at_3": sum(values) / len(values) if values else None,
                "evaluated_pools": len(values),
            }
    return {
        "methods": output,
        "trace": traces,
        "interpretation": "Prequential metrics on the supplied stream; scores are uncalibrated. Only the observed explicit label updates each user's learner.",
    }
