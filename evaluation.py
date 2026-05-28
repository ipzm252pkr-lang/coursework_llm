from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


FACTS: list[str] = [
    "My name is Alex and I am 28 years old.",
    "I work as a data scientist at a startup in Kyiv.",
    "My favorite programming language is Python.",
    "I have a dog named Max.",
    "In my free time I am learning Japanese.",
]

# Semantically unrelated messages used to push facts out of the buffer window
DISTRACTORS: list[str] = [
    "What is the capital of France?",
    "Tell me about quantum computing.",
    "Explain how neural networks work.",
    "What is the boiling point of water?",
    "Tell me a fun fact about space.",
    "What is blockchain technology?",
    "How does photosynthesis work?",
    "Describe the water cycle briefly.",
    "What is the speed of light?",
    "Tell me about the Roman Empire.",
    "How do airplanes fly?",
    "What is the Pythagorean theorem?",
    "Tell me about climate change.",
    "What is DNA?",
    "How does the internet work?",
]

# question, list of keywords that must appear in a correct answer
RECALL_QUESTIONS: list[tuple[str, list[str]]] = [
    ("What is my name?",                          ["alex"]),
    ("How old am I?",                             ["28"]),
    ("What city do I work in?",                   ["kyiv"]),
    ("What is my job title or profession?",       ["data scientist", "scientist"]),
    ("What is my favorite programming language?", ["python"]),
    ("What is my pet's name?",                    ["max"]),
    ("What language am I learning?",              ["japanese"]),
]


@dataclass
class RetentionResult:
    strategy: str
    n_distractors: int
    retention_score: float
    per_question_scores: list[float]
    per_question_answers: list[str]
    context_sizes: list[int]
    latencies_ms: list[float]

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "n_distractors": self.n_distractors,
            "retention_score": round(self.retention_score, 4),
            "per_question_scores": [round(s, 4) for s in self.per_question_scores],
            "per_question_answers": self.per_question_answers,
            "mean_context_size": round(float(np.mean(self.context_sizes)), 1) if self.context_sizes else 0,
            "mean_latency_ms": round(float(np.mean(self.latencies_ms)), 2) if self.latencies_ms else 0,
        }


@dataclass
class LatencyResult:
    strategy: str
    latencies_ms: list[float]
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "mean_ms": round(self.mean_ms, 2),
            "std_ms": round(self.std_ms, 2),
            "min_ms": round(self.min_ms, 2),
            "max_ms": round(self.max_ms, 2),
            "latencies_ms": [round(v, 2) for v in self.latencies_ms],
        }


@dataclass
class ContextGrowthResult:
    strategy: str
    turns: list[int]
    context_sizes: list[int]

    def to_dict(self) -> dict:
        return {"strategy": self.strategy, "turns": self.turns, "context_sizes": self.context_sizes}


@dataclass
class StatTestResult:
    name_a: str
    name_b: str
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "comparison": f"{self.name_a} vs {self.name_b}",
            "test_name": self.test_name,
            "statistic": round(self.statistic, 4),
            "p_value": round(self.p_value, 4),
            "significant": self.significant,
            "interpretation": self.interpretation,
        }



class Evaluator:

    def __init__(
        self,
        llm_client=None,
        model_name: str = "gemini-3.1-flash",
        sleep_between_requests: float = 2.0,
    ) -> None:
        self._llm = llm_client
        self._model_name = model_name
        self._sleep = sleep_between_requests

    def keyword_recall(self, answer: str, keywords: list[str]) -> float:
        """Fraction of expected keywords found in the answer (case-insensitive)."""
        if not keywords:
            return 0.0
        answer_lower = answer.lower()
        found = sum(1 for kw in keywords if kw.lower() in answer_lower)
        return found / len(keywords)

    def llm_judge_score(self, question: str, answer: str, keywords: list[str]) -> float:

        # Falls back to keyword_recall if the API call fails or returns unparseable output

        if self._llm is None:
            return self.keyword_recall(answer, keywords)

        from google.genai import types

        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n"
            f"Expected information: {', '.join(keywords)}\n\n"
            "Rate how well the answer contains the expected information.\n"
            "Return ONLY a decimal number between 0.0 and 1.0."
        )

        try:
            resp = self._llm.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=10),
            )
            return max(0.0, min(1.0, float(resp.text.strip().replace(",", "."))))
        except Exception as exc:
            logger.warning("llm_judge failed (%s), using keyword_recall", type(exc).__name__)
            return self.keyword_recall(answer, keywords)

    def run_retention_test(
        self,
        chatbot,
        n_distractors: int,
        facts: Optional[list[str]] = None,
        distractors: Optional[list[str]] = None,
        questions: Optional[list[tuple[str, list[str]]]] = None,
    ) -> RetentionResult:
        facts = facts or FACTS
        distractors = (distractors or DISTRACTORS)[:n_distractors]
        questions = questions or RECALL_QUESTIONS

        strategy = type(chatbot._memory).__name__
        logger.info("Retention test: %s | distractors=%d", strategy, n_distractors)

        chatbot.reset()
        context_sizes: list[int] = []
        latencies: list[float] = []

        def _chat(msg: str) -> str:
            return self._safe_chat(chatbot, msg)

        def _record() -> None:
            log = chatbot.get_response_log()
            if log:
                context_sizes.append(log[-1]["context_tokens"])
                latencies.append(log[-1]["latency_ms"])

        for fact in facts:
            _chat(fact); _record(); time.sleep(self._sleep)

        for distractor in distractors:
            _chat(distractor); _record(); time.sleep(self._sleep)

        scores, answers = [], []
        for question, keywords in questions:
            answer = _chat(question)
            scores.append(self.keyword_recall(answer, keywords))
            answers.append(answer)
            latencies.append(chatbot.get_response_log()[-1]["latency_ms"])
            time.sleep(self._sleep)

        score = float(np.mean(scores))
        logger.info("  retention_score=%.4f", score)

        return RetentionResult(
            strategy=strategy,
            n_distractors=n_distractors,
            retention_score=score,
            per_question_scores=scores,
            per_question_answers=answers,
            context_sizes=context_sizes,
            latencies_ms=latencies,
        )

    def run_latency_test(self, chatbot, n_turns: int = 15) -> LatencyResult:
        strategy = type(chatbot._memory).__name__
        logger.info("Latency test: %s | turns=%d", strategy, n_turns)

        chatbot.reset()
        latencies: list[float] = []

        for i in range(n_turns):
            self._safe_chat(chatbot, f"Tell me something interesting about topic {i + 1}.")
            log = chatbot.get_response_log()
            if log:
                latencies.append(log[-1]["latency_ms"])
            time.sleep(self._sleep)

        arr = np.array(latencies)
        return LatencyResult(
            strategy=strategy,
            latencies_ms=latencies,
            mean_ms=float(arr.mean()),
            std_ms=float(arr.std()),
            min_ms=float(arr.min()),
            max_ms=float(arr.max()),
        )

    def run_context_growth_test(self, chatbot, n_turns: int = 25) -> ContextGrowthResult:
        strategy = type(chatbot._memory).__name__
        logger.info("Context growth test: %s | turns=%d", strategy, n_turns)

        chatbot.reset()
        turns, sizes = [], []

        for i in range(n_turns):
            self._safe_chat(chatbot, f"Message {i + 1}: tell me about machine learning.")
            log = chatbot.get_response_log()
            if log:
                turns.append(i + 1)
                sizes.append(log[-1]["context_tokens"])
            time.sleep(self._sleep)

        return ContextGrowthResult(strategy=strategy, turns=turns, context_sizes=sizes)

    def statistical_test(
        self,
        scores_a: list[float],
        scores_b: list[float],
        name_a: str,
        name_b: str,
        alpha: float = 0.05,
    ) -> StatTestResult:
        if len(scores_a) != len(scores_b):
            raise ValueError(f"Sample length mismatch: {len(scores_a)} vs {len(scores_b)}")
        if len(scores_a) < 3:
            raise ValueError(f"Need at least 3 observations, got {len(scores_a)}")

        a, b = np.array(scores_a, dtype=float), np.array(scores_b, dtype=float)

        _, p_a = stats.shapiro(a)
        _, p_b = stats.shapiro(b)
        both_normal = p_a > alpha and p_b > alpha

        if np.allclose(a, b):
            return StatTestResult(
                name_a=name_a, name_b=name_b, test_name="N/A",
                statistic=0.0, p_value=1.0, significant=False,
                interpretation=f"{name_a} and {name_b} produced identical scores.",
            )

        if both_normal:
            stat, p = stats.ttest_rel(a, b)
            test_name = "Paired T-test"
        else:
            stat, p = stats.wilcoxon(a, b)
            test_name = "Wilcoxon Signed-Rank"

        significant = p < alpha
        better = name_a if a.mean() > b.mean() else name_b

        if significant:
            interp = (
                f"Reject H₀ (p={p:.4f} < α={alpha}). "
                f"{better} is significantly better "
                f"(mean {name_a}={a.mean():.4f}, {name_b}={b.mean():.4f})."
            )
        else:
            interp = (
                f"Fail to reject H₀ (p={p:.4f} ≥ α={alpha}). "
                f"No significant difference between {name_a} ({a.mean():.4f}) "
                f"and {name_b} ({b.mean():.4f})."
            )

        return StatTestResult(
            name_a=name_a, name_b=name_b, test_name=test_name,
            statistic=float(stat), p_value=float(p),
            significant=significant, interpretation=interp,
        )

    def compare_all(
        self, scores: dict[str, list[float]], alpha: float = 0.05
    ) -> list[StatTestResult]:
        names = list(scores.keys())
        return [
            self.statistical_test(scores[names[i]], scores[names[j]], names[i], names[j], alpha)
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]

    def _safe_chat(self, chatbot, message: str, max_retries: int = 3) -> str:
        for attempt in range(max_retries):
            answer = chatbot.chat(message)
            if "[ERROR" not in answer:
                return answer
            if attempt < max_retries - 1:
                wait = 60 if "rate limit" in answer.lower() else 10
                logger.warning("API error on attempt %d/%d, retrying in %ds", attempt + 1, max_retries, wait)
                time.sleep(wait)
        return answer
