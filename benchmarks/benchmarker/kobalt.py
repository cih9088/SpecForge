"""
KoBALT-700 benchmark
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function


@BENCHMARKS.register("kobalt")
class KoBALTBenchmarker(Benchmarker):
    """KoBALT-700 benchmark implementation."""

    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(num_samples, None)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """Load and preprocess KOBALT-700 dataset."""
        ds = load_dataset("snunlp/KoBALT-700", "kobalt_v1")["raw"]
        questions = []
        labels = []
        for idx, q in enumerate(ds):
            if self.num_samples is not None and idx >= self.num_samples:
                break

            questions.append({"question": q["Question"]})
            labels.append(str(q["Answer"]).strip())
        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        output = output.strip()
        match = re.search(r"정답은 (.)입니다.$", output)
        if match:
            return match.group(1).strip()
        return None

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        if not labels or len(labels) == 0:
            return None
        if all(label is None for label in labels):
            return None

        correct = 0
        valid_count = 0
        for pred, label in zip(predictions, labels):
            if label is not None:
                valid_count += 1
                if pred is not None:
                    # Normalize answers for comparison
                    pred_normalized = str(pred).strip()
                    label_normalized = str(label).strip()
                    # Try exact match first
                    if pred_normalized == label_normalized:
                        correct += 1
                    else:
                        # Try numeric comparison
                        try:
                            pred_num = int(pred_normalized)
                            label_num = int(label_normalized)
                            if pred_num == label_num:
                                correct += 1
                        except ValueError:
                            pass

        return correct / valid_count if valid_count > 0 else 0.0

    def create_sgl_function(self):
        return create_simple_sgl_function(
            function_name="reasoning_gen",
            answer_key="answer",
            system_prompt="당신은 문제를 해결하는 전문가입니다.",
            user_prefix="""
다음 문제에 대해서 충분히 생각하고 추론하여, 10개의 보기(A, B, C, D, E, F, G, H, I, J) 중 정답을 고르세요.

""".lstrip(),
            user_suffix="""

답변은 반드시 다음 형식을 엄격히 지켜야 합니다: "정답은 [정답 보기]입니다." 로 끝나야 하고, [정답 보기]는 A, B, C, D, E, F, G, H, I, J 중 하나여야 합니다.
정답: 문제를 풀기 위해, 한 번 천천히 생각해봅시다.
""".rstrip(),
            max_tokens=self.get_max_new_tokens(),
        )

    def get_max_new_tokens(self) -> int:
        return 32768
