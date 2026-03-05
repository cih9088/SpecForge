"""
AIME 2025 benchmark
"""

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .aime import AIMEBenchmarker
from .registry import BENCHMARKS


@BENCHMARKS.register("aime25")
class AIME25Benchmarker(AIMEBenchmarker):
    """AIME 2025 benchmark implementation."""

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """Load and preprocess AIME25 dataset."""
        dataset = load_dataset("math-ai/aime25")["test"]
        questions = []
        labels = []
        for idx, q in enumerate(dataset):
            if self.num_samples is not None and idx >= self.num_samples:
                break

            questions.append({"question": q["problem"]})
            labels.append(str(q["answer"]).strip())
        return questions, labels
