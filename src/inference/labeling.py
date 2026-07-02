"""Shared post-processing: turn class probabilities into label decisions.

Applies the deployment decision threshold on P(tracciabilita) so every engine
(PyTorch, Triton, TensorRT) behaves identically. Kept dependency-light (numpy only)
so the Triton gateway image, which ships without torch, can import it.

The model is trained ~50/50 but deployed on a population with ~0.3% positives, so a
plain argmax@0.5 over-triggers. `settings.inference.positive_threshold` (tuned on the
gold set) is the operating point; 0.5 reproduces plain argmax for the binary task.
"""

import numpy as np

from src.config import settings

POSITIVE_LABEL = "tracciabilita"


def _positive_id() -> int:
    name_to_id = {name: idx for idx, name in settings.model.label_map.items()}
    # Binary traceability task; fall back to id 1 if the positive label is renamed.
    return name_to_id.get(POSITIVE_LABEL, 1)


def probs_to_results(probs: np.ndarray, threshold: float | None = None) -> list[dict]:
    """Map a (batch, num_labels) probability array to prediction dicts.

    A sample is `tracciabilita` iff ``P(tracciabilita) >= threshold``, else `altro`.
    ``confidence`` is the probability of the *chosen* label; ``positive_prob`` is always
    P(tracciabilita) so downstream code can re-threshold without re-running the model.
    """
    if threshold is None:
        threshold = settings.inference.positive_threshold

    label_map = settings.model.label_map
    pos_id = _positive_id()
    neg_id = next(i for i in label_map if i != pos_id)

    p_pos = probs[:, pos_id]
    results = []
    for pp, row in zip(p_pos, probs):
        label_id = pos_id if pp >= threshold else neg_id
        results.append(
            {
                "label": label_map[label_id],
                "confidence": float(row[label_id]),
                "positive_prob": float(pp),
            }
        )
    return results
