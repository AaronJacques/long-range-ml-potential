"""Learning-rate and loss-weight schedules against the legacy golden values (T03)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import tensorflow as tf

from lrmlp.config import TrainConfig
from lrmlp.schedules import learning_rate_schedule, loss_weight, loss_weight_tf

try:
    from golden import fixtures as fx
except ImportError:  # pragma: no cover - depends on pytest import mode
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from golden import fixtures as fx

GOLDEN = fx.load_schedules_golden()


def _id(case: dict[str, Any]) -> str:
    return "-".join(f"{k}={case[k]}" for k in case if k not in ("steps", "values"))


@pytest.mark.parametrize("case", GOLDEN["learning_rate"], ids=_id)
def test_learning_rate_matches_golden(case: dict[str, Any]) -> None:
    train = TrainConfig(
        learning_rate=case["initial_learning_rate"],
        lr_decay_steps=case["decay_steps"],
        lr_decay_rate=case["decay_rate"],
        lr_staircase=case["staircase"],
    )
    schedule = learning_rate_schedule(train)
    got = [float(schedule(step)) for step in case["steps"]]
    np.testing.assert_allclose(got, case["values"], rtol=1e-7, atol=0)


def test_learning_rate_from_optimizer_iterations() -> None:
    case = GOLDEN["learning_rate"][0]
    train = TrainConfig(
        learning_rate=case["initial_learning_rate"],
        lr_decay_steps=case["decay_steps"],
        lr_decay_rate=case["decay_rate"],
    )
    schedule = learning_rate_schedule(train)
    step = tf.Variable(0, dtype=tf.int64)
    for s, expected in zip(case["steps"], case["values"], strict=True):
        step.assign(s)
        np.testing.assert_allclose(float(schedule(step)), expected, rtol=1e-7, atol=0)


def test_continuous_learning_rate_follows_eq_4_22() -> None:
    train = TrainConfig(learning_rate=1e-3, lr_decay_steps=20, lr_staircase=False)
    schedule = learning_rate_schedule(train)
    for step in (0, 1, 19, 20, 30, 1000):
        np.testing.assert_allclose(float(schedule(step)), 1e-3 * 0.97 ** (step / 20), rtol=1e-5)


@pytest.mark.parametrize("case", GOLDEN["weighting"], ids=_id)
def test_loss_weight_matches_golden(case: dict[str, Any]) -> None:
    args = (case["start"], case["limit"], case["decay_steps"], case["decay_rate"])
    py = [loss_weight(step, *args) for step in case["steps"]]
    np.testing.assert_allclose(py, case["values"], rtol=1e-7, atol=0)
    graph = [float(loss_weight_tf(tf.constant(step, tf.int64), *args)) for step in case["steps"]]
    np.testing.assert_allclose(graph, case["values"], rtol=1e-7, atol=0)
    np.testing.assert_allclose(graph, py, rtol=1e-7, atol=0)


def test_loss_weight_tf_in_graph_from_variable() -> None:
    case = GOLDEN["weighting"][1]
    args = (case["start"], case["limit"], case["decay_steps"], case["decay_rate"])
    step = tf.Variable(0, dtype=tf.int64)

    @tf.function
    def weight() -> tf.Tensor:
        return loss_weight_tf(step, *args)

    for s, expected in zip(case["steps"], case["values"], strict=True):
        step.assign(s)
        np.testing.assert_allclose(float(weight()), expected, rtol=1e-7, atol=0)
    assert weight.experimental_get_tracing_count() == 1


def test_loss_weight_endpoints() -> None:
    assert loss_weight(0, 5.0, 1.0, 10.0, 0.5) == 5.0
    np.testing.assert_allclose(loss_weight(10, 5.0, 1.0, 10.0, 0.5), 3.0)
    np.testing.assert_allclose(loss_weight(1e6, 5.0, 1.0, 10.0, 0.5), 1.0)
