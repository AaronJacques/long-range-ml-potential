"""Learning-rate and loss-weight schedules of the training (thesis Eq. 4.22 and Eq. 4.10).

Learning rate (Eq. 4.22)
------------------------
Adam is driven by an exponentially decaying learning rate,

    η(t) = η_init · 0.97^(t / s),

with the optimizer step t and the decay steps s (Tab. 5.2). The legacy code
(``Training.py:176-181``) and therefore this package decay in discrete steps,
``0.97^floor(t / s)`` (``train.lr_staircase = true``, deviation A07); the
continuous form of Eq. 4.22 is available with ``lr_staircase = false``.

Loss weights (Eq. 4.10, experimental force mode only)
-----------------------------------------------------
The prefactors p_E and p_F of the combined energy/force loss move from a start
value to a limit (``Training.py:14-16``)::

    p(t) = limit · (1 − f) + start · f,    f = rate^(t / decay_steps).

Step k of the training (1-based) uses p(k − 1), i.e. the first step uses the
start value. :func:`loss_weight` is the plain Python form, :func:`loss_weight_tf`
the graph form evaluated from ``optimizer.iterations`` inside the compiled
training step; both compute in float64 and agree to rounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import keras

from lrmlp.config import TrainConfig

if TYPE_CHECKING:
    import tensorflow as tf

__all__ = ["learning_rate_schedule", "loss_weight", "loss_weight_tf"]


def learning_rate_schedule(train: TrainConfig) -> keras.optimizers.schedules.ExponentialDecay:
    """Return the learning-rate schedule of Eq. 4.22.

    Parameters
    ----------
    train : TrainConfig
        Uses ``learning_rate`` (η_init), ``lr_decay_steps`` (s), ``lr_decay_rate``
        (0.97) and ``lr_staircase`` (stepwise decay, deviation A07).

    Returns
    -------
    keras.optimizers.schedules.ExponentialDecay
        ``η_init · rate^(t / s)`` (``floor(t / s)`` with ``lr_staircase``), evaluated
        in float32 at the optimizer step t, exactly as the legacy
        ``tf.keras.optimizers.schedules.ExponentialDecay`` (``Training.py:176-181``).
    """
    return keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=train.learning_rate,
        decay_steps=train.lr_decay_steps,
        decay_rate=train.lr_decay_rate,
        staircase=train.lr_staircase,
    )


def loss_weight(
    step: float, start: float, limit: float, decay_steps: float, decay_rate: float
) -> float:
    """Loss weight p(t) of the force mode (Eq. 4.10), Python form.

    Parameters
    ----------
    step : float
        Optimizer step t (number of updates already applied).
    start : float
        Value at t = 0 (``p_*_start``).
    limit : float
        Value for t → ∞ (``p_*_limit``).
    decay_steps : float
        Decay steps (``train.loss_weight_decay_steps``).
    decay_rate : float
        Decay rate (``train.loss_weight_decay_rate``).

    Returns
    -------
    float
        ``limit · (1 − f) + start · f`` with ``f = decay_rate^(step / decay_steps)``;
        the expression of ``Training.get_weighting_from_step`` (``Training.py:14-16``).
    """
    f = decay_rate ** (step / decay_steps)
    return float(limit * (1 - f) + start * f)


def loss_weight_tf(
    step: Any, start: float, limit: float, decay_steps: float, decay_rate: float
) -> tf.Tensor:
    """Loss weight p(t) of the force mode (Eq. 4.10), graph form.

    Parameters
    ----------
    step : tf.Tensor or int
        Optimizer step t, typically ``optimizer.iterations`` (int64 variable).
    start, limit, decay_steps, decay_rate : float
        See :func:`loss_weight`.

    Returns
    -------
    tf.Tensor
        float64 scalar, equal to :func:`loss_weight` up to rounding. Cast it to
        float32 for the loss, as the legacy code did with ``tf.constant(p, float32)``.
    """
    import tensorflow as tf

    t = tf.cast(step, tf.float64)
    f = tf.pow(tf.constant(decay_rate, tf.float64), t / tf.constant(decay_steps, tf.float64))
    return tf.constant(limit, tf.float64) * (1.0 - f) + tf.constant(start, tf.float64) * f
