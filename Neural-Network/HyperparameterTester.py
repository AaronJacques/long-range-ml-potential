import itertools
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from Training import start_training

LOSS_METRIC = 'loss'
RUN_DIR = 'logs/hparam_tuning'


def hyperparameter_tester():
    session_num = 0

    initial_learning_rate = hp.HParam('initial_learning_rate', hp.Discrete([1e-2, 1e-3, 1e-4]))
    lr_decay_steps = hp.HParam('lr_decay_steps', hp.Discrete([1e+3, 1e+4, 1e+5, 1e+6]))
    decay_steps = hp.HParam('lr_decay_steps', hp.Discrete([1e+3, 1e+4, 1e+5, 1e+6]))
    decay_rate = hp.HParam('decay_rate', hp.Discrete([0.96, 0.97, 0.98, 0.99]))
    p_energy_start = hp.HParam('p_energy_start', hp.Discrete([0.01, 0.05, 0.1]))
    p_energy_limit = hp.HParam('p_energy_limit', hp.Discrete([1.0, 2.0, 3.0]))
    p_force_start = hp.HParam('p_force_start', hp.Discrete([100, 500, 1000]))
    p_force_limit = hp.HParam('p_force_limit', hp.Discrete([1.0, 2.0, 3.0]))

    all_hparams = [initial_learning_rate, lr_decay_steps, decay_steps, decay_rate, p_energy_start,
                   p_energy_limit, p_force_start, p_force_limit]

    with tf.summary.create_file_writer(RUN_DIR).as_default():
        hp.hparams_config(
            hparams=all_hparams,
            metrics=[hp.Metric(LOSS_METRIC, display_name='Energy Loss')],
        )

    for hparams in itertools.product(
            initial_learning_rate.domain.values,
            lr_decay_steps.domain.values,
            decay_steps.domain.values,
            decay_rate.domain.values,
            p_energy_start.domain.values,
            p_energy_limit.domain.values,
            p_force_start.domain.values,
            p_force_limit.domain.values,
    ):
        hparams = {
            initial_learning_rate: hparams[0],
            lr_decay_steps: hparams[1],
            decay_steps: hparams[2],
            decay_rate: hparams[3],
            p_energy_start: hparams[4],
            p_energy_limit: hparams[5],
            p_force_start: hparams[6],
            p_force_limit: hparams[7],
        }
        run_name = "run-%d" % session_num
        print('--- Starting trial: %s' % run_name)
        print({h.name: hparams[h] for h in hparams})
        run(RUN_DIR, hparams)
        session_num += 1


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        loss = start_training(hparams)
        tf.summary.scalar(LOSS_METRIC, loss, step=1)


