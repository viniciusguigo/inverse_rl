import os
import random
import joblib
import json
import contextlib

import rllab.misc.logger as rllablogger
import tensorflow as tf
import numpy as np

from inverse_rl.utils.hyperparametrized import extract_hyperparams

@contextlib.contextmanager
def rllab_logdir(algo=None, dirname=None):
    if dirname:
        rllablogger.set_snapshot_dir(dirname)
    dirname = rllablogger.get_snapshot_dir()
    rllablogger.add_tabular_output(os.path.join(dirname, 'progress.csv'))
    if algo:
        with open(os.path.join(dirname, 'params.json'), 'w') as f:
            params = extract_hyperparams(algo)
            json.dump(params, f)
    yield dirname
    rllablogger.remove_tabular_output(os.path.join(dirname, 'progress.csv'))


def get_expert_fnames(log_dir, n=5):
    print('Looking for paths')
    import re
    itr_reg = re.compile(r"itr_(?P<itr_count>[0-9]+)\.pkl")

    itr_files = []
    for i, filename in enumerate(os.listdir(log_dir)):
        m = itr_reg.match(filename)
        if m:
            itr_count = m.group('itr_count')
            itr_files.append((itr_count, filename))

    itr_files = sorted(itr_files, key=lambda x: int(x[0]), reverse=True)[:n]
    for itr_file_and_count in itr_files:
        fname = os.path.join(log_dir, itr_file_and_count[1])
        print('Loading %s' % fname)
        yield fname

def parse_episode(log_data, obs_log_data, epi_n):
    """ Return states, actions, rewards from a specific episode.

    Args:
        epi_n : episode numnber

    Return:
        states : position, velocity, acceleration  (linear and angular)
        actions : controls applied
        rewards : rewards received by the agent

    """
    # parse specific episode
    steps = np.where(log_data[:,1] == epi_n)
    steps = np.squeeze(np.asarray(log_data[steps,0])).astype(int)

    # find initial and final index in 

    # parse states and actions
    # hardcoded to take a (15,) vector for obs and (2,) for actions
    states = obs_log_data[steps,1:]
    actions = log_data[steps,3:5]
    rewards = log_data[steps,7]

    return states, actions, rewards


def csv2paths(fname):
    """Convert expert data saved in a CSV file to 'paths', as previously
    defined by the IRL algorithm.
    """
    # load csv file
    log_data = np.genfromtxt(fname, delimiter=',', skip_header=True)
    obs_log_data = np.genfromtxt('data/airsim_human_data/low_obs_log.csv', delimiter=',', skip_header=True)
    n_epis = int(np.max(log_data[:,1]))
    print('Found {} episodes of expert data.'.format(n_epis))

    # loop for each episode, parse observations and actions in a dict
    paths = []
    for i in range(n_epis):
        observations, actions, returns = parse_episode(log_data, obs_log_data, i)

        path = {'observations': observations,
                'actions': actions,
                'returns': returns}
        paths.append(path)

    return paths

def load_experts(fname, max_files=float('inf'), min_return=None, pickle_format=True):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    ### VGG: add option to load csv files instead of pickle
    if pickle_format:
        if hasattr(fname, '__iter__'):
            paths = []
            for fname_ in fname:
                tf.reset_default_graph()
                with tf.Session(config=config):
                    snapshot_dict = joblib.load(fname_)
                paths.extend(snapshot_dict['paths'])
        else:
            with tf.Session(config=config):
                snapshot_dict = joblib.load(fname)
            paths = snapshot_dict['paths']
    else:
        # parse csv file
        paths = csv2paths(fname)
        
    tf.reset_default_graph()

    trajs = []
    for path in paths:
        obses = path['observations']
        actions = path['actions']
        returns = path['returns']
        total_return = np.sum(returns)

        # # investigate shape of paths
        # print('obses: ', obses.shape)
        # print('actions: ', actions.shape)
        # print('returns: ', returns.shape)
        # print('total_return: ', total_return.shape)

        if (min_return is None) or (total_return >= min_return):
            traj = {'observations': obses, 'actions': actions}
            trajs.append(traj)
    random.shuffle(trajs)
    print('Loaded %d trajectories' % len(trajs))
    return trajs


def load_latest_experts(logdir, n=5, min_return=None):
    return load_experts(get_expert_fnames(logdir, n=n), min_return=min_return)


def load_latest_experts_multiple_runs(logdir, n=5):
    paths = []
    for i, dirname in enumerate(os.listdir(logdir)):
        dirname = os.path.join(logdir, dirname)
        if os.path.isdir(dirname):
            print('Loading experts from %s' % dirname)
            paths.extend(load_latest_experts(dirname, n=n))
    return paths
