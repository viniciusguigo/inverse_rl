import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv


from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.imitation_learning import GAIL
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts, load_experts

import sys
sys.path.append('../hri_airsim/')
import hri_airsim

def main():
    env = TfEnv(GymEnv('HRI_AirSim_Landing-v0', record_video=False, record_log=False))
    
    ### VGG 11/29/18: Added support to CSV files
    ## this method loads expert data saved as pickle file
    # experts = load_latest_experts('data/airsim_final', n=1)
    # this one uses csv:
    experts = load_experts('data/airsim_human_data/log.csv', pickle_format=False)

    irl_model = GAIL(env_spec=env.spec, expert_trajs=experts)
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=5000,
        batch_size=60,
        max_path_length=60,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=100,
        irl_model_wt=1.0,
        entropy_weight=0.0, # GAIL should not use entropy unless for exploration
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
        n_parallel=0
    )

    with rllab_logdir(algo=algo, dirname='data/airsim_gail'):
        with tf.Session():
            algo.train()

if __name__ == "__main__":
    main()
