import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv


from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.imitation_learning import AIRLStateAction
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts


def main(eval_reward = False):
    env = TfEnv(GymEnv('Pendulum-v0', record_video=False, record_log=False))
    
    n_experts = 10
    experts = load_latest_experts('plotting/pendulum_final', n=n_experts)
    dirname='data/pendulum' # dir to save logs and images

    irl_model = AIRLStateAction(env_spec=env.spec, expert_trajs=experts)
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(32, 32))
    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=1000,
        batch_size=1000,
        max_path_length=100,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=50,
        irl_model_wt=1.0,
        entropy_weight=0.1, # this should be 1.0 but 0.1 seems to work better
        zero_environment_reward=True,
        baseline=LinearFeatureBaseline(env_spec=env.spec),
        eval_reward=True,
        fig_dir = dirname
    )

    # with rllab_logdir(algo=algo, dirname='data/pendulum_gcl{}'.format(n_experts)):
    with rllab_logdir(algo=algo, dirname=dirname):
        with tf.Session():
            algo.fig_dirname = dirname
            algo.train()

if __name__ == "__main__":
    main(eval_reward = True)
