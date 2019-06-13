from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.algos.ddpg import DDPG
from rllab.envs.normalized_env import normalize

from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction

from inverse_rl.utils.log_utils import rllab_logdir

def main():
    env = TfEnv(GymEnv('LunarLanderContinuous-v2', record_video=False, record_log=False))
    # env = normalize(GymEnv('LunarLanderContinuous-v2', record_video=False, record_log=False))
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(20, 20)) # (500, 500)
    # policy = DeterministicMLPPolicy(
    #     env_spec=env.spec,
    #     # The neural network policy should have two hidden layers
    #     hidden_sizes=(100, 100, 100, 100)
    # )

    # es = OUStrategy(env_spec=env.spec)

    # qf = ContinuousMLPQFunction(env_spec=env.spec,hidden_sizes=(100, 100, 100, 100))

    # algo = DDPG(
    #     env=env,
    #     policy=policy,
    #     es=es,
    #     qf=qf,
    #     batch_size=500,
    #     max_path_length=100,
    #     epoch_length=1000,
    #     min_pool_size=10000,
    #     n_epochs=1000,
    #     discount=0.99,
    #     scale_reward=0.1,
    #     qf_learning_rate=1e-3,
    #     policy_learning_rate=1e-4,
    #     # Uncomment both lines (this and the plot parameter below) to enable plotting
    #     # plot=True,
    # )

    algo = TRPO(
        env=env,
        policy=policy,
        n_itr=1000,
        batch_size=7000, # 3000 #2000
        max_path_length=200, # 200
        discount=0.99,
        store_paths=True,
        step_size=0.01,
        # baseline=GaussianMLPBaseline(env_spec=env.spec, regressor_args=dict(batchsize=5000,hidden_sizes=(100, 100, 100, 100)))
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    with rllab_logdir(algo=algo, dirname='data/lunarlander_collect_2'):
        algo.train()

if __name__ == "__main__":
    main()
