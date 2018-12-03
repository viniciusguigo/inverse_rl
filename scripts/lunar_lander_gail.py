import tensorflow as tf

from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
from sandbox.rocky.tf.policies.categorical_conv_policy import CategoricalConvPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
from rllab.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from rllab.envs.gym_env import GymEnv


from inverse_rl.algos.irl_trpo import IRLTRPO
from inverse_rl.models.imitation_learning import GAIL
from inverse_rl.utils.log_utils import rllab_logdir, load_latest_experts

def main():
    env = TfEnv(GymEnv('LunarLanderContinuous-v2', record_video=False, record_log=False))
    
    n_experts = 400
    experts = load_latest_experts('data/lunarlander_collect_1', n=n_experts, min_return=190, max_path_length=200)
    # print(len(experts))

    irl_model = GAIL(env_spec=env.spec, expert_trajs=experts)
    policy = GaussianMLPPolicy(name='policy', env_spec=env.spec, hidden_sizes=(300, 300, 100)) # (200, 200, 100)
    # policy = CategoricalConvPolicy(name='policy', env_spec=env.spec, hidden_sizes=(500, 500),
    #                                 conv_filters=[16, 16],
    #                                 conv_filter_sizes=[4, 4],
    #                                 conv_strides=[2, 2],
    #                                 conv_pads=[0] * len([16, 16]),)

    # network_kwargs = dict(
    #     num_encoding_levels=5,
    #     conv_filters=[16, 16],
    #     conv_filter_sizes=[4, 4],
    #     conv_strides=[2, 2],
    #     conv_pads=[0] * len([16, 16]),
    #     hidden_sizes=[32, 32],
    #     hidden_nonlinearity=LN.rectify,
    #     output_nonlinearity=None,
    #     name="mean_network"
    # )

    # conv_baseline_kwargs = dict(env_spec=env.spec,
    #                             regressor_args=dict(
    #                                 # mean_network=VggConvNetwork(
    #                                 #     input_shape=env.observation_space.shape,
    #                                 #     output_dim=1,
    #                                 #     **network_kwargs),
    #                                 use_trust_region=True,
    #                                 # step_size=args.step_size,
    #                                 normalize_inputs=True,
    #                                 normalize_outputs=True,
    #                                 hidden_sizes=[32, 32],
    #                                 conv_filters=[16, 16],
    #                                 conv_filter_sizes=[4, 4],
    #                                 conv_strides=[2, 2],
    #                                 conv_pads=[0] * len([16, 16]),
    #                                 # batchsize=200,
    #                                 # optimizer=PenaltyLbfgsOptimizer(n_slices=50),
    #                             ))
    # baseline = GaussianConvBaseline(**conv_baseline_kwargs)

    algo = IRLTRPO(
        env=env,
        policy=policy,
        irl_model=irl_model,
        n_itr=500,
        batch_size=7000,
        max_path_length=200,
        discount=0.99,
        store_paths=True,
        discrim_train_itrs=100,
        irl_model_wt=1.0,
        entropy_weight=0.0, # GAIL should not use entropy unless for exploration
        zero_environment_reward=True,
        # baseline=GaussianMLPBaseline(env_spec=env.spec, regressor_args=dict(batchsize=1000,hidden_sizes=(100, 100, 100, 100)))
        baseline=LinearFeatureBaseline(env_spec=env.spec)
    )

    with rllab_logdir(algo=algo, dirname='data/lunarlander_gail_'+str(n_experts)):
        with tf.Session():
            algo.train()

if __name__ == "__main__":
    main()
