# import
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.25)


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

def plot_history(fig, ax, avg_return, n_samples, run_id_label, avg_factor):
    """Plot history results (iterations, mean reward) using Matplotlib.
    """
    # average data
    avg_return = avg_return[0]
    samples = avg_return.shape[0]
    avg_samples = int(np.floor(samples/avg_factor))    
    avg = np.zeros((avg_samples,2))

    for i in range(avg_samples):
        initial_idx = i*avg_factor
        avg[i,0] = initial_idx + avg_factor

        # remove nan values and compute average
        # x = x[~numpy.isnan(x)]
        x = avg_return[initial_idx:initial_idx + avg_factor]
        x = x[~np.isnan(x)]
        avg[i,1] = np.mean(x)

    # plot avg
    ax.plot(avg[:,0], avg[:,1], label=run_id_label)


def process_avg(fig, ax, run_id, run_id_label, avg_factor, n_seeds=1):
    """Load files, calculate average, std dev, plot, and save figure.
    """
    # load first file (seed == 1)
    data = np.genfromtxt('{}/progress.csv'.format(run_id), delimiter=',', skip_header=True)
    n_samples = data.shape[0]
    labels = np.genfromtxt('{}/progress.csv'.format(run_id), dtype=str, delimiter=',', max_rows=1)

    # find returns on progress.csv file
    avg_return_idx = np.where([label == 'OriginalTaskAverageReturn' for label in labels])[0][0]
    avg_return = data[:,avg_return_idx].reshape(1,n_samples)

    # plot
    plot_history(fig, ax, avg_return, n_samples, run_id_label, avg_factor)

    return n_samples

if __name__ == '__main__':
    # setup figure
    avg_factor = 3
    save_pic = True
    name_pic = './plotting/airsim_gail.png'

    # plot rewards
    fig, ax = plt.subplots(1)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Original Task Average Return")
    #ax.set_xlim([0,500])

    # plot data
    ## GAIL (v2)
    #run_id = './data/airsim_gail_v2'
    #run_id_label = 'v1'
    #n_samples = process_avg(fig, ax, run_id, run_id_label, avg_factor)

    ## GAIL (v3)
    #run_id = './data/airsim_gail_v3'
    #run_id_label = 'v2'
    #n_samples_v3 = process_avg(fig, ax, run_id, run_id_label, avg_factor)

    ## GAIL (v4)
    run_id = './data/airsim_gail'
    run_id_label = 'GAIL'
    n_samples = process_avg(fig, ax, run_id, run_id_label, avg_factor=1)
    ax.set_xlim([0,n_samples])

    ## HUMAN
    log_data = np.genfromtxt('data/airsim_human_data/log.csv', delimiter=',', skip_header=True)
    obs_log_data = np.genfromtxt('data/airsim_human_data/low_obs_log.csv', delimiter=',', skip_header=True)
    n_epis = int(np.max(log_data[:,1]))
    human_data = np.zeros(n_epis)

    for i in range(n_epis):
        _, _, returns = parse_episode(log_data, obs_log_data, i)
        human_data[i] = np.sum(returns)
    
    # print human mean reward and stddev bounds
    human_mean = np.mean(human_data)
    human_stddev = np.std(human_data)
    ax.hlines(human_mean, 0, n_samples, linestyle='--', color='black', alpha=0.75, label='Human')
    #ax.hlines(human_mean+human_stddev, 0, n_samples_v3, linestyle='dotted', label='Human Stddev')
    #ax.hlines(human_mean-human_stddev, 0, n_samples_v3, linestyle='dotted')
    ax.fill_between(np.arange(n_samples), human_mean, human_mean+human_stddev, facecolor='black', alpha=0.25)
    ax.fill_between(np.arange(n_samples), human_mean, human_mean-human_stddev, facecolor='black', alpha=0.25)
    
    # save/show pic
    plt.legend()
    plt.tight_layout()
    if save_pic:
        plt.savefig(name_pic, dpi=300)    
    plt.show()
        
