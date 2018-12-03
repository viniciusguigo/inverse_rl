import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
sns.set(style='whitegrid', font_scale=1.25)

def main(data_addr, data_label, stat_label):
    # load data from progress log
    data = np.genfromtxt('{}/progress.csv'.format(data_addr), delimiter=',', skip_header=True)
    labels = np.genfromtxt('{}/progress.csv'.format(data_addr), dtype=str, delimiter=',', max_rows=1)

    # find returns on progress.csv file
    avg_return_idx = np.where([label == stat_label for label in labels])[0][0]

    # plot them
    plt.ylabel(labels[avg_return_idx])
    plt.xlabel('Iteration')
    plt.plot(data[:,avg_return_idx], label=data_label)
    plt.xlim([0,data.shape[0]])
    plt.tight_layout()
    plt.legend()

if __name__ == '__main__':
    # define data location
    data_addrs = ['./data/airsim_gail_v2', './data/airsim_gail_v3']
    data_label = ['v2', 'v3']

    # define statistics to plot
    label_count = 0
    plt.figure()
    for data_addr in data_addrs:
        main(data_addr, data_label[label_count], 'Entropy')
        label_count += 1

    # display at the end
    plt.show()