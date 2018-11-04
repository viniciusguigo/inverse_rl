import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
sns.set(style='whitegrid', font_scale=1.25)

def main(n_idx):
    # define data location
    data_addr = './data/pendulum_gcl{}'.format(n_idx)

    # load data from progress log
    data = np.genfromtxt('{}/progress.csv'.format(data_addr), delimiter=',', skip_header=True)
    labels = np.genfromtxt('{}/progress.csv'.format(data_addr), dtype=str, delimiter=',', max_rows=1)

    # find returns on progress.csv file
    avg_return_idx = np.where([label == 'AverageReturn' for label in labels])[0][0]

    # plot them
    plt.ylabel(labels[avg_return_idx])
    plt.xlabel('Iteration')
    plt.plot(data[:,avg_return_idx],label=n_idx)
    plt.xlim([0,data.shape[0]])
    plt.tight_layout()
    plt.legend()

if __name__ == '__main__':
    ns = ['5','10','15','20']
    plt.figure()
    for n in ns:
        main(n_idx=n)
    plt.show()