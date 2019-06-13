import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
sns.set(style='whitegrid', font_scale=1.25)

def main(n_idx):
    # define data location
    if n_idx == 'collect':
        data_addr = './data/lunarlander_collect_2'
        t_idx = 'expert'
    else:
        data_addr = './data/lunarlander_irl_{}'.format(n_idx)

    if n_idx == '1':
        t_idx = '25'
    if n_idx == '5':
        t_idx = '125'
    if n_idx == '50':
        t_idx = '1250'    
    if n_idx == '100':
        t_idx = '2505'
    if n_idx == '400':
        t_idx = '10035' 
    # load data from progress log
    data = np.genfromtxt('{}/progress.csv'.format(data_addr), delimiter=',', skip_header=True)
    labels = np.genfromtxt('{}/progress.csv'.format(data_addr), dtype=str, delimiter=',', max_rows=1)

    # find returns on progress.csv file
    if n_idx == 'collect':
        avg_return_idx = np.where([label == 'AverageReturn' for label in labels])[0][0]
    else:
        avg_return_idx = np.where([label == 'OriginalTaskAverageReturn' for label in labels])[0][0]

    # plot them
    plt.ylabel('Average Reward')
    plt.xlabel('Iteration')
    # plt.plot(data[25:500,avg_return_idx],label=t_idx)
    # plt.xlim([25,500])
    plt.plot(data[:,avg_return_idx],label=t_idx)
    plt.xlim([0,data.shape[0]])
    plt.tight_layout()
    plt.legend()

if __name__ == '__main__':
    # ns = ['5','30','40','50','60','70','100','200', '300','400','600','700'] # Trajectories 51, 304, 405, -, 606, 812, - , 4078, 6593, 15095, 17653
    ns = ['1','5','50','100','400']#,'700'] # Trajectories 125, 1250, 2505, 10035
    ns = ['1']
    plt.figure()
    # main(n_idx='collect')
    for n in ns:
        print('Processing {} expert trajectories'.format(n))
        main(n_idx=n)
    plt.show()