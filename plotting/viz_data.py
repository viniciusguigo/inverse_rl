import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', font_scale=1.25)

# define data location
data_addr = './plotting/pendulum_final'

# load data from progress log
data = np.genfromtxt('{}/progress.csv'.format(data_addr), delimiter=',', skip_header=True)
label = np.genfromtxt('{}/progress.csv'.format(data_addr), dtype=str, delimiter=',', max_rows=1)
print(data.shape)
print(label)

plt.figure()
plt.ylabel('Average Disc. Return')
plt.xlabel('Iteration')
plt.plot(data[:,-5])
plt.tight_layout()
plt.show()

