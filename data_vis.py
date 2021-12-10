import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = np.loadtxt(open('ResponseMatrix.csv'), delimiter=",")
DnA = []
for statement in range(16):
    d = sum(data[:,statement]==0)
    n = sum(data[:,statement]==0.5)
    a = sum(data[:,statement]==1)
    DnA.append([x/len(data)*100 for x in [d,n,a]])

# sns.color_palette('pastel')

def plot_DNA(dna_i):
    df = pd.DataFrame(DnA[dna_i]).T
    ax = df.plot.barh(stacked=True)
    for rect, value in zip(ax.patches, DnA[dna_i]):
        if value != 0:
            h = rect.get_height()/2.
            w = rect.get_width()/2.
            x, y = rect.get_xy()
            ax.text(x+w, y+h,int(round(value)),horizontalalignment='center',verticalalignment='center')
    plt.xlim((0,100))
    plt.title('x')
    plt.legend(['Disagree','N/A','Agree'])
    plt.axis('off')
    plt.show()

plot_DNA(5)
