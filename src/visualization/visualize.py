import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def rca_density(cdata:pd.DataFrame):
    f = plt.figure(figsize=(10, 8))
    plt.fill([1, 1, 0.1165, 0.1165], [650, 0.1, 0.1, 650], 'g', alpha=0.5)
    plt.fill([0, 0, 0.1165, 0.1165], [650, 0 - 50, 0 - 50, 650], 'r', alpha=0.5)
    plt.fill([1, 1, 0.1165, 0.1165], [0.1, -50, -50, 0.1], 'y', alpha=0.5)
    sns.scatterplot(cdata.density, cdata.rca)
    plt.title('RCA vs Density')
    plt.hlines(1, 0, 1)
    plt.vlines(0.1165, -.1, 650)
    #plt.savefig('rca_density.png')
    return f

def cog_pci(cdata):
    f = plt.figure(figsize=(10,8))
    sns.scatterplot(cdata.pci, cdata.cog)
    plt.title('COG vs PCI')
    plt.vlines(1, cdata.cog.min(), cdata.cog.max())
    plt.hlines(1, cdata.pci.min(), cdata.pci.max())
    return f