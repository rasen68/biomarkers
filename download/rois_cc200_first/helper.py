import numpy as np
from numpy.typing import NDArray
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

def csv_to_np(infile: str) -> NDArray: 
    heatmap = np.genfromtxt(infile, delimiter=',', skip_header=1)

    # set diagonal to 0
    for i in range(len(heatmap)):
        heatmap[i][i] = 0

    return heatmap

def sc_to_heatmap(heatmap: NDArray, outfile: str):
    fig, ax = plt.subplots()
    im = ax.imshow(np.sqrt(heatmap))
    plt.colorbar(mappable=im, ax=ax)
    fig.tight_layout()
    plt.savefig(outfile+".png", bbox_inches='tight')
    plt.close("all")

def fc_to_heatmap(heatmap: NDArray, outfile: str):
    # only positive / negative weight edges
    heatmap_pos = np.maximum(heatmap, 0)
    heatmap_neg = np.minimum(heatmap, 0)

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)
    plt.colorbar(mappable=im, ax=ax)
    fig.tight_layout()
    plt.savefig(outfile+"_full.png", bbox_inches='tight')
    plt.close("all")

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap_pos)
    plt.colorbar(mappable=im, ax=ax)
    fig.tight_layout()
    plt.savefig(outfile+"_pos.png", bbox_inches='tight')
    plt.close("all")

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap_neg)
    plt.colorbar(mappable=im, ax=ax)
    fig.tight_layout()
    plt.savefig(outfile+"_neg.png", bbox_inches='tight')
    plt.close("all")

def sc_to_histogram(heatmap: NDArray, outfile: str): 
    vector = np.triu(heatmap).flat
    vector = list(filter(lambda x: x != 0, vector))
    
    plt.hist(vector, bins=50)
    plt.savefig(outfile+".png")
    plt.close("all")

def fc_to_histogram(heatmap: NDArray, outfile: str): 
    heatmap_pos = np.maximum(heatmap, 0)
    heatmap_neg = np.minimum(heatmap, 0)

    vector_full = heatmap[np.triu_indices(heatmap.shape[0])]
    vector_pos = heatmap_pos[np.triu_indices(heatmap_pos.shape[0])]
    vector_neg = heatmap_neg[np.triu_indices(heatmap_neg.shape[0])]
    
    plt.hist(vector_full, bins=50)
    plt.savefig(outfile+"_full.png")
    plt.close("all")

    plt.hist(vector_pos, bins=50)
    plt.savefig(outfile+"_pos.png")
    plt.close("all")

    plt.hist(vector_neg, bins=50)
    plt.savefig(outfile+"_neg.png")
    plt.close("all")

def csv_to_histogram(infile: str, column: str, outfile: str):
    df = pd.read_csv(infile)
    vector = df[column][pd.to_numeric(df[column]).notnull()].to_numpy()

    plt.hist(vector)
    plt.savefig(outfile+".png")
    plt.close("all")

def connectome_to_scatter_plot(sc_heatmap: NDArray, fc_heatmap: NDArray, outfile: str):
    sc_vector = sc_heatmap[np.triu_indices(sc_heatmap.shape[0])]
    fc_vector = fc_heatmap[np.triu_indices(fc_heatmap.shape[0])]

    plt.scatter(sc_vector, fc_vector)
    plt.savefig(outfile+".png")
    plt.close("all")

def csv_to_scatter_plot(infile: str, xcol: str, ycol: str, outfile: str):
    df = pd.read_csv(infile)
    try:
        df_m = df[df['sex'] == 'Male']
        df_f = df[df['sex'] == 'Female']
    except:
        df_m = df[df['Sex'] == 'Male']
        df_f = df[df['Sex'] == 'Female']
    xvec_m, yvec_m = df_m[[xcol]], df_m[[ycol]]
    xvec_f, yvec_f = df_f[[xcol]], df_f[[ycol]]

    plt.scatter(xvec_m, yvec_m)
    plt.scatter(xvec_f, yvec_f)
    plt.savefig(outfile+".png")
    plt.close("all")

def sc_to_box_plot(heatmaps: List[NDArray], outfile: str):
    vectors = []
    for heatmap in heatmaps:
        vectors.append(np.sum(heatmap, axis=0))

    plt.boxplot(vectors)
    plt.savefig(outfile+".png")
    plt.close("all")
    
def csv_to_box_plot(infile: str, columns: List[str], outfile: str):
    df = pd.read_csv(infile)
    try:
        df_m = df[df['sex'] == 'Male']
        df_f = df[df['sex'] == 'Female']
    except:
        df_m = df[df['Sex'] == 'Male']
        df_f = df[df['Sex'] == 'Female']

    m_func = lambda x: df_m[x]
    f_func = lambda x: df_f[x]
    vectors = [f(col) for col in columns for f in (m_func, f_func)]

    plt.boxplot(vectors)
    plt.savefig(outfile+".png")
    plt.close("all")

def sc_to_violin_plot(heatmaps: List[NDArray], outfile: str):
    vectors = []
    for heatmap in heatmaps:
        vectors.append(np.sum(heatmap, axis=0))

    plt.violinplot(vectors)
    plt.savefig(outfile+".png")
    plt.close("all")
    
def csv_to_violin_plot(infile: str, columns: List[str], outfile: str):
    df = pd.read_csv(infile)
    try:
        df_m = df[df['sex'] == 'Male']
        df_f = df[df['sex'] == 'Female']
    except:
        df_m = df[df['Sex'] == 'Male']
        df_f = df[df['Sex'] == 'Female']

    m_func = lambda x: df_m[x]
    f_func = lambda x: df_f[x]
    vectors = [f(col) for col in columns for f in (m_func, f_func)]
    print(columns)

    for i in vectors:
        mean = np.mean(i)
        stddev = np.std(i, ddof=1)
        print(stddev)
        print(mean-1.645*stddev, ",", mean+1.645*stddev)

    plt.violinplot(vectors)
    plt.savefig(outfile+".png")
    plt.close("all")
