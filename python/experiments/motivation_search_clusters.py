import torch
import random
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import csv
import math

root = os.getenv("AUTOSPARSE_HOME")

def get_data(filepath: str):
    data = torch.load(filepath) # [['2 i 2 i0 16 i1 256 k 2 k0 4096 k1 1 1 A 4 k0 k1 i0 i1 A 4 i0 2 i1 1 k0 1 k1 2 1 j 2 j...1 2 6 j0 j1 k0 k1 i0 i1 j0 None j1 2 128 8', 30.153186]]
    # sorted_data = sorted(data, key=lambda x: x[1])
    # data = sorted_data[:int(len(sorted_data))]
    tile_i = []
    tile_k = []
    tile_j = []
    fmodes = []
    performence = []
    for item in data:
        schedule = item[0].split()
        assert schedule[5] == 'i1'
        tile_i.append(int(schedule[6]))
        assert schedule[11] == 'k1'
        tile_k.append(int(schedule[12]))
        assert schedule[35] == 'j1'
        tile_j.append(int(schedule[36]))
        # fmodes.append(int(item[22: 22 + 8: 2])
        performence.append(float(item[-1]))
    max_per = max(performence)
    min_per = min(performence)
    color = [item / max_per for item in performence]
    return tile_i, tile_k, tile_j, color

def draw_clusters(filepaths: str):
    tile_i, tile_k, tile_j, colors = [], [], [], []
    for fp in filepaths:
        i, k, j, c = get_data(fp)
        tile_i.extend(i)
        tile_k.extend(k)
        tile_j.extend(j)
        colors.extend(c)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_facecolor('white')

    # ax.grid(False) # close grid
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    
    ax.xaxis.pane.set_edgecolor('#808080')
    ax.yaxis.pane.set_edgecolor('#808080')
    ax.zaxis.pane.set_edgecolor('#808080')

    sc = ax.scatter(tile_i, tile_k, tile_j, c=colors, cmap='viridis', s=8, alpha=0.85)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.6])
    cbar = plt.colorbar(sc, cax=cbar_ax, pad=0.05)
    cbar.set_label(label='Excution Overhead', fontsize=10, labelpad=5)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlabel('Tile I', labelpad=-5, fontsize=10)
    ax.set_ylabel('Tile K', labelpad=-5, fontsize=10)
    ax.set_zlabel('Tile J', labelpad=-4, fontsize=10)

    ax.tick_params(axis='both', which='major', labelsize=8, pad = -3)
    ax.tick_params(axis='z', which='major', labelsize=8, pad = -1)

    plt.savefig(os.path.join(os.path.dirname(filepath[0]), "search_clusters.pdf"), format='pdf')
    plt.savefig(os.path.join(os.path.dirname(filepath[0]), "search_clusters.png"), format='png', transparent=True, dpi=600)



if __name__ == "__main__":
    mtx_names = [
        # "bcsstk38",
        # "mhd4800a",
        # "conf5_0-4x4-18",
        # "cca",
        # "Trefethen_20000",
        # "pf2177",
        # "msc10848",
        # "cfd1",
        # "net100",
        # "vanbody",
        # "net150",
        # "Chevron3_4x16_1",
        # "vibrobox_1x1_0",
        # "NACA0015_16x8_9",
        "nemspmm1_16x4_0",
        # "Trec6_16x16_9",
        # "crystk01_2x16_1",
        # "t2dal_a_8x4_3",
        # "EX1_8x8_4"
    ]
    for mtx_name in mtx_names:
        filepath = [
            os.path.join(
                root, 'python', 'experiments', 'epyc_evaluation_spmm', 
                mtx_name, 'random_searchingschedule_data.pth'
            ),
            os.path.join(
                root, 'python', 'experiments', 'epyc_evaluation_spmm', 
                mtx_name, 'p_searchingschedule_data.pth'
            ),
            # os.path.join(
            #     root, 'python', 'experiments', 'epyc_evaluation_spmm', 
            #     mtx_name, 'q_sa_searchingschedule_data.pth'
            # ),
        ]
        draw_clusters(filepath)
