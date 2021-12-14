import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from numpy.lib.function_base import append

def gflops(m,t):
    """
    Calculate the GFlops for a matrix multiplication kernel.
    """
    operations = (m**3)*2
    return operations*1e-9/t

def main():
    # algo = sys.argv[1]
    file_name = "mulMat.txt"
    mat_size = []
    res = []
    t = []
    lengend = ["mulMat_naive", "mulMat_1x4", "mulMat_4x4", "mulMat_Tiling", "mulMat_Tiling_Coalesing", "mulMat_Tiling_noBankflict"]
    with open(file_name,'r') as f:
        for line in f.readlines():
            if line[0] == "r":
                if len(t)!=0: res.append(t)
                t = []
                mat_size.append(int(line.strip().split(" ")[1]))
            elif line[0] == "m":
                t.append(float(line.strip().split("\t")[1]))
    res.append(t)
    fig = plt.figure(dpi=200,figsize=[8,6])
    for j in range(len(res[0])):
        plt.plot(mat_size,[gflops(mat_size[i],res[i][j]) for i in range(len(res))],linewidth=2,marker='o',markersize=6)
    plt.legend(lengend, fontsize = 6)
    plt.xticks(mat_size,rotation = 30, fontsize = 6)
    plt.xlabel("Matrix size")
    plt.ylabel("GFLOPS")
    plt.savefig("result.png")


if __name__ == '__main__':
    main()