import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from numpy.lib.function_base import append

def main():
    # algo = sys.argv[1]
    file_name = "mulMat.txt"
    mat_size = []
    res = []
    t = []
    lengend = ["mulMat_cublas", "mulMat_naive", "mulMat_1x4", "mulMat_4x4", "mulMat_Tiling", "mulMat_Tiling_Coalesing", "mulMat_Tiling_noBankflict", "mulMat_outProd"]
    with open(file_name,'r') as f:
        for line in f.readlines():
            if line[0] == "r":
                if len(t)!=0: res.append(t)
                t = []
                mat_size.append(int(line.strip().split(" ")[1]))
            elif line[0] == "A":
                t.append(float(line.strip().split(" ")[6]))
    res.append(t)
    fig = plt.figure(dpi=200,figsize=[8,6])
    for j in range(len(res[0])):
        plt.plot(mat_size,[res[i][j] for i in range(len(res))],marker='o')
    plt.legend(lengend, fontsize = 6)
    plt.xticks(mat_size,rotation = 30, fontsize = 6)
    plt.xlabel("Matrix size")
    plt.ylabel("GFLOPS")
    plt.savefig("result.png")


if __name__ == '__main__':
    main()