#!/bin/bash

echo this is result of GEMM > mulMat.txt
tmp=(384 512 640 768 896 1024 1152 1408 1664 1920 2176 2432 2816 3200 3584 3968 4096)
for i in ${tmp[@]}
do
echo result $i
echo result $i >> mulMat.txt
# nsight_cmp --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second mulMat_naive $i>> mulMat.txt
# nsight_cmp --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second mulMat_1x4 $i>> mulMat.txt
mulMat_naive $i>> mulMat.txt
mulMat_1x4 $i>> mulMat.txt
mulMat_4x4 $i>> mulMat.txt
mulMat_Tiling $i>> mulMat.txt
mulMat_Tiling_Coalesing $i>> mulMat.txt
mulMat_Tiling_noBankflict $i>> mulMat.txt
done