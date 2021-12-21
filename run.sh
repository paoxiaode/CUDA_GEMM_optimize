#!/bin/bash
flie_name="data/mulMat.txt"
echo this is result of GEMM > $flie_name
tmp=(384 512 640 768 896 1024 1152 1408 1664 1920 2176 2432 2816 3200 3584 3968 4096)
for i in ${tmp[@]}
do
echo result $i
echo result $i >> $flie_name
mulMat_all_v2 $i>> $flie_name

done