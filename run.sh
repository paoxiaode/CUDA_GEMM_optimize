#!/bin/bash
flie_name="data/mulMat.txt"
echo this is result of GEMM > $flie_name
tmp=(256 1024  1408 1664 1920 2560 2816 3200 3584 4096 4864 5632) 
for ((i=512; i<=5632; i+=512))
do
echo result $i
echo result $i >> $flie_name
mulMat_all_v2 $i>> $flie_name

done