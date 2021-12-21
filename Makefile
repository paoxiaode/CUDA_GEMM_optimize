ARCH ?= 80
# $(warning $(GENCODE_FLAG))

ifeq ($(GENCODE_FLAG),)
# Generate SASS code for each SM architecture listed in $(ARCH)
$(foreach sm,$(ARCH),$(eval GENCODE_FLAG += -gencode arch=compute_$(sm),code=sm_$(sm)))
# Generate PTX code for each SM architecture listed in $(ARCH)
$(foreach sm,$(ARCH),$(eval GENCODE_FLAG += -gencode arch=compute_$(sm),code=compute_$(sm)))
endif

NVCC=nvcc

NVCC_FLAGS=$(GENCODE_FLAG)
# CU_APPS= mulMat_naive mulMat_1x4 mulMat_4x4 mulMat_Tiling Deviceinfo mulMat_Tiling_Coalesing mulMat_Tiling_noBankflict mulMat_outProd mulMat_all
INCLUDE_DIR=/home/scratch.jiahuil_gpu/cuda_utest/GEMM_optimize/include
CU_APPS=mulMat_all_v2
INCLUDES += -I$(INCLUDE_DIR)
OBJS = $(INCLUDE_DIR)/GEMM.cu
OBJDIR = obj
all: ${CU_APPS}

# %: %.cu
# 	$(NVCC)  $(NVCC_FLAGS) include/GEMM.o  -o $@ $< 

# build:$(OBJS)
# 	$(NVCC) -c $(NVFLAGS) $(INCLUDES) $< -o ${OBJS}.o

%: %.cu
	$(NVCC)  $(NVCC_FLAGS)   -o $@ $< -lcublas
	
clean:
	rm -f ${CU_APPS}