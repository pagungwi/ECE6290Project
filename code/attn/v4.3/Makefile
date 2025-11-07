##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))


#---------------------------------------------------------------------
# VITIS related variables

# VITIS related variables
AIETOOLS_DIR ?= $(shell realpath $(dir $(shell which xchesscc))/../)
AIE_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/versal_prod/lib
AIE2_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/aie_ml/lib

AIEOPT_DIR ?= $(shell realpath $(dir $(shell which aie-opt))/..)

WARNING_FLAGS = -Wno-parentheses -Wno-attributes -Wno-macro-redefined -Wno-empty-body

CHESSCC1_FLAGS = -f -p me -P ${AIE_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESSCC2_FLAGS = -f -p me -P ${AIE2_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
CHESS_FLAGS = -P ${AIE_INCLUDE_DIR}

CHESSCCWRAP1_FLAGS = aie -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2_FLAGS = aie2 -I ${AIETOOLS_DIR}/include
CHESSCCWRAP2P_FLAGS = aie2p -I ${AIETOOLS_DIR}/include
PEANOWRAP2_FLAGS = -O2 -std=c++20 --target=aie2-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include
PEANOWRAP2P_FLAGS = -O2 -std=c++20 --target=aie2p-none-unknown-elf ${WARNING_FLAGS} -DNDEBUG -I ${AIEOPT_DIR}/include

TEST_POWERSHELL := $(shell command -v powershell.exe >/dev/null 2>&1 && echo yes || echo no)
ifeq ($(TEST_POWERSHELL),yes)
	powershell = powershell.exe
	getwslpath = wslpath -w
else
	powershell =
	getwslpath = echo
endif

#---------------------------------------------------------------------

# MMUL rst=888, need to have this in xchess compiler

CHESSCCWRAP1_FLAGS += -D AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
CHESSCCWRAP2_FLAGS += -D AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
CHESSCCWRAP2P_FLAGS += -D AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
#---------------------------------------------------------------------





MLIR_HOME_TOP=/home/alan/workspace/amd_npu/mlir-aie








targetname = flashattn

# 1MB
trace_size = 1048576

# 4 MB
trace_size = 4194304


# 512KB
#trace_size = 524288


devicename = npu2



verbosity = 1

iters = 1
warmup = 1

#----------------------------------


lut_exp_dir = ${srcdir}/lut_exp



#$(info "-----------------------")
#$(info ${AIEOPT_DIR})
# /home/alan/workspace/amd_npu/mlir-aie/ironenv/lib/python3.12/site-packages/mlir_aie
#$(info "-----------------------")


aie2p_runtime_dir = ${MLIR_HOME_TOP}/aie_runtime_lib/AIE2P


aie_py_src=${targetname}.py


# NPU2: AIE2P
ifeq ($(devicename),npu2)
	VPATH :=${MLIR_HOME_TOP}/aie_kernels/aie2p
else
	VPATH :=${MLIR_HOME_TOP}/aie_kernels/aie2
endif



ifeq ($(devicename),npu2)
	colshift = 0
else
	colshift = 1
endif







all: build/final.xclbin build/insts.bin




#------------------------------------------------------------

CHESS=true


ifeq ($(devicename),npu)
build/lut_based_ops.o: ${lut_exp_dir}/lut_based_ops.cpp
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS} -I. -c $< -o ${@F}

else ifeq ($(devicename),npu2)
build/lut_based_ops.o: ${lut_exp_dir}/lut_based_ops.cpp
	mkdir -p ${@D}
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -I. -c $< -o ${@F}

endif



build/flash.cc.o: ${srcdir}/flash.cc
	mkdir -p ${@D}

ifeq ($(devicename),npu)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2_FLAGS}  -I. -I${lut_exp_dir} -c $< -o ${@F}

else ifeq ($(devicename),npu2)
	cd ${@D} && xchesscc_wrapper ${CHESSCCWRAP2P_FLAGS} -I. -I${lut_exp_dir} -c $< -o ${@F}
else
	echo "Device type not supported"
endif


build/kernels.a: build/flash.cc.o  #build/lut_based_ops.o
	ar rvs $@ $+

#------------------------------------------------------------




#cp ${srcdir}/aie2.mlir ${srcdir}/build/

build/aie.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} > $@

build/aie_trace.mlir: ${srcdir}/${aie_py_src}
	mkdir -p ${@D}
	python3 $< ${devicename} ${trace_size} > $@




#----------------------------------------

# Chess

build/final.xclbin: build/aie.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
			     --aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)


build/final_trace.xclbin: build/aie_trace.mlir build/kernels.a
	mkdir -p ${@D}
	cd ${@D} && aiecc.py --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
			     --aie-generate-npu-insts --npu-insts-name=insts.bin $(<:%=../%)

#----------------------------------------




${targetname}.exe: ${srcdir}/test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && ${powershell} cmake ${srcdir} -DTARGET_NAME=${targetname}
	cd _build && ${powershell} cmake --build . --config Release
ifeq "${powershell}" "powershell.exe"
	cp _build/${targetname}.exe $@
else
	cp _build/${targetname} $@
endif

run: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -v ${verbosity} --iters ${iters} --warmup ${warmup}

profile: ${targetname}.exe build/final.xclbin
	${powershell} ./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -p results.csv -v ${verbosity} --iters ${iters} --warmup ${warmup}

trace: ${targetname}.exe build/final_trace.xclbin
	${powershell} ./$< -x build/final_trace.xclbin -i build/insts.bin -k MLIR_AIE -t ${trace_size} -v ${verbosity} --iters ${iters} --warmup ${warmup}

	${MLIR_HOME_TOP}/programming_examples/utils/parse_trace.py --input trace.txt --mlir build/aie_trace.mlir --colshift ${colshift} --output trace.json

	#${MLIR_HOME_TOP}/programming_examples/utils/get_trace_summary.py --input trace_softmax.json

clean:
	rm -rf build _build ${targetname}.exe trace.txt trace.json
