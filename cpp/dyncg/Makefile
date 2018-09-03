DYNET_PATH ?= ${HOME}/code/dynet
EIGEN_PATH ?= ${HOME}/code/eigen
AD3_PATH ?= ${HOME}/code/ad3

OBJS = $(AD3_PATH)/examples/cpp/parsing/FactorTree.o
CC = g++
DEBUG = 
INCLUDES = -I$(DYNET_PATH) -I$(EIGEN_PATH) -I$(AD3_PATH) 
LIBS = -L$(AD3_PATH)/ad3 
CFLAGS = -O3 -Wall -Wno-sign-compare -Wno-int-in-bool-context -c -fmessage-length=0 $(INCLUDES) -DEIGEN_FAST_MATH -fPIC -fno-finite-math-only -Wno-missing-braces -std=c++11 -funroll-loops $(DEBUG) 
LFLAGS = -ldynet -lad3

ifdef MKL_PATH
INCLUDES += -I$(MKL_PATH)/include
LIBS += -L$(MKL_PATH)/lib/intel64
LFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
CFLAGS += -DEIGEN_USE_MKL_ALL
endif

LIBS_GPU = $(LIBS) -L$(DYNET_PATH)/build-cuda/dynet/ 
LIBS_CPU = $(LIBS) -L$(DYNET_PATH)/build/dynet/ 
LFLAGS_GPU = $(LIBS_GPU) $(LFLAGS)
LFLAGS_CPU = $(LIBS_CPU) $(LFLAGS) 

HEADERS = ../common/utils.h ../common/args.h ../common/treernn.h tree_encoders.h

sparsemst-gpu.o : sparsemst.cc sparsemst.h ../../src/FactorTreeFast.h
	$(CC) $(CFLAGS) -DUSING_CUDA sparsemst.cc -o sparsemst-gpu.o

sparsemst-cpu.o : sparsemst.cc sparsemst.h ../../src/FactorTreeFast.h
	$(CC) $(CFLAGS) sparsemst.cc -o sparsemst-cpu.o

test_sparsemst: sparsemst-cpu.o test_sparsemst.o
	$(CC) $(OBJS) sparsemst-cpu.o test_sparsemst.o $(LFLAGS_CPU) -o test_sparsemst

test_sparsemst.o : test_sparsemst.cc sparsemst.h
	$(CC) $(CFLAGS) test_sparsemst.cc

revdict-cpu: $(OBJS) revdict-cpu.o sparsemst-cpu.o
	$(CC) $(OBJS) sparsemst-cpu.o revdict-cpu.o $(LFLAGS_CPU) -o revdict-cpu

revdict-gpu: $(OBJS) revdict-gpu.o sparsemst-gpu.o
	$(CC) $(OBJS) sparsemst-gpu.o revdict-gpu.o $(LFLAGS_GPU) -o revdict-gpu

revdict-gpu.o : revdict.cc revdict.h $(HEADERS)
	$(CC) $(CFLAGS) -DUSING_CUDA revdict.cc -o revdict-gpu.o

revdict-cpu.o : revdict.cc revdict.h $(HEADERS)
	$(CC) $(CFLAGS) revdict.cc -o revdict-cpu.o

sentclf-cpu: $(OBJS) sentclf-cpu.o sparsemst-cpu.o
	$(CC) $(OBJS) sparsemst-cpu.o sentclf-cpu.o $(LFLAGS_CPU) -o sentclf-cpu

sentclf-gpu: $(OBJS) sentclf-gpu.o sparsemst-gpu.o
	$(CC) $(OBJS) sparsemst-gpu.o sentclf-gpu.o $(LFLAGS_GPU) -o sentclf-gpu

sentclf-cpu.o : sentclf.cc sentclf.h $(HEADERS)
	$(CC) $(CFLAGS) sentclf.cc -o sentclf-cpu.o

sentclf-gpu.o : sentclf.cc sentclf.h $(HEADERS)
	$(CC) $(CFLAGS) -DUSING_CUDA sentclf.cc -o sentclf-gpu.o

nliclf-cpu: $(OBJS) nliclf-cpu.o sparsemst-cpu.o
	$(CC) $(OBJS) sparsemst-cpu.o nliclf-cpu.o $(LFLAGS_CPU) -o nliclf-cpu

nliclf-gpu: $(OBJS) nliclf-gpu.o sparsemst-gpu.o
	$(CC) $(OBJS) sparsemst-gpu.o nliclf-gpu.o $(LFLAGS_GPU) -o nliclf-gpu

nliclf-cpu.o : nliclf.cc nliclf.h ../common/basenli.h $(HEADERS)
	$(CC) $(CFLAGS) nliclf.cc -o nliclf-cpu.o

nliclf-gpu.o : nliclf.cc nliclf.h ../common/basenli.h $(HEADERS)
	$(CC) $(CFLAGS) -DUSING_CUDA nliclf.cc -o nliclf-gpu.o

clean:
	rm -f *.o *~ revdict-gpu revdict-gpu sentclf-cpu sentclf-gpu nliclf-cpu nliclf-gpu test_sparsemst
