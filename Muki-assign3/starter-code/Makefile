CXX = g++
CXXFLAGS = -std=c++11 -O2

# Use environment-detected paths or fall back to defaults
# For conda/pip opencv: python3 -c "import cv2; import os; print(os.path.dirname(cv2.__file__))"
CONDA_PREFIX ?= $(shell python3 -c "import sys; print(sys.prefix)" 2>/dev/null)
OPENCV_ROOT ?= $(CONDA_PREFIX)
CUDA_ROOT ?= $(CUDA_HOME)

# Fallback hardcoded paths for zaratan/module builds
ifeq ($(OPENCV_ROOT),)
  OPENCV_ROOT = /cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen2/gcc-9.4.0/opencv-4.5.2-xxyodykxk3vuw64tlvm6sujgaxnctgep
endif
ifeq ($(CUDA_ROOT),)
  CUDA_ROOT = /cvmfs/hpcsw.umd.edu/spack-software/2022.06.15/linux-rhel8-zen2/gcc-9.4.0/cuda-11.6.2-eonihhhvlh4s2d6riyb7al2qivzn477u
endif

CXX_INCLUDES = -I$(OPENCV_ROOT)/include/opencv4 -I$(CUDA_ROOT)/include
CXX_LIBS = -L$(OPENCV_ROOT)/lib -L$(OPENCV_ROOT)/lib64 -L$(CUDA_ROOT)/lib64 -lopencv_core -lopencv_videoio -lopencv_imgproc -lcudart

CUDA = nvcc
CUDAFLAGS = -std=c++11 -O2 -Xcompiler "$(CXXFLAGS)"
GENCODE = --generate-code arch=compute_80,code=sm_80

TARGET = video-effect

all: $(TARGET)

video-effect.o: video-effect.cu
	$(CUDA) $(CUDAFLAGS) $(GENCODE) -o $@ -c $<

video-effect: driver.cpp video-effect.o
	$(CXX) $(CXXFLAGS) $(CXX_INCLUDES) -o $@ $^ $(CXX_LIBS)

clean:
	rm -f $(TARGET) *.o
