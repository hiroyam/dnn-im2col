#
# To generate linker option, use Intel Math Kernel Library Link Line Advisor
# - https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor/
#
# To use intel mkl, set environment variables before using make
# source /opt/intel/mkl/bin/mklvars.sh intel64
#


CXX     = g++
TARGET  = target/a.out

all: main

run:
	$(TARGET)

main:
	$(CXX) src/main.cpp \
		-o $(TARGET) \
		-std=c++11 -Wall -O3
