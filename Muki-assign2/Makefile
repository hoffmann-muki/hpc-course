CC = mpicc
CXX = mpicxx
CFLAGS = -O2
CXXFLAGS = -O2

all: jacobi2d

jacobi2d: jacobi2d.c
	$(CC) $(CFLAGS) -o $@ $< -lm

debug: CFLAGS = -g -Wall
debug: jacobi2d

clean:
	rm -f jacobi2d
