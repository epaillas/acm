# compiler choice
CC    = gcc

all: c

c:
	make -C acm/estimators/galaxy_clustering/src all

clean:
	rm -rf build/
	rm -f acm/estimators/galaxy_clustering/src/*.*o
	rm -f acm/estimators/galaxy_clustering/src/fastmodules.c
	rm -f acm/estimators/galaxy_clustering/src/fastmodules*.so
	rm -f acm/estimators/galaxy_clustering/src/minkowski.c
	rm -f acm/estimators/galaxy_clustering/src/minkowski*.so
	rm -f acm/estimators/galaxy_clustering/src/*.pyc
	rm -f acm/estimators/galaxy_clustering/src/*.exe
	rm -f acm/estimators/galaxy_clustering/src/c/*.o
	rm -f acm/estimators/galaxy_clustering/src/c/*.exe
	rm -f acm/estimators/galaxy_clustering/src/pydive*.so
	rm -f acm/estimators/galaxy_clustering/src/pydive.cpp