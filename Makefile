# compiler choice
CC    = gcc

all: fastmodules c

.PHONY : fastmodules

c:
	make -C acm/estimators/galaxy_clustering/src all

fastmodules:
	python acm/estimators/galaxy_clustering/src/setup.py build_ext --inplace
	mv fastmodules*.so acm/estimators/galaxy_clustering/src/.
	mv minkowski*.so acm/estimators/galaxy_clustering/src/.
	mv pydive*.so acm/estimators/galaxy_clustering/src/.


clean:
	rm -f acm/estimators/galaxy_clustering/src/*.*o
	rm -f acm/estimators/galaxy_clustering/src/fastmodules.c
	rm -f acm/estimators/galaxy_clustering/src/fastmodules*.so
	rm -f acm/estimators/galaxy_clustering/src/minkowski.c
	rm -f acm/estimators/galaxy_clustering/src/minkowski*.so
	rm -f acm/estimators/galaxy_clustering/src/*.pyc
	rm -f acm/estimators/galaxy_clustering/src/c/*.o
	rm -f acm/estimators/galaxy_clustering/src/c/*.exe
	rm -f acm/estimators/galaxy_clustering/src/pydive*.so
	rm -f acm/estimators/galaxy_clustering/src/pydive.cpp