# compiler choice
CC    = gcc

all: fastmodules c

.PHONY : fastmodules

c:
	make -C acm/src all

fastmodules:
	python acm/src/setup.py build_ext --inplace
	mv fastmodules*.so acm/src/.

clean:
	rm -f acm/src/*.*o
	rm -f acm/src/fastmodules.c
	rm -f acm/src/fastmodules*.so
	rm -f acm/src/*.pyc
	rm -f acm/src/c/*.o
	rm -f acm/src/c/*.exe