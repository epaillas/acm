# Compiler choice:
CC    = gcc
MPICC = mpicc
CFLAGS = -O2 -g -ansi -Wno-unused-result -std=c99
MPICFLAGS = -O2 -g -ansi -std=c99

MLIBS	=   -lm

EXES = jozov-grid.exe

jozov-grid.exe: jozov-grid.o
	$(CC) $(CFLAGS) -o jozov-grid.exe jozov-grid.o $(MLIBS)

all: $(EXES)

install: $(EXES)

.c.o:
	$(CC) -c -o $@ $(CFLAGS) $(QINC) $*.c

clean:
	rm -f $(EXES) *.o *~
