# ml

CC=gcc
CFLAGS=-Wall -Wextra -g
OPTS=

LA_SRC=la.c
NN_SRC=nn.c
TEST_SRC=la_test.c

LA_OBJS=$(LA_SRC:.c=.o)
NN_OBJS=$(NN_SRC:.c=.o)
TEST_OBJS=$(TEST_SRC:.c=.o)

SRCDIR=src/
TESTDIR=test/

all: la test

%.o: $(SRCDIR)%.c $(TESTDIR)%.c
	$(CC) -c $(CFLAGS) $< -o $@ $(OPTS)

la: $(LA_OBJS)
	$(CC) -o $@ $^ $(OPTS)

run: all
	 @printf '[\e[0;36mTest\e[0m] Executing binary...\n'
	 @./la

test: $(TEST_OBJS)
	$(CC) -c $(CFLAGS) $< -o $@ $(OPTS)

clean:
	rm -f *.o la la_test
