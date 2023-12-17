# ml

CC=gcc
CFLAGS=-Wall -Wextra -g
SRC=src
OBJ=obj
BIN=bin

SRCS=$(wildcard $(SRC)/*.c)
OBJS=$(patsubst $(SRC)/%.c, $(OBJ)/%.o, $(SRCS))
BINS=$(BIN)/nn

all: $(BINS)

$(BINS): $(OBJS)
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $(OBJS) -o $@

$(OBJ)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BIN)/ $(OBJ)/

.PHONY: all clean
