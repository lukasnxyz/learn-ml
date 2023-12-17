CC=gcc
CFLAGS=-Wall -Wextra

SRC=src
OBJ=obj
BIN=bin
TEST=tests
TEST_BIN=tests/bin

SRC_FILES=$(wildcard $(SRC)/*.c)
OBJ_FILES=$(patsubst $(SRC)/%.c,$(OBJ)/%.o,$(SRC_FILES))
TEST_SRC_FILES=$(wildcard $(TEST)/*.c)
TEST_BIN_FILES=$(patsubst $(TEST)/%.c,$(TEST)/$(BIN)/%,$(TEST_SRC_FILES))

all: $(BIN)/nn

$(BIN)/nn: $(OBJ_FILES) nn.c
	@mkdir -p $(BIN)
	$(CC) $(CFLAGS) $^ -o $@

$(OBJ)/%.o: $(SRC)/%.c
	@mkdir -p $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST_BIN_FILES)
	@echo "Running tests..."
	@for test in $^ ; do \
		$$test ; \
	done

$(TEST)/$(BIN)/%: $(TEST)/%.c $(OBJ_FILES)
	@mkdir -p $(TEST_BIN)
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm -rf $(BIN) $(OBJ) $(TEST_BIN)

.PHONY: all test clean
