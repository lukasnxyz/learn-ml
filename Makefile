# Compiler and flags
CC = gcc
CFLAGS = -Wall -Werror -I./src

# Directories
SRC_DIR = src
OBJ_DIR = obj
BIN_DIR = bin
TEST_DIR = tests
BIN_TESTS = tests/bin

# Source files
SRC_FILES = $(wildcard $(SRC_DIR)/*.c)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC_FILES))
TEST_SRC_FILES = $(wildcard $(TEST_DIR)/*.c)
TEST_BIN_FILES = $(patsubst $(TEST_DIR)/%.c,$(TEST_DIR)/$(BIN_DIR)/%,$(TEST_SRC_FILES))

# Targets
all: $(BIN_DIR)/main

$(BIN_DIR)/main: $(OBJ_FILES) main.c
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST_BIN_FILES)
	@echo "Running tests..."
	@for test in $^ ; do \
		$$test ; \
	done

$(TEST_DIR)/$(BIN_DIR)/%: $(TEST_DIR)/%.c $(OBJ_FILES)
	@mkdir -p $(BIN_TESTS)
	$(CC) $(CFLAGS) $^ -o $@

# Clean
clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR) $(BIN_TESTS)

.PHONY: all test clean
