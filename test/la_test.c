#include <stdio.h>

#include "../src/la.h"

/*
#define NUM_OF_TESTS 11
int test_new_mat(const size_t, const size_t);
int test_mat_init_from_file(FILE *);
int test_mat_fill(Matrix **, const double);
int test_mat_free(Matrix **);
int test_mat_print(Matrix **);
int test_mat_dim_compare(Matrix **, Matrix **);
int test_mat_transpose(Matrix **);
int test_mat_dot_mat(Matrix **, Matrix **);
int test_mat_dot_scalar(Matrix **, const double);
int test_mat_add_scalar(Matrix **, const double);
int test_mat_add_mat(Matrix **, Matrix **);
*/

int test(void);

int main(void)
{
  //printf("Number of tests run: %d\n", NUM_OF_TESTS);
  printf("Testing...\n");

  return test();
}

int test(void)
{
  unsigned int passed = 5;

  return passed;
}
