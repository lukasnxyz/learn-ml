#include <stdio.h>

#include "../src/la.h"
#include "../src/test.h"

int test_mat_init(void);
int test_mat_dim_compare(void);

/*
void test_mat_init_from_file(FILE *);
void test_mat_fill(Matrix **, const double);
void test_mat_free(Matrix **);
void test_mat_print(Matrix **);
void test_mat_transpose(Matrix **);
void test_mat_dot_mat(Matrix **, Matrix **);
void test_mat_dot_scalar(Matrix **, const double);
void test_mat_add_scalar(Matrix **, const double);
void test_mat_add_mat(Matrix **, Matrix **);
*/

int main(void)
{
  t_new("Test matrix initialize", test_mat_init);
  t_new("Test matrix dimensions compare", test_mat_dim_compare);

  t_end();

  return 0;
}

int test_mat_init(void)
{
  size_t rows = 4;
  size_t cols = 3;

  Matrix *m1 = mat_init(rows, cols);

  for(size_t y = 0; y < rows; ++y) {
    for(size_t x = 0; x < cols; ++x) {
      t_assrt(m1->data[y][x] == 0);
    }
  }

  mat_free(&m1);

  return 0;
}

int test_mat_dim_compare(void)
{
  Matrix *m1 = mat_init(1, 4);
  Matrix *m2 = mat_init(1, 4);
  Matrix *m3 = mat_init(3, 4);
  Matrix *m4 = mat_init(2, 4);

  t_assrt(mat_dim_compare(&m1, &m2));
  t_assrt(mat_dim_compare(&m3, &m4) == 0);

  mat_free(&m1);
  mat_free(&m2);
  mat_free(&m3);
  mat_free(&m4);

  return 0;
}
