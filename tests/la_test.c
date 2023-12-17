#include <stdio.h>

#include "../src/la.h"
#include "../src/test.h"

int test_mat_init(void);
int test_mat_dim_compare(void);
int test_mat_fill(void);
int test_mat_transpose(void);
int test_mat_dot_mat(void);
int test_mat_dot_scalar(void);
int test_mat_add_mat(void);
int test_mat_add_scalar(void);

int main(void)
{
  t_new("Test matrix initialize", test_mat_init);
  t_new("Test matrix dimensions compare", test_mat_dim_compare);
  t_new("Test matrix fill", test_mat_fill);
  t_new("Test matrix transpose", test_mat_transpose);
  t_new("Test matrix multiply", test_mat_dot_mat);
  t_new("Test matrix multiply scalar", test_mat_dot_scalar);
  t_new("Test matrix addition", test_mat_add_mat);
  t_new("Test matrix add scalar", test_mat_add_scalar);

  t_end();

  return 0;
}

int test_mat_init(void)
{
  size_t rows = 4;
  size_t cols = 3;

  Matrix *m1 = mat_init(rows, cols);

  for(size_t y = 0; y < m1->rows; ++y) {
    for(size_t x = 0; x < m1->cols; ++x) {
      t_assrt((m1->data[y][x] == 0));
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
  t_assrt((mat_dim_compare(&m3, &m4) == 0));

  mat_free(&m1);
  mat_free(&m2);
  mat_free(&m3);
  mat_free(&m4);

  return 0;
}

int test_mat_fill(void)
{
  size_t rows = 25;
  size_t cols = 20;

  Matrix *m1 = mat_init(rows, cols);
  mat_fill(&m1, 4);

  for(size_t y = 0; y < m1->rows; ++y) {
    for(size_t x = 0; x < m1->cols; ++x) {
      t_assrt((m1->data[y][x] == 4.0));
    }
  }

  mat_free(&m1);

  return 0;
}

int test_mat_transpose(void)
{
  Matrix *m1 = mat_init(2, 2);
  m1->data[0][1] = 4;

  Matrix *m2 = mat_transpose(&m1);
  t_assrt((m2->data[1][0] == 4));

  mat_free(&m1);
  mat_free(&m2);

  return 0;
}

int test_mat_dot_mat(void)
{
  Matrix *m1 = mat_init(3, 2);
  Matrix *m2 = mat_init(2, 3);

  mat_fill(&m1, 3);
  mat_fill(&m2, 4);

  Matrix *m3 = mat_dot_mat(&m1, &m2);

  t_assrt((m3->rows == 3));
  t_assrt((m3->cols == 3));

  for(size_t y = 0; y < m3->rows; ++y) {
    for(size_t x = 0; x < m3->cols; ++x) {
      t_assrt((m3->data[y][x] == 24.0));
    }
  }

  mat_free(&m1);
  mat_free(&m2);
  mat_free(&m3);

  return 0;
}

int test_mat_dot_scalar(void)
{
  Matrix *m1 = mat_init(17, 19);
  mat_fill(&m1, 1.0);
  const double lambda = 4.4;

  mat_dot_scalar(&m1, lambda);

  for(size_t y = 0; y < m1->rows; ++y) {
    for(size_t x = 0; x < m1->cols; ++x) {
      t_assrt((m1->data[y][x] == lambda));
    }
  }

  mat_free(&m1);

  return 0;
}

int test_mat_add_scalar(void)
{
  Matrix *m1 = mat_init(32, 12);
  const double lambda = 4.4;

  mat_add_scalar(&m1, lambda);

  for(size_t y = 0; y < m1->rows; ++y) {
    for(size_t x = 0; x < m1->cols; ++x) {
      t_assrt((m1->data[y][x] == lambda));
    }
  }

  mat_free(&m1);

  return 0;
}

int test_mat_add_mat(void)
{
  Matrix *m1 = mat_init(2, 3);
  Matrix *m2 = mat_init(2, 3);
  mat_fill(&m1, 4);
  mat_fill(&m2, 9);

  mat_add_mat(&m1, &m2);

  for(size_t y = 0; y < m1->rows; ++y) {
    for(size_t x = 0; x < m1->cols; ++x) {
      t_assrt((m1->data[y][x] == 13));
    }
  }

  mat_free(&m1);
  mat_free(&m2);

  return 0;
}
