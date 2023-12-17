#include "la.h"

Matrix *mat_init(const size_t rows, const size_t cols)
{
  Matrix *new_mat = (Matrix *)malloc(sizeof(Matrix));
  if(new_mat == NULL) {
    return NULL;
  }

  new_mat->rows = rows;
  new_mat->cols = cols;
  new_mat->data = (double **)calloc(new_mat->rows, sizeof(double *));
  if(new_mat->data == NULL) {
    return NULL;
  }

  for(size_t i = 0; i < new_mat->rows; ++i) {
    new_mat->data[i] = (double *)calloc(new_mat->cols, sizeof(double));
    if(new_mat->data[i] == NULL) {
      mat_free(&new_mat);

      return NULL;
    }
  }

  return new_mat;
}

Matrix *mat_init_from_file(FILE *fp)
{
  size_t rows, cols = 0;
  char entry[MAX];

  rewind(fp);

  fgets(entry, MAX, fp);
	rows = atoi(entry);

	fgets(entry, MAX, fp);
	cols = atoi(entry);

  Matrix *new_mat = mat_init(rows, cols);

  for (size_t y = 0; y < new_mat->rows; ++y) {
    for (size_t x = 0; x < new_mat->cols; ++x) {
			fgets(entry, MAX, fp);

      new_mat->data[y][x] = strtod(entry, NULL);
    }
  }

  return new_mat;
}

void mat_fill(Matrix **m1, const double num)
{
  for(size_t y = 0; y < (*m1)->rows; ++y) {
    for(size_t x = 0; x < (*m1)->cols; ++x) {
      (*m1)->data[y][x] = num;
    }
  }
}

void mat_free(Matrix **m1)
{
  for(size_t i = 0; i < (*m1)->rows; ++i) {
    free((*m1)->data[i]);
  }

  free((*m1)->data);
  free(*m1);
}

void mat_print(Matrix **m1)
{
  for(size_t y = 0; y < (*m1)->rows; ++y) {
    for(size_t x = 0; x < (*m1)->cols; ++x) {
      printf("%.3lf ", (*m1)->data[y][x]);
    }

    putchar('\n');
  }
}

int mat_dim_compare(Matrix **m1, Matrix **m2)
{
  if((*m1)->rows != (*m2)->rows || (*m1)->cols != (*m2)->cols) {
    return 0;
  }

  return 1;
}

Matrix *mat_transpose(Matrix **m1)
{
  Matrix *new_matrix = mat_init((*m1)->cols, (*m1)->rows);

  for(size_t y = 0; y < (*m1)->rows; ++y) {
    for(size_t x = 0; x < (*m1)->cols; ++x) {
      new_matrix->data[x][y] = (*m1)->data[y][x];
    }
  }

  return new_matrix;
}

void mat_dot_scalar(Matrix **m1, const double scalar)
{
  for(size_t y = 0; y < (*m1)->rows; ++y) {
    for(size_t x = 0; x < (*m1)->cols; ++x) {
      (*m1)->data[y][x] *= scalar;
    }
  }
}

Matrix *mat_dot_mat(Matrix **m1, Matrix **m2)
{
  if((*m1)->cols != (*m2)->rows) {
    printf("%s: The cols of the first matrix do not equal the rows of the second matrix!\n", __func__);

    return NULL;
  }

  Matrix *new_matrix = mat_init((*m1)->rows, (*m2)->cols);

  for(size_t y = 0; y < (*m1)->rows; ++y) {
    for(size_t x = 0; x < (*m2)->cols; ++x) {
      double sum = 0;

      for(size_t z = 0; z < (*m2)->rows; z++) {
        sum += (*m1)->data[y][z] * (*m2)->data[z][x];
      }
      new_matrix->data[y][x] = sum;
    }
  }

  return new_matrix;
}

void mat_add_scalar(Matrix **m1, const double scalar)
{
  for(size_t y = 0; y < (*m1)->rows; ++y) {
    for(size_t x = 0; x < (*m1)->cols; ++x) {
      (*m1)->data[y][x] += scalar;
    }
  }
}

void mat_add_mat(Matrix **m1, Matrix **m2)
{
  if(!mat_dim_compare(m1, m2)) {
    printf("%s: Matrices don't match sizes!", __func__);

    return;
  }

  for(size_t y = 0; y < (*m1)->rows; ++y) {
    for(size_t x = 0; x < (*m1)->cols; ++x) {
      (*m1)->data[y][x] += (*m2)->data[y][x];
    }
  }
}
