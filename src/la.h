#ifndef LA_H
#define LA_H

#include <stdio.h>
#include <stdlib.h>

#define MAX 100

typedef struct {
    size_t rows;
    size_t cols;
    double **data;
} Matrix;

Matrix *mat_init(const size_t, const size_t);
Matrix *mat_init_from_file(FILE *);
void mat_fill(Matrix **, const double);
void mat_free(Matrix **);
void mat_print(Matrix **);
int mat_dim_compare(Matrix **, Matrix **);
Matrix *mat_transpose(Matrix **);

Matrix *mat_dot_mat(Matrix **, Matrix **);
void mat_dot_scalar(Matrix **, const double);
void mat_add_scalar(Matrix **, const double);
void mat_add_mat(Matrix **, Matrix **);

#endif /* LA_H */
