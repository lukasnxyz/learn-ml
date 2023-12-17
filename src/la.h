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

extern Matrix *mat_init(const size_t, const size_t);
extern Matrix *mat_init_from_file(FILE *);
extern void mat_fill(Matrix **, const double);
extern void mat_free(Matrix **);
extern void mat_print(Matrix **);
extern int mat_dim_compare(Matrix **, Matrix **);
extern Matrix *mat_transpose(Matrix **);

extern Matrix *mat_dot_mat(Matrix **, Matrix **);
extern void mat_dot_scalar(Matrix **, const double);
extern void mat_add_scalar(Matrix **, const double);
extern void mat_add_mat(Matrix **, Matrix **);

#endif /* LA_H */
