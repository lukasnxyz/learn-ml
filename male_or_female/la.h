#ifndef LA_H
#define LA_H

#include <stdio.h>
#include <stdlib.h>

#define MAX 100

/*
 * rows x columns
 */

typedef struct {
    unsigned int rows, columns;
    double **data;
} Matrix;

/* Basic library functions */
Matrix *matInit(unsigned int, unsigned int);
Matrix *matInitFromFile(FILE *);
void matDInit(Matrix *);
void matPrint(Matrix *);

/* Operator functions */
Matrix *matDotScalar(Matrix *, double);
Matrix *matDotMat(Matrix *, Matrix *);

Matrix *matAddScalar(Matrix *, double);
Matrix *matAddMat(Matrix *, Matrix *);

Matrix *matSubtractScalar(Matrix *, double);
Matrix *matSubtractMat(Matrix *, Matrix *);

int matCompareDimensions(Matrix *, Matrix *);
Matrix *matTranspose(Matrix *);

#endif /* LA_H */
