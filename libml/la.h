#ifndef LA_H
#define LA_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define MAT_AT(m, i, j) (m).data[(i)*(m).cols + (j)]

/* rows x columns */
struct Mat {
	size_t rows, cols;
	float *data;
};

struct Mat mat_init(size_t, size_t);
void mat_rand(struct Mat, int, int);
void mat_fill(struct Mat, float x);
void mat_dot_mat(struct Mat, struct Mat, struct Mat);
void mat_add_mat(struct Mat, struct Mat);

void mat_print(struct Mat, const char *);
#define MAT_PRINT(m) mat_print(m, #m)

float rand_float(void);

#endif /* LA_H */
