#include "la.h"

float rand_float(void) {
	return (float) rand()/ (float) RAND_MAX;
}

struct Mat mat_init(size_t rows, size_t cols)
{
	struct Mat m;
	m.rows = rows;
	m.cols = cols;
	m.data = malloc((sizeof(*m.data))*rows*cols);
	assert(m.data != NULL);

	return m;
}

void mat_print(struct Mat m, const char *name)
{
	printf("%s = [\n", name);
	for(size_t i = 0; i < m.rows; ++i) {
		for(size_t j = 0; j < m.cols; ++j) {
			printf(" %lf", MAT_AT(m, i, j));
		}
		printf("\n");
	}
	printf("]\n");
}

void mat_rand(struct Mat m, int low, int high)
{
	for(size_t i = 0; i < m.rows; ++i) {
		for(size_t j = 0; j < m.cols; ++j) {
			MAT_AT(m, i, j) = rand_float() * (high - low) + low;
		}
	}
}

void mat_fill(struct Mat m, float x)
{
	for(size_t i = 0; i < m.rows; ++i) {
		for(size_t j = 0; j < m.cols; ++j) {
			MAT_AT(m, i, j) = x;
		}
	}
}

void mat_dot_mat(struct Mat dest, struct Mat m1, struct Mat m2)
{
	assert(m1.cols == m2.rows);
	assert((dest.rows == m1.rows) && (dest.cols == m2.cols));

	for(size_t i = 0; i < dest.rows; ++i) {
		for(size_t j = 0; j < dest.cols; ++j) {
            float sum = 0;

            for(size_t z = 0; z < m2.rows; ++z) {
				sum += MAT_AT(m1, i, z) * MAT_AT(m2, z, j);
            }

			MAT_AT(dest, i, j) = sum;
		}
	}
}

void mat_add_mat(struct Mat dest, struct Mat m)
{
	for(size_t i = 0; i < dest.rows; ++i) {
		for(size_t j = 0; j < dest.cols; ++j) {
			MAT_AT(dest, i, j) += MAT_AT(m, i, j);
		}
	}
}
