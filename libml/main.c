#include <stdio.h>
#include <time.h>

#include "la.h"

struct Xor {
	struct Mat a0;

	struct Mat w1, b1, a1;
	struct Mat w2, b2, a2;
};

void forward_xor(struct Xor m)
{
	mat_dot(m.a1, m.a0, m.w1);
	mat_sum(m.a1, m.b1);
	mat_sig(m.a1);

	mat_dot(m.a2, m.a1, m.w2);
	mat_sum(m.a2, m.b2);
	mat_sig(m.a2);
}

float cost(struct Xor m, struct Mat ti, struct Mat to)
{
	assert(ti.rows == to.rows);
	size_t n = ti.rows;

	for(size_t i = 0; i < n; ++i) {
	}
}

int main(void)
{
	srand(time(0));

	struct Xor m;

	m.a0 = mat_init(1, 2);

	m.w1 = mat_init(2, 2);
	m.b1 = mat_init(1, 2);
	m.a1 = mat_init(1, 2);

	m.w2 = mat_init(2, 1);
	m.b2 = mat_init(1, 1);
	m.a2 = mat_init(1, 1);

	mat_rand(m.w1, 0, 1);
	mat_rand(m.b1, 0, 1);
	mat_rand(m.w2, 0, 1);
	mat_rand(m.b2, 0, 1);

	for(size_t i = 0; i < 2; ++i) {
		for(size_t j = 0; j < 2; ++j) {
			MAT_AT(m.a0, 0, 0) = i;
			MAT_AT(m.a0, 0, 1) = j;
			forward_xor(m);

			float y = *m.a2.data;

			printf("%zu ^ %zu = %f\n", i, j, y);
		}
	}

	return 0;
}
