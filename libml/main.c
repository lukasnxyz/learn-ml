#include <stdio.h>
#include <time.h>

#include "la.h"

int main(void) {
	srand(time(0));

	struct Mat w1 = mat_init(2, 2);
	struct Mat b1 = mat_init(1, 2);
	struct Mat w2 = mat_init(2, 1);
	struct Mat b2 = mat_init(1, 1);

	mat_rand(w1, 0, 1);
	mat_rand(b1, 0, 1);
	mat_rand(w2, 0, 1);
	mat_rand(b2, 0, 1);

	MAT_PRINT(w1);
	MAT_PRINT(b1);
	MAT_PRINT(w2);
	MAT_PRINT(b2);

	free(w1.data);
	free(b1.data);
	free(w2.data);
	free(b2.data);

	return 0;
}
