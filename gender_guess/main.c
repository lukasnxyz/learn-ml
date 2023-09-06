#include <stdio.h>
#include <time.h>

#include "la.h"

/* rows, columns, height(cm), weight(kg) */ /* male or female, 1 || 0 */

struct Ml_Data {
	float *weights;
	float *weights_derived;
	float learning_rate;
	float *xvalues;
};

float cost(float w1, float w2, Matrix *matIn) {
	float result = 0.0f;

	for(size_t i = 0; i < matIn->rows; i++) {
		float x1 = matIn->data[i][0];
		float x2 = matIn->data[i][1];
		float y = w1*x1 + w2*x2;
		float d = y - matIn->data[i][2];

		result += d*d;
	}
	result /= matIn->rows;

	return result;
}

float rand_float() {
	return (float) rand()/ (float) RAND_MAX;
}

char *estimate(float *weights, float x1, float x2) {
	float y = weights[0]*x1 + weights[1]*x2;
	if(y < 0.5) {
		return "Female";
	}

	return "Male";
}

struct Ml_Data train(struct Ml_Data data, unsigned int iterations) {
	float h = 1e-2;

	/* Train */
	for(int i = 0; i < unsigned; ++i) {
		float c = cost(data.weights[0], data.weights[1], input);

		printf("weights[0] :%f, weights[1]: %f, cost: %f\n",
				data.weights[0], data.weights[1], c);

		/*data.*/
		float weights_derived[2] = {
			(cost(data.weights[0] + h, data.weights[1], input) - c)/h, /* weights[0] + h */
			(cost(data.weights[0], data.weights[1] + h, input) - c)/h /* weights[1] + h */
		};

		/* Distances */
		data.weights[0] -= data.learning_rate*weights_derived[0];
		data.weights[1] -= data.learning_rate*weights_derived[1];
	}
}

int main() {
	FILE *fp = fopen("weight-height.csv", "r");
	if(fp == NULL) {
		printf("There was an error opening the file!");
		return 1;
	}

	Matrix *input = matInitFromFile(fp);
	struct Ml_Data data;
	srand(time(0));

	float weights[2] = {(rand_float() * 100), (rand_float() * 100)};
	data.weights = weights;

	data.learning_rate = 1e-7;

	train(data, 10000*10);

	/* Test model */
	float test_height = 120;
	float test_weight = 60;

	printf("Height: %.2lfcm, Weight: %.2lfkg, Gender: %s\n",
			test_height, test_weight, estimate(data.weights,
			test_height, test_weight));

	return 0;
}
