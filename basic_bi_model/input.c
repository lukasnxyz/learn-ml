#include <stdio.h>
#include <time.h>

#include "la.h"

/* height(m), weight(lbs) */ /* male or female, 1 || 0 */

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

float randFloat() {
    return (float) rand()/ (float) RAND_MAX;
}

char *estimate(float w1, float w2, float x1, float x2) {
    float y = w1*x1 + w2*x2;
    if(y < 0.5) {
        return "Female";
    }

    return "Male";
}

int main() {
    FILE *fp = fopen("input.txt", "r");
    Matrix *input = matInitFromFile(fp);

    srand(time(0));
    float w1 = randFloat() * 10;
    float w2 = randFloat() * 10;

    float rate = 1e-8;
    float h = 1e-5;

    for(int i = 0; i < 10000*100; i++) {
        float c = cost(w1, w2, input);

        //printf("w1 :%f, w2: %f, c: %f\n", w1, w2, c);
        //printf("%f\n", c);

        float dw1 = (cost(w1 + h, w2, input) - c)/h;
        float dw2 = (cost(w1, w2 + h, input) - c)/h;

        w1 -= rate*dw1;
        w2 -= rate*dw2;
    }

    float height = 162.56;
    float weight = 130;
    printf("Height: %.2lfFt, Weight: %.2lflbs, Gender: %s\n", height/30.48, weight, estimate(w1, w2, height, weight));

    return 0;
}
