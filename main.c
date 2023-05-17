#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * single input ->
 * single layer ->
 * single output
 */

float tSet[][2] = {
    {0, 0},
    {1, 2},
    {2, 4},
    {3, 6},
    {4, 8}
};

#define tSet_size (sizeof(tSet)/sizeof(tSet[0]))

float randFloat() {
    return (float) rand()/ (float) RAND_MAX;
}

float cost(float w) {
    float result = 0.0f;
    for(size_t i = 0; i < tSet_size; i++) {
        float x = tSet[i][0];
        float y = w*x;
        float d = y - tSet[i][1];
        result += d*d;
    }
    result /= tSet_size;

    return result;
}

int main() {
    srand(time(0))
    float w = randFloat() * 10.0f; /* weight */
    float h = 1e-3;
    float rate = 1e-2;

    for(size_t i = 0; i < 500; i++) {
        float dcost = (cost(w + h) - cost(w))/h;
        w -= rate*dcost;
        printf("cost: %f, w: %f\n", cost(w), w);
    }

    return 0;
}
