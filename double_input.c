#include <stdio.h>
#include <stdlib.h>

/* or-gate */
int tSet[][3] = {
    {0, 0, 0},
    {1, 1, 1},
    {0, 1, 1},
    {1, 0, 1}
};

#define setSize ((float)sizeof(tSet)/sizeof(tSet[0]))

float cost(float w1, float w2) {
    float result = 0.0f;
    for(size_t i = 0; i < setSize; i++) {
        float x1 = tSet[i][0];
        float x2 = tSet[i][1];
        float y = w1*x1 + w2*x2;
        float d = y - tSet[i][2];
        result += d*d;
    }
    result /= setSize;
    return result;
}

float randFloat() {
    return (float) rand()/ (float) RAND_MAX;
}

int main() {
    srand(69);
    float w1 = randFloat() * 10 - 5;
    float w2 = randFloat() * 10 - 5;

    float rate = 0.01;
    float h = 0.0001;

    for(int i = 0; i < 10000; i++) {
        float c = cost(w1, w2);
        //printf("w1 :%f, w2: %f, c: %f\n", w1, w2, c);
        printf("%f\n", c);
        float dw1 = (cost(w1 + h, w2) - c)/h;
        float dw2 = (cost(w1, w2 + h) - c)/h;

        w1 -= rate*dw1;
        w2 -= rate*dw2;
    }

    return 0;
}
