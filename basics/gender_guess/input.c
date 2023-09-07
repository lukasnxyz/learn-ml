#include <stdio.h>
#include <time.h>

#include "../libml/la.h"

/* height(cm), weight(kg) */ /* male or female, 1 || 0 */

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
    FILE *fp = fopen("500_Person_Gender_Height_Weight_Index.csv", "r");
    /*FILE *fp = fopen("weight-height.csv", "r");*/
    Matrix *input = matInitFromFile(fp); /* rows, columns, height(cm), weight(kg), male/female 1/0 */

    srand(time(0));
	float w1 = randFloat() * 5;
	float w2 = randFloat() * 5;

    float rate = 1e-7; /* learning rate */
    float h = 1e-4;

    for(int i = 0; i < 10000*10; i++) {
        float c = cost(w1, w2, input);

        printf("w1 :%f, w2: %f, c: %.15f\n", w1, w2, c);

        float dw1 = (cost(w1 + h, w2, input) - c)/h; /* derivative */
        float dw2 = (cost(w1, w2 + h, input) - c)/h;

        w1 -= rate*dw1; /* distances */
        w2 -= rate*dw2;
    }

    float height = 184;
    float weight = 83;
    printf("Height: %.2lfFt, Weight: %.2lfkg, Gender: %s\n", height/30.48, weight, estimate(w1, w2, height, weight));

    return 0;
}
