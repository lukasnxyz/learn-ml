#include <stdio.h>

#include "la.h"

/* height(m), weight(lbs) */ /* male or female, 1 || 0 */

int main() {
    FILE *fp = fopen("input.txt", "r");
    Matrix *input = matInitFromFile(fp);

    matPrint(input);
    return 0;
}
