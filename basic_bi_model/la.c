#include "la.h"

Matrix *matInit(unsigned int rows, unsigned int columns) {
    Matrix *newMat = malloc(sizeof(Matrix));
    newMat->rows = rows;
    newMat->columns = columns;
    newMat->data = malloc(rows * sizeof(double*));

    for(int i = 0; i < rows; i++) {
        newMat->data[i] = malloc(columns * sizeof(double));
    }

    return newMat;
}

Matrix *matInitFromFile(FILE *fp) {
    int rows, columns = 0;
	char entry[MAX];

	fgets(entry, MAX, fp);
	rows = atoi(entry);
	fgets(entry, MAX, fp);
	columns = atoi(entry);

	Matrix *newMat = matInit(rows, columns);
	for (int y = 0; y < newMat->rows; y++) {
		for (int x = 0; x < newMat->columns; x++) {
			fgets(entry, MAX, fp);
			newMat->data[y][x] = strtod(entry, NULL);
            //printf("%lf\n", newMat->data[y][x]);
		}
	}


    return newMat;
}

void matDInit(Matrix *m1) {
    for(int i = 0; i < m1->rows; i++) {
        free(m1->data[i]);
    }
    free(m1->data);
    //free(m1);
}

int matCompareDimensions(Matrix *m1, Matrix *m2) {
    if(m1->rows != m2->rows || m1-> columns != m2->columns) {
        printf("matCompareDimensions: The two matricies are not of the same dimensions!\n");
        return 0;
    }
    return 1;
}

void matPrint(Matrix *m1) {
    for(int y = 0; y < m1->rows; y++) {
        for(int x = 0; x < m1->columns; x++) {
            printf("%lf ", m1->data[y][x]);
        }
        putchar('\n');
    }
}

Matrix *matTranspose(Matrix *m1) {
    Matrix *newMatrix = matInit(m1->columns, m1->rows);
    for(int y = 0; y < m1->rows; y++) {
        for(int x = 0; x < m1->columns; x++) {
            newMatrix->data[x][y] = m1->data[y][x];
        }
    }

    return newMatrix;
}

Matrix *matDotScalar(Matrix *m1, double scalar) {
    Matrix *newMat = m1;
    for(int y = 0; y < m1->rows; y++) {
        for(int x = 0; x < m1->columns; x++) {
            newMat->data[y][x] *= scalar;
        }
    }

    return newMat;
}

Matrix *matDotMat(Matrix *m1, Matrix *m2) {
    if(m1->columns != m2->rows) {
        printf("matDotMat: The columns of the first matrix do not equal the rows of the second matrix!\n");
        return 0;
    }

    Matrix *newMatrix = matInit(m1->rows, m2->columns);

    for(int y = 0; y < m1->rows; y++) {
        for(int x = 0; x < m2->columns; x++) {
            double sum = 0;
            for(int z = 0; z < m2->rows; z++) {
                sum += m1->data[y][z] * m2->data[z][x];
            }
            newMatrix->data[y][x] = sum;
        }
    }

    return newMatrix;
}

Matrix *matAddScalar(Matrix *m1, double scalar) {
    Matrix *newMat = m1;
    for(int y = 0; y < newMat->rows; y++) {
        for(int x = 0; x < newMat->columns; x++) {
            newMat->data[y][x] += scalar;
        }
    }

    return newMat;
}

Matrix *matAddMat(Matrix *m1, Matrix *m2) {
    matCompareDimensions(m1, m2);
    Matrix *newMat = matInit(m1->rows, m1->columns);
    for(int y = 0; y < m1->rows; y++) {
        for(int x = 0; x < m1->columns; x++) {
            newMat->data[y][x] = m1->data[y][x] + m2->data[y][x];
        }
    }

    return newMat;
}

Matrix *matSubtractScalar(Matrix *m1, double scalar) {
    Matrix *newMat = m1;
    for(int y = 0; y < newMat->rows; y++) {
        for(int x = 0; x < newMat->columns; x++) {
            newMat->data[y][x] -= scalar;
        }
    }

    return newMat;
}

Matrix *matSubtractMat(Matrix *m1, Matrix *m2) {
    matCompareDimensions(m1, m2);
    Matrix *newMat = matInit(m1->rows, m1->columns);
    for(int y = 0; y < m1->rows; y++) {
        for(int x = 0; x < m1->columns; x++) {
            newMat->data[y][x] = m1->data[y][x] - m2->data[y][x];
        }
    }

    return newMat;
}
