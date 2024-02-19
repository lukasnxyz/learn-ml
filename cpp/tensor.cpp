#include <iostream>

// just a matrix class (numpy array in cpp)
class Tensor {
    private:
        size_t rows;
        size_t cols;
        double **data;
    public:
        Tensor(size_t n_rows, size_t n_cols) {
            rows = n_rows;
            cols = n_cols;

            // new
        }

        ~Tensor(void) {
            // delete
        }
};
