class Mat {
    private:
        float **mat;

    public:
        Mat(int rows=0, int cols=0) {
            assert(rows != 0);
            assert(cols != 0);

            mat = new *float[rows];

            for(size_t i = 0; i < rows; ++i) {
                mat[i] = new float[cols];
            }
        }

        float **mat(void) {
            return mat;
        }

        ~Mat() {
        }
};
