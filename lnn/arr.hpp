#ifndef _ARR_HPP_
#define _ARR_HPP_

#include <vector>

namespace Arr {
	class Arr {
		private:
			size_t _rows;
			size_t _cols;
			double **_arr; // acounts for scalar, vector, and matrix

		public:
			Arr(size_t rows, size_t cols) {
				_rows = rows;
				_cols = cols;

				_arr = new double *[_rows];
				for(size_t r = 0; r < _rows; ++r)
					_arr[r] = new double[_cols];
			}

			Arr(size_t rows, size_t cols, double fill) {
				_rows = rows;
				_cols = cols;

				_arr = new double *[_rows];
				for(size_t r = 0; r < _rows; ++r)
					_arr[r] = new double[_cols];

				for(size_t r = 0; r < _rows; ++r) {
					for(size_t c = 0; c < _cols; ++c)
						_arr[r][c] = fill;
				}
			}

			Arr(size_t rows, size_t cols, double **arr) {
				_rows = rows;
				_cols = cols;
				_arr = arr;
			}

			Arr(std::vector<std::vector<double>> arr) {
				_rows = arr.size();
				_cols = arr[0].size();

				_arr = new double *[_rows];
				for(size_t r = 0; r < _rows; ++r)
					_arr[r] = new double[_cols];

				for(size_t r = 0; r < _rows; ++r) {
					for(size_t c = 0; c < _cols; ++c)
						_arr[r][c] = arr[r][c];
				}
			}

			// construct from file
			/* Arr() {} */

			// copy constructor

			~Arr() {
				for(size_t r = 0; r < _rows; ++r)
					delete[] _arr[r];

				delete[] _arr;
			}

			size_t getRows(void);
			size_t getCols(void);
			double **getArr(void);
			void print(void);

			bool compare(Arr *);
			void transpose(void);

			Arr *mul(Arr *);
			Arr *dot(Arr *);
			void dot(double);
			Arr *add(Arr *);
			void add(double);
	};
}

#endif // _ARR_HPP_
