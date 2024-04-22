#include "arr.hpp"

#include <iostream>
#include <iomanip>

#include "util.hpp"

namespace Arr {
	size_t Arr::getRows(void) {
		return _rows;
	}

	size_t Arr::getCols(void) {
		return _cols;
	}

	double **Arr::getArr(void) {
		return _arr;
	}

	void Arr::print(void) {
		std::cout << "Arr::Arr [" << std::endl;

		for(size_t r = 0; r < _rows; ++r) {
			std::cout << std::setprecision(4) << "[" << _arr[r][0];
			for(size_t c = 1; c < _cols; ++c) {
				std::cout << std::setprecision(4) << ", " << _arr[r][c];
			}

			std::cout << "]" << std::endl;
		}
	}

	bool Arr::compare(Arr *in) {
		if(_rows == in->getRows() && _cols == in->getCols())
			return true;

		return false;
	}

	void Arr::transpose(void) {
		double **arr = new double *[_cols];
		for(size_t r = 0; r < _cols; ++r)
			arr[r] = new double [_rows];

		for(size_t r = 0; r < _rows; ++r) {
			for(size_t c = 0; c < _cols; ++c)
				arr[c][r] = _arr[r][c];
		}

		for(size_t r = 0; r < _rows; ++r)
			delete[] _arr[r];
		delete[] _arr;

		_arr = arr;

		size_t t_rows = _rows;
		_rows = _cols;
		_cols = t_rows;
	}

	Arr *Arr::mul(Arr *in) {
		if(!compare(in))
			return NULL;

		double **arr = new double *[_rows];
			for(size_t r = 0; r < _rows; ++r)
				arr[r] = new double [_cols];

		for(size_t r = 0; r < _rows; ++r) {
			for(size_t c = 0; c < _cols; ++c)
				arr[r][c] = _arr[r][c] * in->getArr()[r][c];
		}

		Arr *ret = new Arr(_rows, _cols, arr);
		return ret;
	}

	Arr *Arr::dot(Arr *in) {
		if(in->getRows() == 1 && in->getCols() == 1) {
			double **arr = new double *[_rows];
			for(size_t r = 0; r < _rows; ++r)
				arr[r] = new double [_cols];

			Arr *ret = new Arr(_rows, _cols, arr);
			return ret;
		} else {
			if(_cols != in->getRows())
				return NULL;

			double **arr = new double *[_cols];
			for(size_t r = 0; r < _cols; ++r)
				arr[r] = new double [in->getRows()];

			TODO("Implement multiplication logic");

			Arr *ret = new Arr(_cols, in->getRows(), arr);
			return ret;
		}
	}

	void Arr::dot(double s) {
		for(size_t r = 0; r < _rows; ++r) {
				for(size_t c = 0; c < _cols; ++c)
					_arr[r][c] *= s;
			}
	}

	Arr *Arr::add(Arr *in) {
		if(in->getRows() == 1 && in->getCols() == 1) {
			double **arr = new double *[_rows];
			for(size_t r = 0; r < _rows; ++r)
				arr[r] = new double [_cols];

			for(size_t r = 0; r < _rows; ++r) {
				for(size_t c = 0; c < _cols; ++c)
					arr[r][c] = _arr[r][c] + in->getArr()[0][0];
			}

			Arr *ret = new Arr(_rows, _cols, arr);
			return ret;
		} else {
			if(!compare(in)) // self.dimcompare(in)
				return NULL;

			double **arr = new double *[_rows];
			for(size_t r = 0; r < _rows; ++r)
				arr[r] = new double [_cols];

			for(size_t r = 0; r < _rows; ++r) {
				for(size_t c = 0; c < _cols; ++c)
					arr[r][c] = _arr[r][c] + in->getArr()[r][c];
			}

			Arr *ret = new Arr(_rows, _cols, arr);
			return ret;
		}
	}

	void Arr::add(double s) {
		for(size_t r = 0; r < _rows; ++r) {
				for(size_t c = 0; c < _cols; ++c)
					_arr[r][c] += s;
			}
	}
}
