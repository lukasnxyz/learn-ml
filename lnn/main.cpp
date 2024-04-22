#include <iostream>
#include <vector>

#include "arr.hpp"

int main(void) {
	Arr::Arr *a = new Arr::Arr(2, 2, 4.0);
	Arr::Arr *b = new Arr::Arr(2, 2, 2.0);

	Arr::Arr *c = b->add(a);

	if(c != NULL)
		c->print();
	else
		std::cout << "NULL" << std::endl;

	return 0;
}
