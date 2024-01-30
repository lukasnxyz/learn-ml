#include <iostream>

float foo(float a = 0, float b = 1, float c = 2) {
    /*
     * you can pass default values to parameters
     * if less function params are given, default values are used
     */

    return a + b + c;
}

/*
 * namespaces group classes, functions, and variables under a common scope name
 * that can be referneced elsewhere
 */
namespace first {
    int var = 5;
}

namespace second {
    int var = 3;
}

void foo1(void) {
    std::cout << "first: " << first::var << ", second: " << second::var << std::endl;
}

/*
 * function overloading
 * it's illegal to overload functions based on return type
 */
float add(float a, float b) {
    return a + b;
}

int add(int a, int b) {
    return a + b;
}

/*
 * const and inline
 */
#define SQUARE(x) x*x
const int two = 2;

// for macros:
int inline square(int x) {
    return x*x;
}

// how to mix c and c++
#ifdef __cplusplus
extern "C" {
#endif
#include "stdio.h"

#ifdef __cplusplus
}
#endif

void swap(int &a, int &b) {
    int tmp = b;
    b = a;
    a = tmp;
}

/*
 * int add( int a, int b ) { return a+b; }
 * int add( int a, int b, int c=0 ) { return a+b+c; }
 * This is NOT legal because obvious...
 */

#include <cassert>
const double divide(const double a, const double b) {
    assert(b != 0);

    return a / b;
}

// CLASSES
// Can have 0-n constructors
// Can be only 1 destructor
class Foo {
    // public: can be accessed by everyone
    // protected: can be accessed by only derived classes
    // private: can not be accessed outside of class
    public:
        Foo(void) {
            std::cout << "Foo constructor 1" << std::endl;
        }

        Foo(int value) {
            std::cout << "Foo constructor " << value << std::endl;
        }

        // define destructor with ~
        // never needs to be explicitly called
        ~Foo(void) {
            std::cout << "Foo destructor called" << std::endl;
        }

        // init lists
        //Foo(int value = 0) : _value(value) { };

        // operator overloading
};

/*
 * friends are either functions or other classes that are granted privileged access to a class
 */

class Foo1 {
public:
    friend std::ostream& operator<< (std::ostream& output, Foo1 const & that) {
        return output << that._value;
    }

private:
    double _value;
};

int main(int argc, char **argv) {
    int i;
    std::cout << "Please enter a number: ";
    std::cin >> i;
    std::cout << "The value you entered is " << i << std::endl;

    int *a = new int;
    delete a; // calls deconstructor and deallocates memory
    /*
     * new and delete is the c++ version of malloc and free
     * don't ever cross use like new/free or malloc/delete
     */

    int *b = new int[5];
    delete [] b;

    int x;
    int &foo = x; // a refernece variable
                  // as long as the aliased variable lived (x), you can use indifferently
                  // the variable or the alias
                  // very useful for fuction arguments, saves from copying params into the stack
                  // when calling the function

    foo1();
    int result = SQUARE(3+3);

    const size_t n = 2;
    int **array = new int *[n];

    for(size_t i = 0; i < n; ++i) {
        array[i] = new int[n];
    }

    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < n; ++j) {
            array[i][j] = i * n + j;
        }
    }

    for(size_t i = 0; i < n; ++i) {
        for(size_t j = 0; j < n; ++j) {
            std::cout << "array[" << i << ", " << j << "] = "
                << array[i][j] << std::endl;
        }
    }

    for(size_t i = 0; i < n; ++i) {
        delete [] array[i];
    }
    delete [] array;

    int z = 1;
    int y = 2;

    std::cout << "z: " << z << ", y: " << y << std::endl;
    swap(z, y);
    std::cout << "z: " << z << ", y: " << y << std::endl;

    int i1 = 1;
    int i2 = 2;

    int const *p1 = &i1; // pointer with constant int value
    int *const p2 = &i1; // pointer with constant memory location
    int const *const p3 = &i1; // pointer with constant memory location and int value

    p1 = &i2; // correct
    *p2 = 2; // correct

    Foo foo_1, foo_2(2);
    Foo1 foo_3;
    std::cout << "Foo object: " << foo_3 << std::endl;

    return 0;
}
