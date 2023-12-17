#ifndef TEST_H
#define TEST_H

#include <stdio.h>
#include <time.h>

static unsigned int total_tests = 0;
static unsigned int failed_tests = 0;
static double total_time = 0;

#define t_end() do {\
  if(failed_tests == 0) {\
    printf("All tests passed ");\
  } else if(failed_tests > 0) {\
    printf("Tests failed ");\
  }\
  printf("(%d/%d) ", (total_tests - failed_tests), total_tests);\
  printf("in %lf seconds.\n", total_time);\
} while(0)

#define t_new(name, test_func) do{\
  ++total_tests;\
  clock_t t = clock();\
  unsigned int failed = test_func();\
  t = clock() - t;\
  double time = ((double)t)/CLOCKS_PER_SEC;\
  total_time += time;\
  printf("%d. [%s]: ", total_tests, name);\
  if(failed) {\
    ++failed_tests;\
    printf("FAILED\n");\
  } else {\
    printf("\e[1;32mPASSED\e[0m\n");\
  }\
} while(0)

#define t_assrt(bool_val) do {\
  if(!bool_val) {\
    return 1;\
  }\
} while(0)

#endif /* TEST_H */
