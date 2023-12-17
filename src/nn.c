#include <stdio.h>

#include "la.h"

int main(void)
{
  printf("Hello World!\n");

  Matrix *m1 = mat_init(4, 3);
  Matrix *m2 = mat_init(4, 3);

  mat_fill(&m1, 2);
  mat_fill(&m2, 3);
  mat_add_mat(&m1, &m2);

  mat_print(&m1);

  mat_free(&m1);
  mat_free(&m2);

  return 0;
}
