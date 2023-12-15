#include <stdio.h>

#include "la.h"

int main(void)
{
  Matrix *mat1 = mat_init(3, 3);

  mat_fill(&mat1, 4);
  mat_print(&mat1);

  putchar('\n');

  Matrix *mat2 = mat_init(3, 3);
  mat_fill(&mat2, 3);
  mat_print(&mat2);

  mat_add_mat(&mat1, &mat2);
  putchar('\n');

  mat_print(&mat1);

  mat_free(&mat1);
  mat_free(&mat2);

  return 0;
}
