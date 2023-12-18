#include <stdio.h>

#include "src/la.h"

int sigmoid(const int);
int sigmoid_deriv(const int);

int main(void)
{
  Matrix *data = mat_init(4, 3);

  data->data[0][0] = 0;
  data->data[0][1] = 0;
  data->data[0][2] = 0;
  data->data[1][0] = 1;
  data->data[1][1] = 0;
  data->data[1][2] = 1;
  data->data[2][0] = 0;
  data->data[2][1] = 1;
  data->data[2][2] = 1;
  data->data[3][0] = 1;
  data->data[3][1] = 1;
  data->data[3][2] = 0;

  mat_print(&data);

  mat_free(&data);

  return 0;
}
