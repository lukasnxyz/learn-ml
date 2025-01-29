from typing import Callable

def derivative(f: Callable[[float], float], x: float) -> float:
  """
  calculates the derivative of a function at x.

  parameters:
  f (function): the function to derivate
  x (float): x point at which to derivate the function

  returns:
  float: approximate value of the derivative at x
  """
  h = 1e-10
  return (f(x+h)-f(x))/h

def _linspace(start: float, stop: float, num: int) -> list[float]:
  if num < 2: return [float(start)]
  step = (stop-start)/(num-1)
  return [start+step*i for i in range(num)]

def integral(f: Callable, a: float, b: float, n: int=1000) -> float:
  """
  calculate the definite integral of a function using the trapezoidal rule.
    
  parameters:
  f (function): the function to integrate
  a (float): lower bound of integration
  b (float): upper bound of integration
  n (int): number of intervals for approximation (default: 1000)
  
  returns:
  float: approximate value of the definite integral
  """
  x = _linspace(a, b, n+1)
  y = [f(xi) for xi in x]
  intg = (y[0] + y[-1])/2.0 + sum(y[1:-1])
  return intg * (b-a)/n

if __name__ == "__main__":
  f = lambda x : x**3 + x**2 + 4*x
  print(derivative(f, 2))
  print(integral(f, 0, 4))
