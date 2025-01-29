# wahrscheinlichkeitstheorie
import math

# binomial verteilung
def bionomial_v(n: int, k: int, p: float):
  a = math.factorial(n)/(math.factorial(k)*math.factorial(n-k))
  return a*(p**k)*((1-p)**(n-k))

# geometrische verteilung
def geom_v(k: int, p: float):
  return ((1-p)**(k-1))*p

# poisson verteilung
def poisson_v(k: int, la: float):
  return ((la**k)*math.e**(-la))/math.factorial(k)

# cumulative distribution function 
def cdf(x: int, pk): # pk: lambda func
  ret = 0
  for i in range(x): 
    ret += pk(i)
  return ret

# normal verteilung
