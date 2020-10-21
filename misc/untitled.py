import numpy as np
from sympy import *
from sympy import pprint
from sympy.polys.ring_series import rs_series
from sympy import collect, Mul, Order, S
import sys
sys.displayhook = pprint

x = Symbol('x')

#~ A = Symbol('A')
#~ B = Symbol('B')
#~ C = Symbol('C')

#~ Q = Symbol('Q')
q1 = Symbol('q1')
q2 = Symbol('q2')

f = Symbol('f')
k = Symbol('k')
u1 = Symbol('u1')
u2 = Symbol('u2')
un = Symbol('un')

P6 = Symbol('P6')
P5 = Symbol('P5')
P4 = Symbol('P4')
P3 = Symbol('P3')
P2 = Symbol('P2')
P1 = Symbol('P1')
P0 = Symbol('P0')

Psymb = [P0,P1,P2,P3,P4,P5,P6]

F2 = Rational(5,7) +   Rational(1,2)*x * (q1/q2 + q2/q1) + Rational(2,7) * x**2 
G2 = Rational(3,7) +  Rational(1,2)*x * (q1/q2 + q2/q1) + Rational(4,7) * x**2 
#~ macLN = (2*r + Rational(2,3)*r**3 + Rational(2,5)*r**5 + Rational(2,7)*r**7 + + Rational(2,9)*r**9)

#~ S2 = x**2 - Rational(1,3)
#~ S2 = (u*r - r**2)**2 / (r**2 * (r**2 + 1 - 2*r*u)) - Rational(1,3)
#~ D2 = Rational(2,7) * (S2 - Rational(2,3))


#~ F3 = 12/r**4 - 158/r**2 + 100 - 42*r**2 + 3/r**5 * (7*r**2 + 2) * (r**2 -1)**3 * Q
#~ G3 = 12/r**4 - 82/r**2 + 4 - 6*r**2 + 3/r**5 * (r**2 + 2) * (r**2 -1)**3 * Q

#~ A = Rational(5,128)*(1+r**2)*(-3+14*r**2 - 3*r**4) / r**4
#~ B = (r-1)**4 * (r+1)**4
#~ C = (r**2-1)**4

 
def legPoly(n):
	if n == 0:
		P = 1
	elif n == 1:
		P = x 
	elif n == 2:
		P = Rational(3,2) * x**2 - Rational(1,2)
	elif n == 3:
		P = Rational(5,2) * x**3 - Rational(3,2) * x
	elif n == 4:
		P = Rational(35,8) * x**4 - Rational(15,4) * x**2 + Rational(3,8)
	elif n == 5:
		P = Rational(63,8) * x**5 - Rational(35,4) * x**3 + Rational(15,8) * x
	elif n == 6:
		P = Rational(231,16) * x**6 - Rational(315,16) * x**4 + Rational(105,16) * x**2 - Rational(5,16)

	return P
	


def PolExp(f,p):
	if p == 1:
		q1, r1 = div(f, legPoly(p), domain='QQ')  
		return q1 * P1 + r1 * P0
	elif p == 2:
		q2, r2 = div(f, legPoly(p), domain='QQ') 
		q1, r1 = div(r2, legPoly(p-1), domain='QQ') 
		return q2 * P2 + q1 * P1 + r1 * P0
	elif p == 3:
		q3, r3 = div(f, legPoly(p), domain='QQ') 
		q2, r2 = div(r3, legPoly(p-1), domain='QQ') 
		q1, r1 = div(r2, legPoly(p-2), domain='QQ') 
		return q3 * P3 + q2 * P2 + q1 * P1 + r1 * P0
	elif p == 4:
		q4, r4 = div(f, legPoly(p), domain='QQ') 
		q3, r3 = div(r4, legPoly(p-1), domain='QQ') 
		q2, r2 = div(r3, legPoly(p-2), domain='QQ') 
		q1, r1 = div(r2, legPoly(p-3), domain='QQ') 
		return q4 * P4 + q3 * P3 + q2 * P2 + q1 * P1 + r1 * P0
	elif p == 5:
		q5, r5 = div(f, legPoly(p), domain='QQ') 
		q4, r4 = div(r5, legPoly(p-1), domain='QQ') 
		q3, r3 = div(r4, legPoly(p-2), domain='QQ') 
		q2, r2 = div(r3, legPoly(p-3), domain='QQ') 
		q1, r1 = div(r2, legPoly(p-4), domain='QQ') 
		return q5 * P5 + q4 * P4 + q3 * P3 + q2 * P2 + q1 * P1 + r1 * P0
	elif p == 6:
		q6, r6 = div(f, legPoly(p), domain='QQ') 
		q5, r5 = div(r6, legPoly(p-1), domain='QQ') 
		q4, r4 = div(r5, legPoly(p-2), domain='QQ') 
		q3, r3 = div(r4, legPoly(p-3), domain='QQ') 
		q2, r2 = div(r3, legPoly(p-4), domain='QQ') 
		q1, r1 = div(r2, legPoly(p-5), domain='QQ') 
		return q6 * P6 + q5 * P5 + q4 * P4 + q3 * P3 + q2 * P2 + q1 * P1 + r1 * P0
	
	


Fsquare = F2 * F2
Gsquare = G2 * G2
Cross = F2 * G2

coeff1 = -Rational(3,2)*u1*u2
coeff2 = (-4*u1*un*x)*F2

#~ print factor(3/r**5 * (7*r**2 + 2) * (r**2 -1)**3 * Q + 9/r**5 * (r**2 + 2) * (r**2 -1)**3 * Q)

#~ print PolExp(Fsquare, 4)
#~ print PolExp(Gsquare, 4)
#~ print expand(Cross)
print PolExp(Cross, 4)




