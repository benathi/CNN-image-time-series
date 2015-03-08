'''
Created on Mar 7, 2015

@author: Ben Athiwaratkun (pa338)

fmincg code translated from Octave to Matlab

Minimize a continuous differentialble multivariate function. Starting point
is given by "X" (D by 1), and the function named in the string "f", must
return a function value and a vector of partial derivatives. The Polack-
Ribiere flavour of conjugate gradients is used to compute search directions,
and a line search using quadratic and cubic polynomial approximations and the
Wolfe-Powell stopping criteria is used together with the slope ratio method
for guessing initial step sizes. Additionally a bunch of checks are made to
make sure that exploration is taking place and that extrapolation will not
be unboundedly large. The "length" gives the length of the run: if it is
positive, it gives the maximum number of line searches, if negative its
absolute gives the maximum allowed number of function evaluations. You can
(optionally) give "length" a second component, which will indicate the
reduction in function value to be expected in the first line-search (defaults
to 1.0). The function returns when either its length is up, or if no further
progress can be made (ie, we are at a minimum, or so close that due to
numerical problems, we cannot get any closer). If the function terminates
within a few iterations, it could be an indication that the function value
and derivatives are not consistent (ie, there may be a bug in the
implementation of your "f" function). The function returns the found
solution "X", a vector of function values "fX" indicating the progress made
and "i" the number of iterations (line searches or function evaluations,
depending on the sign of "length") used.

Copyright (C) 2001 and 2002 by Carl Edward Rasmussen. Date 2002-02-13

(C) Copyright 1999, 2000 & 2001, Carl Edward Rasmussen

Permission is granted for anyone to copy, use, or modify these
programs and accompanying documents for purposes of research or
education, provided this copyright notice is retained, and note is
made of any changes that have been made.
 
These programs and documents are distributed without any warranty,
express or implied.  As the programs were written for research
purposes only, they have not been tested to the degree that would be
advisable in any important application.  All use of these programs is
entirely at the user's own risk.

Changes made:
translated from matlab to python
modify Thetas to be a list of matrices instead of 1d array


'''
import numpy as np
import math
import cmath
import sys
from scipy.optimize._tstutils import f2
from test.test_userdict import d2

def sToSflat(s):
    s_flat = []
    for i in range(np.shape(s)[0]):
        s_flat = np.concatenate( (s_flat,np.ndarray.flatten(s[i])) )
    return s_flat

def generalizedNorm(s):
    s_flat = sToSflat(s)
    return np.dot(s_flat.T, s_flat)

def generalizedDot(df2, s):
    s_flat = sToSflat(s)
    df2_flat = sToSflat(df2)
    d2 = np.dot(df2_flat.T,s_flat)
    return d2

def update(Thetas, z, s):
    for _j in range(np.shape(Thetas)[0]):
        Thetas[_j] += z*s[_j]

def fmincg(f, X, **args):
    Thetas = np.copy(X)
    print 'Shape of Thetas', np.shape(Thetas)
    length = 100
    if 'MaxIter' in args:
        length = args['MaxIter']
    RHO = 0.01
    SIG = 0.5
    INT = 0.1
    EXT = 3.0
    MAX = 20.0
    RATIO = 100.0
    
    red=1.0
    S=['Iteration']
    
    i = 0
    ls_failed = 0
    fX = []
    f1, df1 = f(Thetas)
    print f1, "f1"
    i += (length<0)
    s = -np.copy(df1)
    d1 = -generalizedNorm(s)
    #print d1, "d1"
    z1 = red/(1.0-d1)
    #print z1, "z1"
    while i < abs(length):
        i += (length > 0)
        
        Thetas0 = np.copy(Thetas)
        f0 = f1
        df0 = np.copy(df1)
        #print df1
        #print z1*s
        #update(Thetas, z1, s)
        for _j in range(np.shape(Thetas)[0]):
            Thetas[_j] += z1*s[_j]
        f2, df2 = f(Thetas)
        print 'Iteration %d. Cost=\t%f' % (i, f2)
        i += (length < 0)
        d2 = generalizedDot(df2,s)
        f3 = f1
        d3 = d1
        z3 = -z1
        if length > 0:
            M = MAX
        else:
            M = min(MAX, - length - i)
        success = 0
        limit = -1
        while 1:
            #print "check while loop"
            while  ( (f2> f1 + z1*RHO*d1)  or (d2 > -SIG*d1)) and (M>0):
                limit = z1
                if (f2 > f1):
                    z2 = z3 - (0.5 * d3*z3*z3)/(d3*z3 + f2 - f3) 
                else:
                    A = 6.0*(f2-f3)/z3 + 3.0*(d2+d3)
                    B = 3.0*(f3-f2) - z3*(d3 + 2.0*d2)
                    z2 = (cmath.sqrt(B*B - A * d2*z3*z3) - B)/A
                if isinstance(z2, complex) or math.isnan(z2) or math.isinf(z2):
                    z2 = z3/2.0
                #print "f2", f2
                #print f1*RHO* z1*d1, "f1*RHO*z1*D1"
                #print -SIG*d1, "-SIG * d1"
                #print d2, "d2"
                z2 = max(min(z2, INT*z3), (1.0-INT)*z3)
                z1 += z2
                for _j in range(np.shape(Thetas)[0]):
                    Thetas[_j] += z2*s[_j]
                #update(Thetas, z2, s)
                f2, df2 = f(Thetas)
                M -= 1
                i += (length<0)
                d2 = generalizedDot(df2, s)
                z3 -= z2
                #print M
            #print 'Could break out of loop'
            #print "f2 = %f. M = %f" % (f2, M)
            #print d2, "d2"
            #print M, "M"
            #print d2
            #print (SIG*d1)
            if (f2 > f1 +  z1*RHO*d1) or (d2 > -SIG*d1):
                #print 'first condition - not success'
                break
            elif d2 > SIG * d1:
                #print 'mark success'
                success = 1
                break
            elif M == 0:
                break
            #print 'Didnt break'
            A = 6.0*(f2-f3)/(1.0*z3)  + 3.0*(d2+d3)
            B = 3.0*(f3-f2) - z3*(d3 + 2.0*d2)
            #print 'B*B - ...', B*B - A*d2*z3*z3
            z2 = -d2*z3*z3 / ( B + cmath.sqrt( B*B - A*d2*z3*z3))
            if (isinstance(z2, complex) or math.isnan(z2)) or math.isinf(z2) or z2 < 0:
                if limit < -0.5:
                    z2 = z1 * (EXT-1.0)
                else:
                    z2 = (limit - z1)/2.0
            elif (limit > -0.5) and (z2+z1 > limit):
                z2 = (limit-z1)/2.0
            elif (limit < -0.5) and (z2 +z1 > z1 * EXT):
                z2 = z1*(EXT-1.0)
            elif z2 < -z3*INT:
                z2 = -z3*INT
            elif (limit > -0.5) & (z2 < (limit-z1)*(1.0-INT)):
                z2 = (limit-z1) * (1.0-INT)
            f3 = f2
            d3 = d2
            z3 = -z2
            z1 += z2
            for _j in range(np.shape(Thetas)[0]):
                Thetas[_j] += z2*s[_j]
            #update(Thetas, z2, s)
            f2, df2 = f(Thetas)
            M -= 1
            i += (length<0)
            d2 = generalizedDot(df2, s)
            
            #print i 
        #print 'Outside while 1'
        if success:
            #print 'success'
            f1 = f2
            #fX = [fX.T, f1].T
            fX.append(f1)
            #s = (np.dot(df2.T , df2) - np.dot(df1.T,df2))/ (np.dot(np.dot(df1.T,df1),s) ) - df2
            _mult = ( generalizedNorm(df2) - generalizedDot(df1, df2) )/generalizedNorm(df1)
            for _j in range(np.shape(s)[0]):
                s[_j] *= _mult
                s[_j] -= df2[_j]
            
            tmp = np.copy(df1)
            df1 = np.copy(df2)
            df2 = np.copy(tmp)
            d2 = generalizedDot(df1, s)
            if d2 > 0:
                s = -np.copy(df1)
                d2 = - generalizedNorm(s)
            z1 = z1 * min(RATIO, d1/(d2-sys.float_info.min))
            d1 = d2
            ls_failed = 0
        else:
            print 'Not Success'
            Thetas = np.copy(Thetas0)
            f1 = f0
            df1 = np.copy(df0)
            if ls_failed or (i > abs(length)):
                print 'Line Search Failed'
                break
            tmp = np.copy(df1)
            df1 = np.copy(df2)
            df2 = np.copy(tmp)
            s = -np.copy(df1)
            d1 = -generalizedNorm(s)
            z1 = 1/(1-d1)
            ls_failed = 1
    
    print 'fmincg Return'
    #print Thetas
    return (Thetas, fX)


def main():
    pass
    
if __name__ == "__main__":
    main()