'''
Created on Mar 7, 2015

@author: Ben Athiwaratkun (pa338)

fmincg code translated from Octave to Matlab

'''
import numpy as np

def fmincg(f, Thetas, **args):
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
    i += (length<0)
    s = -df1
    d1 = -np.dot(s.T, s)
    z1 = red/(1.0-d1)
    while i < abs(length):
        i += (length > 0)

def main():
    pass
    
if __name__ == "__main__":
    main()