#!/usr/bin/env python

import sys



def max_index(x):
    return max(xrange(len(x)), key=lambda i: x[i])



def voting(array, l, out):

    length = len(array)

    for i in range(0,l):
        out.append( array[i] )

    for i in range(l,length-l):
        v = [0]*9
        for d in range(-l,l+1):
            v[ array[i+d] ] += 1
        out.append( max_index(v) )

    for i in range(length-l,length):
        out.append( array[i] )



def main(filename, vnum):

    array = []
    dst = []
    length = vnum/2
    # read
    for line in open(filename,'r'):
        elems = line[:-1].split(',')
        n = int(elems[-1])
        if n < 0:
            array.append( 0 )
        else:
            array.append( n )

    # voting
    voting(array, length, dst)

    # out
    for l in dst:
        if l == 0:
            sys.stdout.write('-1000\n')
        else:
            sys.stdout.write(str(l)+'\n')



if __name__ == "__main__":

    if len(sys.argv) < 2:
        print 'Usage:',sys.argv[0].split('/')[-1],'[labels] [#of vote]'
        exit()

    main(sys.argv[1], int(sys.argv[2]))
