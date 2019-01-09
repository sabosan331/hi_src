#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

def main():

    if len(sys.argv) < 2 :
        print 'USAGE: %s [input file]' % sys.argv[0].split('/')[-1]
        quit()

    filename = sys.argv[1]
    filetype = filename.split('.')[-1]
    i0 = 0
    i1 = 8
    s_char = ' '
    l_idx = 0
    if filetype == 'csv':
        i0 = -9
        i1 = -1
        s_char = ','
        l_idx = -1


    inputfile = open(filename,'r')
    line = inputfile.readline()
    while line[0] == '#':
        line = inputfile.readline()
    arr = [[ int(line[i0:i1].split(s_char)[l_idx]), 1 ]]


    line = inputfile.readline()
    while line:
        while line[0] == '#':
            line = inputfile.readline()
        elem = int( line[i0:i1].split(s_char)[l_idx] )
        f_notfound = True
        for i in range(0,len(arr)):
            if arr[i][0] == elem:
                arr[i][1] += 1
                f_notfound = False
                break
        if f_notfound:
            arr.append( [elem,1] )
        line = inputfile.readline()

    nr_max = arr[0][1]
    nr_min = arr[0][1]
    for a in arr[1:]:
        if nr_max < a[1]:
            nr_max = float(a[1])
        if nr_min > a[1]:
            nr_min = float(a[1])

    for a in arr:
        print '-w%s' % a[0],
        print '%f' % ( nr_min/a[1] ),


if __name__ == '__main__':
    main()
