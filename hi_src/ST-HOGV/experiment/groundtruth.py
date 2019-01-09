#!/usr/bin/env python

import sys, getopt
argv = sys.argv
argc = len(argv)

# error check
if argc < 2:
    print 'Usage: %s [input filename]' % argv[0].split('/')[-1]
    quit()

# extract last element
seq = -1
n_frames = -1
n_samples = -1
a = -1
cnt = 0
for line in open(argv[1],'r'):
    if line[0] != '#':
        items = line[:-1].split(',')
        if a < 0:
            print items[-1]
        else:
            print str(seq)+','+str(cnt*a)+','+items[-1]
            cnt = cnt + 1
    else:
        words = line[1:-1].split(' ')
        if words[0] == 'sequence':
            seq = int(words[1])
            cnt = 0
        elif words[0] == 'n':
            if words[1].split('\t')[0] == 'frames':
                n_frames = int(words[-1])
            elif words[1].split('\t')[0] == 'samples':
                n_samples = int(line[:-1].split('\t')[-1])
            a = n_frames / n_samples
        else:
            a = -1
            cnt = 0
            
        
