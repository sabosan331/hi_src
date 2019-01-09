#!/usr/bin/env python

import sys
argv = sys.argv
argc = len(argv)

# error check
if (argc < 2):
    print 'Usage: %s [input filename]' % sys.argv[0]
    quit()

# convert to svm format
for line in open(argv[1],'r'):
    if line[0] != '#':
        items = line[:-1].split(',')
        print items.pop(),
        count = 1
        for item in items:
            print str(count) + ':' + item ,
            count += 1
        print ''
