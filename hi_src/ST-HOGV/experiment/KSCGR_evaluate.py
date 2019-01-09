#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, getopt, os, numpy as np


#
# get Groundtruth Path
#
def getPath(n_seq):
    path = '/home/KSCGR/'
    actor = n_seq / 5 + 1
    menu = n_seq % 5
    if actor <= 5:
        if menu==0:
            path = path + "boild-egg-"
        elif menu==1:
            path = path + "ham-egg-"
        elif menu==2:
            path = path + "kinshi-egg-"
        elif menu==3:
            path = path + "omelette-"
        else:
            path = path + "scramble-egg-"
        path = path + str(actor)
    elif actor <= 7:
        if actor == 6:
            path = path + "test_data_10_0" + str(menu+1)
        elif actor == 7:
            path = path + "test_data_11_0" + str(menu+1)
    else:
        print "error"
        quit();
    path = path + "/labels.txt"
    return path


# 
# get List of Groundtruth
# 
def LabelList(path):
    list_out = []
    cnt = 0
    if path.split('.')[-1] == 'csv':
        for elems in open(path):
            if elems[0] == '#':
                continue
            l = int( elems[-8:].split(',')[-1] )
            if l < 0:
                l = 0
            list_out.append(l)
    else:
        for elems in open(path):
            l = int( elems[:8].split(' ')[0] )
            if l < 0:
                l = 0
            list_out.append(l)
    return list_out


# 
# show detail of result
# 
def show_detail(tp, fp, fn, p, r, f):
    print "================================================"
    for i in range(0,9):
        if i > 0:
            print '### Label %d' % i
        else:
            print '### Label -1000'
        print 'TP =\t%d,\t' % tp[i] ,
        print 'FP =\t%d,\t' % fp[i] ,
        print 'FN =\t%d' % fn[i]
        print 'Precision\t%f' % p[i]
        print 'Recall   \t%f' % r[i]
        print 'F-measure\t%f' % f[i]
        print "================================================"


# 
# output result as CSV format file
# 
def out_csv(filename, precision, recall, fmeasure, rate, score):
    ostr =  '"Label","Precision","Recall","F-measure"\n'
    for i in range(0,9):
        if i > 0:
            ostr += '%d,%f,%f,%f\n' % (i, precision[i], recall[i], fmeasure[i])
        else:
            ostr += "-1000,%f,%f,%f\n" % (precision[i], recall[i], fmeasure[i])
    ostr += '#Rate,%f\n' % rate
    ostr += '#Score,%f\n' % score
    fout = open(filename, 'w')
    fout.write(ostr)
    fout.close()


# 
# output result as GNUplot data format file
# 
def out_gnuplot(filename, precision, recall, fmeasure):
    fout = open(filename,'w')
    fout.writelines('# Precision Recall F-measure\n')
    for i in range(0,9):
        fout.write( '%d %f %f %f\n'%(i,precision[i],recall[i],fmeasure[i]) )
    fout.close()

def plot_graph(filename, precision, recall, fmeasure):
    
    script_path = '.gnuplot_script.dat'
    data_path = '.gnuplot_data.dat'

    gnu_script = '#script\n'
    gnu_script += 'set xrange [-1:9]\n'
    gnu_script += 'set yrange [0:1]\n'
    gnu_script += 'set key left top\n'
    gnu_script += 'set xlabel "Motion Label(#)"\n'
    gnu_script += 'set boxwidth 0.25\n'
    gnu_script += 'set xtics ("-1000" 0, "1" 1, "2" 2, "3" 3, "4" 4, "5" 5, "6" 6, "7" 7, "8" 8)\n'
    gnu_script += 'plot "{0}" using ($0-0.25):2 title "Precision" with boxes fs pattern 1, "{0}" using ($0):3 title "Recall" with boxes fs pattern 2, "{0}" using ($0+0.25):4 title "F measure" with boxes fs pattern 3\n'.format(data_path)
    gnu_script += 'set term postscript enhanced 22\n'
    gnu_script += 'set output "{0}"\n'.format(filename)
    gnu_script += 'replot\n'
    f = open(script_path,'w')
    f.write(gnu_script)
    f.close()
    
    out_gnuplot(data_path, precision, recall, fmeasure)
    cmd = 'gnuplot {0}'.format(script_path)
    os.system(cmd)
    # cmd = 'rm -f {0} {1}'.format(script_path,data_path)
    # os.system(cmd)

#
# plot confusion matrix
#
def plot_confusion_matrix(filename, truth, detected):
    fout = open(filename,'w')
    mat = [[0 for j in range(9)] for j in range(9)]
    max_index = len(truth)
    if max_index > len(detected):
        max_index = len(detected)
    for i in range(0,max_index):
        mat[truth[i]][detected[i]] += 1
    # matrix 1 ; number of samples
    fout.write( '#m1\n' )
    for arr in mat:
        for elem in arr[:-1]:
            fout.write( (str(elem)[:7]+'      ')[:7]+',' )
        fout.write( str(arr[-1])[:7] + '\n' )
    # matrix 2 ; rate of detected labels
    fout.write( '#m2\n' )
    for arr in mat:
        total = float(sum(arr))
        for elem in arr[:-1]:
            fout.write( (str(elem/total)[:7]+'      ')[:7]+',' )
        fout.write( str(arr[-1]/total)[:7] + '\n' )

    fout.close()

# 
# show usage
# 
def usage():
    print 'Usage: %s {opt} [detected result] [ground truth(svm)]' % sys.argv[0].split('/')[-1]
    print 'Options:'
    print '-h' + '\t ; show help'
    print '-t [seq num]' + '\t ; first sequence num (default=25)'
    print '-r [seq num]' + '\t ; last sequence num (default=34)'
    print '-c [filename]' + '\t ; output result as csv format'
    print '-g [filename]' + '\t ; output result as gnuplot data format'
    print '-m [filename]' + '\t ; output confusion matrix'


# 
# main
# 
def main():

    # check args
    try:
        opts, args = getopt.getopt( sys.argv[1:], 'hdt:r:c:g:m:' )
        if len(args) < 2:
            usage()
            quit()
    except getopt.GetoptError, err:
        print str(err)
        usage()
        quit()

    # check opts
    f_detail = True
    n_csv = None
    n_gnp = None
    n_mat = "n_mat.csv"
    seq_t = 25
    seq_r = 34
    for o, a in opts:
        if o == '-h':
            usage()
            quit()
        elif o == '-d':
            f_detail = True
        elif o == '-t':
            seq_t = int(a)
        elif o == '-r':
            seq_r = int(a)
        elif o == '-c':
            n_csv = a
        elif o == '-g':
            n_gnp = a
        elif o == '-m':
            n_mat = a
        else:
            assert False, "unhandled option"

    # read input file
    detected = []
    for elems in open(args[0]):
        elem = elems[:-1].split(' ')
        l = int(elem[-1])
        if l < 0:
            l = 0
        detected.append(int(l))

    # read ground truth
    truth = LabelList( args[1] )
    # truth = []
    # for n_seq in range(seq_t,seq_r+1):
    #     truth = truth + LabelList( args[1] )
    


    # error check
    if len(detected) > len(truth):
        print 'bad groundtruth size'
        quit()

    # compare list
    tp = np.zeros(9)
    fp = np.zeros(9)
    fn = np.zeros(9)
    for i in range(0,len(detected)):
        if detected[i] == truth[i]:
            tp[ detected[i] ] += 1.
        else:
            fp[ detected[i] ] += 1.
            fn[ truth[i] ] += 1.


    # evaluate
    p = (tp / (tp+fp))
    p[ np.where(p != p)[0] ] = 0
    r = tp / (tp+fn)
    r[ np.where(r != r)[0] ] = 0
    f = ((p*r*2) / (p+r))
    f[ np.where(f != f)[0] ] = 0
    rate = np.sum(tp) / (np.sum(tp)+np.sum(fn))
    score = np.mean(f[1:])


    # show result
    if f_detail == True:
        show_detail(tp,fp,fn,p,r,f)
    print 'Rate\t%f' % rate
    print 'Score\t%f' % score

    if n_csv != None:
        out_csv( n_csv, p, r, f, rate, score )
    if n_gnp != None:
        plot_graph( n_gnp, p, r, f )
    if n_mat != None:
        plot_confusion_matrix( n_mat, truth, detected )


if __name__ == "__main__":
    main()
