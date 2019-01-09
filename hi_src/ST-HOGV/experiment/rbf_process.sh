#!/bin/sh

path='/home/seiji/jikken/3DHOG/3DGF/'
path_libsvm='/home/seiji/dev/libsvm-3.21/'
path_liblin='/home/seiji/dev/liblinear-1.94/'


    ${path_libsvm}/svm-train -c 32.0 -g 0.0078125 -s 0 -t 2 train.scale train_svm.txt.model
#     rm -f train_svm.txt 

    # detection
    rm -f out.txt truth.csv
    for i in `seq 25 34`
    do
	echo 'detecting sequence '${i}
	 # ${path}/others/csv2svm.py sequence.${i}.csv > ${i}_svm.txt
	 # ${path_libsvm}/svm-scale -r data.minmax ${i}_svm.txt > ${i}.scale
	${path_libsvm}/svm-predict ${i}.scale train_svm.txt.model detected.txt #識別
	${path}/others/voting.py detected.txt 30 >> out.txt #投票処理
	${path}/others/groundtruth.py sequence.${i}.csv >> truth.csv #正解データ作成
	rm -f ${i}_svm.txt detected.txt
    done
    cat sequence.2[5-9].csv sequence.3[0-4].csv > test.csv
    # rm -f sequence.2[5-9].csv sequence.3[0-4].csv

    # evaluate
    echo 'Result'
    ${path}/others/KSCGR_evaluate.py out.txt truth.csv > res${j}.txt