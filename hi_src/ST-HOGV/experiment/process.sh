#!/bin/sh

path='/net/xserve0/users/kojima/research/ST-HOGV/'
path_libsvm='/net/xserve0/users/kojima/dev/libsvm-3.21/'
path_liblin='/net/xserve0/users/kojima/dev/liblinear-1.94/'

echo "(0,1) scaling"
for j in 11
do
for k in 1
do
    # echo "skip_rate '${j}' "
    echo 'sequence '${i}' is proceeding'
    ${path}/featureExtraction/build/sthogv 30 ${j} ${k}

    # concat training samples
    cat sequence.[0-9].csv sequence.1[0-9].csv sequence.2[0-4].csv > train.csv
    rm -f sequence.[0-9].csv sequence.1[0-9].csv sequence.2[0-4].csv
    ${path}/experiment/csv2svm.py train.csv > train_svm.txt

    # scaling
    # echo "scaling"
    ${path_libsvm}/svm-scale -l 0 -u 1 -s data.minmax train_svm.txt > train.scale

    # training with liblinear
    w=`${path}/experiment/svm_weight.py train.scale` 
    ${path_liblin}/train -s 6 ${w} train.scale train_svm.txt.model
    rm -f train_svm.txt 

    # detection
    rm -f out.txt truth.csv
    for i in `seq 25 34`
    do
	echo 'detecting sequence '${i}
	${path}/experiment/csv2svm.py sequence.${i}.csv > ${i}_svm.txt
	${path_libsvm}/svm-scale -l 0 -u 1 -r data.minmax ${i}_svm.txt > ${i}.scale
	${path_liblin}/predict ${i}.scale train_svm.txt.model detected.txt #識別
	${path}/experiment/voting.py detected.txt 30 >> out.txt #投票処理
	${path}/experiment/groundtruth.py sequence.${i}.csv >> truth.csv #正解データ作成
	rm -f ${i}_svm.txt detected.txt
    done
    cat sequence.2[5-9].csv sequence.3[0-4].csv > test.csv
    rm -f sequence.2[5-9].csv sequence.3[0-4].csv

    # evaluate
    echo 'Result'
    ${path}/experiment/KSCGR_evaluate.py out.txt truth.csv > res${j}_${k}.txt
done
done