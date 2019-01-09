/////////////////////////////////////////////////
//2017年3月30日
//小島聖司
//目的：動画の画像フレーム列から動作の特徴パターンを抽出
//手法：3DHOGに対して，濃度こう配法の考えを利用して，
//      ガウシアンフィルターにより各セルの各勾配方向を平滑化
//結果：KSCGRスコアが71.5%→73.5% (input = 11)
//関数：main()
/////////////////////////////////////////////////


#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include "KSCGRdatalist.hpp"
#include "sthogv.h"
#include <omp.h>

#define IMG_WIDTH 640
#define IMG_HEIGHT 480
#define BLOCK_SIZE 40
#define IMG_DEPTH 3
#define IMG_SPACE 3
#define SAMP_RATE 30
#define IMG_SCALE 1.
#define N_AZIMUTH_BINS 8
#define N_ZENITH_BINS 4
#define N_COLUMNS 16
#define N_ROWS 12
#define N_BLOCK (N_COLUMNS * N_ROWS)
#define SEQUENCE_TOP 0
#define SEQUENCE_REAR 34

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
  
  Datainfo data;
  int depth = atoi( argv[2] ); //
  int space = atoi( argv[3] );
  int nabins = N_AZIMUTH_BINS; // 方向角量子化
  int nzbins = N_ZENITH_BINS; // 仰角量子化
  int nbins = nabins * nzbins;
  int nblocks = N_COLUMNS * N_ROWS;
  int count_finished_sequences =0;
  int srate= SAMP_RATE;
  srate = atoi( argv[1] );

  cout << "---------時空間HOGV特徴----------" << endl;

  cout << "depth : " << depth << endl;
  cout << "space : " << space << endl;
  cout << "srate : " << srate << endl;
  cout << "----------------------------------" << endl;
	
  	for(int seq=0;seq<=34;seq++){
  	///////////////// danger ///////////////////
  	if(0 <= seq && seq <= 24) srate = atoi( argv[1] );
  	else srate = 2;
  	//////////////////////////////////////////////

    /* init in sequence */
    stringstream ss;
    ss << "sequence." << seq << ".csv";
    ofstream ofs( ss.str().c_str() );
   
    int end_frame = data.getFrames(seq)-(depth*space)/2;
    int start_frame = (int)(depth*space / 2);
    int *label = data.getGTArray(seq);

#pragma omp parallel for
    for(int f=start_frame;f<end_frame;f+=srate){
      for(int ch=0;ch<2;ch++){
       	Mat feat = get_3DHOG(seq,f,ch,depth,space);
      // /  cout << feat.size[1] << endl;
         for(int i=0;i<feat.size[1];i++){
       	   ofs << feat.at<double>(i) << ",";
       //		cout << i << "," <<  feat.at<float>(i) << endl;
        	}
        	feat.release();
      }
      /* output feature */
      ofs << label[f] << endl;
       printf("seqence%d %.0f%% finished\r",seq,(double)f*100/end_frame);
      fflush(stdout);
    //  printf("seqence%d %d / %d\r",seq,f,n_frames);
    }
    /* despose */


    cerr << "Sequence " << seq << " finished. (" << ++count_finished_sequences << "/" << 35 << ")" << endl;
    ofs.close();

    delete[] label;
  }


}
