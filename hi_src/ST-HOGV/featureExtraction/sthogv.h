#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include <omp.h>
#include <time.h>

#define IMG_WIDTH 640
#define IMG_HEIGHT 480
#define BLOCK_SIZE 40
#define IMG_DEPTH 3
#define IMG_SPACE 3
#define N_AZIMUTH_BINS 8
#define N_ZENITH_BINS 4
#define N_COLUMNS 16
#define N_ROWS 12
#define N_BLOCK (N_COLUMNS * N_ROWS)

using namespace cv;
using namespace std;

Mat get_slice(Mat &in3D){

  Mat res2D; 
  res2D.create( in3D.size[1] , in3D.size[0] ,CV_8UC1);

    for(int y=0;y<in3D.size[1];y++){
      for(int x=0;x<in3D.size[0];x++){
        res2D.at<uchar>(y,x) = in3D.at<uchar>(x,y,5); 
      }
    }

  imshow("get_slice" , res2D);

  waitKey(0);
  
  return res2D;
}

double *create_gauss ( int col, int row, double sigma )
{//col=5 row=5 sigma=3
  double num; 
  double *ret_value;
  
  ret_value = new double [col * row ];
  sigma *= sigma;
  num = 0;
  for( int i = 0; i < row; i++ )
    for( int j = 0; j < col; j++ )
      {
  int p = i * col + j;
  ret_value[p] = exp ( - pow((double)(j-col/2),2.0)/(2*sigma) );
  ret_value[p] *= exp ( - pow((double)(i-row/2),2.0)/(2*sigma) );
  ret_value[p] /= 2 * M_PI * sigma;
  num += ret_value[p];
      }
  for( int i = 0; i < row; i++ )
    for( int j = 0; j < col; j++ )
      {
  int p = i * col + j;
  ret_value[p] /= num;
      }   
   
  return ret_value;
}


Mat feature_gaussian_filter ( Mat& feature,
             int div_x,
             int div_y,
             int direction )
{
  static double *filt = NULL;
  
  int new_div_x, new_div_y;
  
  int filt_w = 5;
  int filt_h = 5;
  int skip = 1; // 書き換えた
  //int skip =2;

  double t = 0.295598;
  double sigma = - skip / ( 4 * log(t) );

  if( filt == NULL )
    {
      filt = create_gauss ( filt_w, filt_h, sigma ); 
    }
  new_div_x = (div_x-1)/skip + 1;
  new_div_y = (div_y-1)/skip + 1;

  Mat ret_feature = Mat::zeros(Size( new_div_y * new_div_x * direction ,  1), CV_64F);
  //ret_feature = new double[new_div_y * new_div_x * direction];

  for( int d = 0; d < direction ; d++ )
    for( int new_y = 0; new_y < new_div_y; new_y++ )
      for( int new_x = 0; new_x < new_div_x; new_x++ )
  {
    int new_p = (new_y*new_div_x+new_x) * direction + d;

    int old_x = new_x * skip;
    int old_y = new_y * skip;
    
    double num = 0;
    ret_feature.at<double>(new_p) = 0;
    for( int i = 0; i < filt_h; i++ )
      for( int j = 0; j < filt_w; j++ )
        {
    int p = (old_y + i)-(filt_h/2);
    int q = (old_x + j)-(filt_w/2);
    
    int old_p = (p * div_x + q)*direction + d;
    if ( p>=0 && p<div_y &&
         q>=0 && q<div_x )
      {
        ret_feature.at<double>(new_p) += feature.at<double>(old_p) * filt[i*filt_w+j];
        num += filt[i*filt_w+j];
      }
        }
    
    ret_feature.at<double>(new_p) /= num;
  }

  return ret_feature;
}



Mat get_feat(Mat &Img3D,int nabins, int nzbins)
{

  double ta = M_PI/nabins;
  double tz = M_PI/nzbins;

  Mat feat = Mat::zeros(Size( N_BLOCK*nabins*nzbins ,  1), CV_64F);

 // cout << feat.size() <<endl;
  
  int id_blc=0;

  //cout << Img3D.size[2] << endl;

int cnt=0;
  for(int x=0;x<Img3D.size[0];x+=BLOCK_SIZE){
    for(int y=0;y<Img3D.size[1];y+=BLOCK_SIZE){
      //  cout << "(x,y) =" << x << "," << y << endl;
          for(int zz=0;zz<Img3D.size[2];zz++){
            for(int yy=y;yy<y+BLOCK_SIZE;yy++){
            for(int xx=x;xx<x+BLOCK_SIZE;xx++){
            int id = id_blc*nabins*nzbins;
           // Img3D.at<uchar>(xx,yy,zz) = id_blc;
            if(xx<=0||xx>=Img3D.size[0]-1||yy<=0||yy>=Img3D.size[1]-1||zz<=0||zz>=Img3D.size[2]-1){
              continue;
            }
            // 3D gradient orientation //
            double dx,dy,dz,m,a,z;
            dx = (double)( Img3D.at<uchar>(xx-1,yy,zz)+Img3D.at<uchar>(xx+1,yy,zz)-2*Img3D.at<uchar>(xx,yy,zz) );
            dy = (double)( Img3D.at<uchar>(xx,yy-1,zz)+Img3D.at<uchar>(xx,yy+1,zz)-2*Img3D.at<uchar>(xx,yy,zz) );
            dz = (double)( Img3D.at<uchar>(xx,yy,zz-1)+Img3D.at<uchar>(xx,yy,zz+1)-2*Img3D.at<uchar>(xx,yy,zz) );
            // calec mag direct //
            m = sqrt( dx*dx + dy*dy + dz*dz );
            a = atan2( dy, dx );
            if(m>0) z = acos( dz/m );
            else z = acos( 0 );
            // calc gradient code //
            int dir1,dir2,dir;
            dir1 = ((int)((a+M_PI)/ta))%nabins;
            dir2 = ((int)(z/tz))%nzbins;
            dir = dir2*nabins + dir1;
         //   cout << dir << endl;
            feat.at<double>(id+dir) += m;
          }
        }
      }
      id_blc++;
    }
  }

  Img3D.release();

  Mat ret_feat = feature_gaussian_filter( feat,16,12,32 );

  return ret_feat;

}

//複数のフレームをボクセルデータに統合
Mat to3D(Mat *in2D,int depth){

  Mat res3D;
  const int size3D[] = { in2D[0].size[1] , in2D[0].size[0] , depth }; 
  res3D.create(sizeof(size3D) / sizeof(int), size3D, CV_8UC1);

  for(int z=0;z<res3D.size[2];z++){
    for(int y=0;y<res3D.size[1];y++){
      for(int x=0;x<res3D.size[0];x++){
        res3D.at<uchar>(x,y,z)  = in2D[z].at<uchar>(y,x);
      }
    }
  }
  return res3D;
}

Mat get_3DHOG(int seq,int center_frame,int ch,int depth,int space)
{


  Datainfo data;
  int nabins = N_AZIMUTH_BINS; // 方向角量子化
  int nzbins = N_ZENITH_BINS; // 仰角量子化
  int nbins = nabins * nzbins;
  int nblocks = N_COLUMNS * N_ROWS;

  if(ch == 0) depth = 5;
  //cout << depth << endl;

  // 入力フレームデータ
  Mat *imgs = new Mat[depth];
  for(int i=0,f=center_frame-(int)(depth*space/2);i<depth;i++,f+=space){
    if(ch == 0) imgs[i] = imread( data.getPath(seq,f,false), 0 );
    if(ch == 1) imgs[i] = imread( data.getPath(seq,f,true), 0 );
 //   cout << f << endl;
    if(imgs[i].cols < 1){
      cout << "フレーム画像を取得できていない" << endl;
    //  return -1;
    }
  }

  // 2Ds to 3D
  Mat Img3D = to3D(imgs,depth);

  for(int i=0;i<depth;i++){
     imgs[i].release();
  }

  Mat feat = get_feat(Img3D,nabins,nzbins);

  Img3D.release();


  return feat;
}
