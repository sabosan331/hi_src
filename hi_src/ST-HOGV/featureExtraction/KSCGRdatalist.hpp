#include <stdio.h>
#include <stdlib.h>
#include <fstream>

#define KSCGR_DATASET_BASE_PATH "/home//seiji/Data/KSCGR"

class Datainfo
{
private:
  char base[64];
  char menu[35][16];
  int n_frames[35];
  int n_sequences;
  char str[128];
public:
  Datainfo()
  {
    // base
    sprintf(base, KSCGR_DATASET_BASE_PATH );
    // train data
    int i=0;
    for(int id=1;id<=5;id++){
      sprintf(menu[i++],"boild-egg-%d",id);
      sprintf(menu[i++],"ham-egg-%d",id);
      sprintf(menu[i++],"kinshi-egg-%d",id);
      sprintf(menu[i++],"omelette-%d",id);
      sprintf(menu[i++],"scramble-egg-%d",id);
    }
    // test data
    for(int id=10;id<=11;id++){
      for(int cnt=1;cnt<=5;cnt++)sprintf(menu[i++],"test_data_%d_0%d",id,cnt);
    }
    // n_frames
    int f[35] = { 7927,  4983, 10390, 11828,  9958,
		  6734,  6642,  6903,  4907,  8048,
		  4266,  4288,  5381,  5021,  2257,
		  6625, 11822,  9563,  6461,  5192,
		  6455,  7704,  6592,  5537,  6609,
		  5509,  5217,  4341,  6493,  5162,
		  6007,  6093,  5333,  5113,  6509  };
    for(int cnt=0;cnt<35;cnt++) n_frames[cnt] = f[cnt];
    // path
    str[0] = '\0';
    n_sequences = 35;
  }
  char *getLabelsPath(int seq)
  {
    sprintf(str,"%s/%s/labels.txt",base,menu[seq]);
    return str;
  }
  char *getPath(int seq, int frame_num, bool isColor)
  {
    if(isColor == true)
      sprintf(str,"%s/%s/image_jpg/%d.jpg",base,menu[seq],frame_num);
       else
      sprintf(str,"%s/%s/depth_8bit/%d.png",base,menu[seq],frame_num);
    return str;
  }
  char *getDepth(int seq, int frame_num)
  {
    sprintf(str,"%s/%s/depth_8bit/%d.png",base,menu[seq],frame_num);
    return str;
  }
  char *getImage(int seq, int frame_num)
  {
    sprintf(str,"%s/%s/image_jpg/%d.jpg",base,menu[seq],frame_num);
    return str;
  }
  int getFrames(int seq)
  {
    if(seq<n_sequences) return n_frames[seq];
    else return 0;
  }
  int getGTLength(int seq)
  {
    return ( this->getFrames(seq) - 1 );
  }
  int *getGTArray(int seq)
  {
    std::ifstream ifs( this->getLabelsPath(seq) );
    int length = this->getGTLength(seq);
    int *arr = new int[ length ];
    char str[16];
    for(int i=0;i<length;i++){
      ifs.getline(str,16,9); ifs.getline(str,16);
      arr[i] = atoi(str);
    }
    return arr;
  }
  int getMajorLabel(int *label, int offset, int length)
  {
    int *l = label + offset;
    int cnt[9] = {0};
    for(int i=0;i<length;i++){
      if( *l>0 && *l<9 ) cnt[*l]++;
      else cnt[0]++;
      l++;
    }
    int max_ind = -1, max_cnt = -1;
    for(int i=0;i<9;i++){
      if( max_cnt < cnt[i] ){ max_cnt=cnt[i]; max_ind = i; }
    }
    if( max_ind>0 && max_ind<9 ) return max_ind;
    else return -1000;
  }
  char *getLabelName(int label)
  {
    if( label==1 ) sprintf(str,"Breaking");
    else if( label==2 ) sprintf(str,"Mixing");
    else if( label==3 ) sprintf(str,"Baking");
    else if( label==4 ) sprintf(str,"Turning");
    else if( label==5 ) sprintf(str,"Cutting");
    else if( label==6 ) sprintf(str,"Boiling");
    else if( label==7 ) sprintf(str,"Seasoning");
    else if( label==8 ) sprintf(str,"Peeling");
    else sprintf(str,"None");
    return str;
  }
};
