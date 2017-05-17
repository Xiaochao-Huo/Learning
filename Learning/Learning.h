#include <opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <malloc.h>
#include <map>

#define POSNUM 2429//��ѵ������С
#define NEGNUM 32
#define _CRT_SECURE_NO_WARNINGS

//The scale number for double rate difference
#define level_num_one_region 3
//Block number used for 16*16 training
#define LAB_block_size_num 2
//Total LAB feature size for each pixel
#define LAB_buf_num  LAB_block_size_num*3

#define LAB_feat_type 6
#define total_LAB_num (face_rect_ht*face_rect_wd*LAB_feat_type)
//Maximum scale number for face detection
#define MAX_IMG_SCALE_NUM 24
//The rate difference between two adjacent scales
#define total_delta_rate_one_level 1.25992
//Minumum face detection width and height
#define wd 20	
#define ht 20

using namespace std;
using namespace cv;

struct pyramidSample {
	int down_sample_rate_2;//double downsampling rate for this scale
	int down_sample_move_2;//double downsample move value for this scale
	float deflate_rate;//downsampling rate for this scale after double sampling
	float total_deflate_rate;//total downsampling rate for this scale
};
struct weakModel {

};
struct Model{
	int step;
	int thres;
	weakModel *pModel;
};
class Learning {
public:
	Learning();
	~Learning();

	Model *detectModel;

	//const int total_multi_data;
	int total_multi_data;

	int image_wd;
	int image_ht;
	int face_detect_end_search_scale_num;
	int face_detect_start_search_scale_num;
	int nDetectLevel_First;
	int Candidate_Combine_Weight_MinThres;
	int Candidate_Combine_Conf_MinThres;
	int Detect_Search_XShift;
	int Detect_Search_YShift;

	//������haar����������lab��������
	vector < vector <int>> pos_block_sum_data;
	vector < vector <int>> pos_total_lab_data;
	//������haar����������lab��������
	vector < vector <int>> neg_block_sum_data;
	vector < vector <int>> neg_total_lab_data;

	//the buffer definition for block sum value calsulation
	unsigned short *haar_data_x, *haar_data_y;
	unsigned short *haar_sum_x, *haar_sum_y;
	//the precalculated buffer array model for optimization
	int **pOriLABCenFeatNum;
	int **pLABFeat_Array;
	int ***pOriLABFeat_Array;

	int *pFileBuf;

	//the downsampling scale and rate struct
	pyramidSample *pSampleRate;

	void init();
	//���ò���
	void setParam(int confMinThres = 0, int weightMinThres = 2, int Xshift = 2, int Yshift = 2, int endScaleNum = 24, int startScaleNum = 4);
	//��ʼ���ڴ棨�ڴ����c++����ɾ������
	//void initMemory(int wd, int ht);
	//�ͷ��ڴ�
	void releaseMemory();
	void releaseAll();

	//���6ͨ����haar����ͼ��
	void GetBlockSumHaarData_All(vector < vector <int>> &img, vector < vector <int>> &block_sum_data);
	//���6ͨ����LAB����ͼ��
	void GetLABData_All(vector < vector <int>> &block_sum_data, vector < vector <int>> &total_lab_data);
	//ѧϰ
	void AdaBoost();
};