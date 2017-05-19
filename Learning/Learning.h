#include <opencv.hpp>
#include <string>
#include <iostream>
#include <vector>
#include <malloc.h>
#include <map>

#define POSNUM 2429//正训练集大小
#define NEGNUM 559
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
//样本的结构
struct sample {
	vector <int> img;//	图像
	bool polar;//正例=true，反例=false
	double weight;//样本权重
	sample(vector <int> im, bool pon, double weig) :img(im), polar(pon), weight(weig) {}
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

	//正例的haar特征向量和lab特征向量
	vector < vector <int>> pos_block_sum_data;
	vector < vector <int>> pos_total_lab_data;
	//反例的haar特征向量和lab特征向量
	vector < vector <int>> neg_block_sum_data;
	vector < vector <int>> neg_total_lab_data;
	//样本集
	vector <sample> samp;

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
	//设置参数
	void setParam(int confMinThres = 0, int weightMinThres = 2, int Xshift = 2, int Yshift = 2, int endScaleNum = 24, int startScaleNum = 4);
	//初始化内存（内存管理c++可以删掉？）
	//void initMemory(int wd, int ht);
	//释放内存
	void releaseMemory();
	void releaseAll();

	//获得6通道的haar特征图像
	void GetBlockSumHaarData_All(vector < vector <int>> &img, vector < vector <int>> &block_sum_data);
	//获得6通道的LAB特征图像
	void GetLABData_All(vector < vector <int>> &block_sum_data, vector < vector <int>> &total_lab_data);
	//学习
	void AdaBoost();
	void initSample();
};