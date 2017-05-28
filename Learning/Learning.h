//#include <opencv.hpp>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <malloc.h>
#include <map>
#include <algorithm>
#include <cmath>
#include <iomanip>


#define POSNUM 2429//��ѵ������С
#define NEGNUM 559

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
//ÿ��ǿ�������������ʶ�ʺ���ͼ����
#define maxErrorDetectRate 0.5
#define detectRate 0.99
//���ղ������������ʶ��
#define errorDetectRate 0.001

#define adjustRate 0.05

using namespace std;


//ofstream output("log.txt");

struct weakModel {
	int num;//����
	int thre;//��ֵ
	double err;//���
	bool direct;//����
};

struct classfier {
	vector <weakModel> weaks;
	vector <double> weight;
	double thre = 0;
};
struct cascade {
	vector <classfier> strongs;
};
//�����Ľṹ
struct sample {
	vector <int> img;//	ͼ��
	bool polar;//����=true������=false
	double weight;//����Ȩ��
	sample(vector <int> im, bool pon, double weig) :img(im), polar(pon), weight(weig) {}
};


class Learning {
public:
	Learning();
	~Learning();


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
	//������
	vector <sample> samp;

	cascade model;

	//the buffer definition for block sum value calsulation
	unsigned short *haar_data_x, *haar_data_y;
	unsigned short *haar_sum_x, *haar_sum_y;
	//the precalculated buffer array model for optimization
	int **pOriLABCenFeatNum;
	int **pLABFeat_Array;
	int ***pOriLABFeat_Array;

	int *pFileBuf;

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
	void initSample();

	//�ԻҶ�ֵ���������� ���ݶԽ��������ν�ʺ���
	static bool imgSmaller(const pair<int, bool> &s1, const pair<int, bool> &s2)
	{
		return s1.first < s2.first;
	}
	//ѵ�����������ĺ���
	void trainWeakClassfier(weakModel &best);
	//ѵ��ǿ�������ĺ���,����ֵΪ��ʶ�ʣ�DΪ�����
	double trainStrongClassfier(classfier &strongClassfier, double &D);

	void adjustSampleWeight(const weakModel &best, double beta);
	
	void cascade();
	//��buff��itָ��������е�ͼ����м��
	bool test(const classfier &buff, decltype(samp.begin()) it);
	//����ǿ��������ֵʹ����ʴﵽ��׼
	double adjustDetectRate(classfier &strongClassfier, double &d);
};