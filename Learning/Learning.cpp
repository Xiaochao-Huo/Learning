#include "Learning.h"

Learning::Learning()
{
	detectModel = NULL;

	//the precalculated buffer array model for optimization
	pOriLABCenFeatNum = NULL;
	pLABFeat_Array = NULL;
	pOriLABFeat_Array = NULL;

	//the downsampling scale and rate struct
	pSampleRate = NULL;

	//the file buffer
	pFileBuf = NULL;
	//the buffer definition for block sum value calsulation
	haar_data_x = NULL;
	haar_data_y = NULL;
	haar_sum_x = NULL;
	haar_sum_y = NULL;
}
Learning::~Learning()
{

}
void Learning::init()
{
	int i, j;
	int k, n;

	releaseAll();

	detectModel = (Model *)malloc(sizeof(Model));
	detectModel->step = 0;
	detectModel->thres = 0;
	detectModel->pModel = NULL;

	pOriLABCenFeatNum = (int **)malloc(sizeof(int *));

	pSampleRate = (pyramidSample *)malloc(sizeof(pyramidSample)*MAX_IMG_SCALE_NUM);
	n = 0;
	for (i = 0; i < MAX_IMG_SCALE_NUM; i += level_num_one_region)
	{
		for (j = 0; j < level_num_one_region; ++j)
		{
			pSampleRate[i + j].deflate_rate = 1.0;
			for (k = 0; k < j; ++k)
				pSampleRate[i + j].deflate_rate *= total_delta_rate_one_level;
			pSampleRate[i + j].down_sample_move_2 = n;
			pSampleRate[i + j].down_sample_rate_2 = 1;
			for (k = 0; k < n; ++k)
				pSampleRate[i + j].down_sample_rate_2 *= 2;
			pSampleRate[i + j].total_deflate_rate = pSampleRate[i + j].down_sample_rate_2*pSampleRate[i + j].deflate_rate;
		}
		n++;
	}
}
//设置参数
void Learning::setParam(int confMinThres, int weightMinThres, int Xshift, int Yshift, int endScaleNum, int startScaleNum)
{
	Candidate_Combine_Conf_MinThres = confMinThres;
	Candidate_Combine_Weight_MinThres = weightMinThres;
	Detect_Search_XShift = Xshift;
	Detect_Search_YShift = Yshift;
	face_detect_end_search_scale_num = endScaleNum;
	face_detect_start_search_scale_num = startScaleNum;
}
//初始化内存（内存管理c++可以删掉？）
//void Learning::initMemory(int wd, int ht)
//{
//	int i, j, k, l, n;
//	int cur_wd, cur_ht;
//	int image_size;
//
//	releaseMemory();
//
//	image_wd = wd;
//	image_ht = ht;
//
//	haar_data_x = (unsigned short *)malloc(sizeof(unsigned short)*ht*wd);
//	haar_data_y = (unsigned short *)malloc(sizeof(unsigned short)*ht*wd);
//	haar_sum_x = (unsigned short *)malloc(sizeof(unsigned short)*ht*wd);
//	haar_sum_y = (unsigned short *)malloc(sizeof(unsigned short)*ht*wd);
//	memset(haar_data_x, 0, sizeof(unsigned short)*ht*wd);
//	memset(haar_data_y, 0, sizeof(unsigned short)*ht*wd);
//	memset(haar_sum_x, 0, sizeof(unsigned short)*ht*wd);
//	memset(haar_sum_y, 0, sizeof(unsigned short)*ht*wd);
//
//	pLABFeat_Array = (int **)malloc(sizeof(int *)*MAX_IMG_SCALE_NUM);
//	for (i = 0; i < MAX_IMG_SCALE_NUM; ++i)
//	{
//		cur_ht = int(ht / pSampleRate[i].total_deflate_rate);
//		cur_wd = int(wd / pSampleRate[i].total_deflate_rate);
//		image_size = cur_wd*cur_ht;
//		pLABFeat_Array[i] = (int *)malloc(sizeof(int)*total_LAB_num);
//		for (j = 0; j < LAB_feat_type; ++j)
//			for (k = 0; k < face_rect_ht; ++k)
//				for (l = 0; l < face_rect_wd; ++l)
//				{
//					n = j*face_rect_wd*face_rect_ht + k*face_rect_wd + l;
//					pLABFeat_Array[i][n] = j*image_size + k*cur_wd + l;
//				}
//	}
//}
//释放内存
void Learning::releaseMemory()
{
	int i, j;

	if (pLABFeat_Array)
	{
		for (j = 0; j < MAX_IMG_SCALE_NUM; ++j)
			free(pLABFeat_Array[j]);
		free(pLABFeat_Array);
		pLABFeat_Array = NULL;
	}
	if (pOriLABFeat_Array)
	{
		for (i = 0; i < MAX_IMG_SCALE_NUM; ++i)
		{
			for (j = 0; j < 1; ++j)
				free(pOriLABFeat_Array[i][j]);
			free(pOriLABFeat_Array[i]);
		}
		free(pOriLABFeat_Array);
		pOriLABFeat_Array = NULL;
	}
	if (haar_data_x)
	{
		free(haar_data_x);
		haar_data_x = NULL;
	}
	if (haar_data_y)
	{
		free(haar_data_y);
		haar_data_y = NULL;
	}
	if (haar_sum_x)
	{
		free(haar_sum_x);
		haar_sum_x = NULL;
	}
	if (haar_sum_y)
	{
		free(haar_sum_y);
		haar_sum_y = NULL;
	}
}
void Learning::releaseAll()
{
	int j;
	if (pOriLABCenFeatNum)
	{
		for (j = 0; j < 1; ++j)
			free(pOriLABCenFeatNum[j]);
		free(pOriLABCenFeatNum);
		pOriLABCenFeatNum = NULL;
	}
	if (pSampleRate)
	{
		free(pSampleRate);
		pSampleRate = NULL;
	}
	if (detectModel)
	{
		free(detectModel);
		detectModel = NULL;
	}
	if (pFileBuf)
	{
		free(pFileBuf);
		pFileBuf = NULL;
	}
	releaseMemory();
}

//获得6通道的haar特征图像
void Learning::GetBlockSumHaarData_All(vector < vector <int>> &img,vector < vector <int>> &block_sum_data)
{
	int k, i, j;
	int haar_ht, haar_wd;

	for (auto eachImg:img)
	{
		vector <int> temp;
		haar_wd = 2;
		haar_ht = 1;
		for (i = 0; i < ht + 1 - haar_ht; ++i)
		{
			for (j = 0; j < wd + 1 - haar_wd; ++j)
			{
				temp.push_back(eachImg[i * wd + j] + eachImg[i * wd + j + 1]);
			}
			temp.push_back(0);
		}
		for (j = 0; j < haar_ht * wd; ++j)
			temp.push_back(0);
		for (i = haar_ht; i < ht + 1 - haar_ht; ++i)
		{
			for (j = 0; j < haar_wd; ++j)
				temp.push_back(0);
			for (j = haar_wd; j < wd + 1 - haar_wd; ++j)
			{
				temp.push_back(abs(temp[i * wd + j] + temp[(i - haar_ht) * wd + j] - temp[i * wd + j - haar_wd] - temp[(i - haar_ht) * wd + j - haar_wd])<<1);
			}
			temp.push_back(0);
		}
		for (j = 0; j < haar_ht * wd; ++j)
			temp.push_back(0);
		for (i = haar_ht; i < ht + 1 - haar_ht; ++i)
		{
			for (j = 0; j < haar_wd; ++j)
				temp.push_back(0);
			for (j = haar_wd; j < wd + 1 - haar_wd; ++j)
			{
				temp.push_back(abs(temp[i * wd + j] - temp[(i - haar_ht) * wd + j] + temp[i * wd + j - haar_wd] - temp[(i - haar_ht) * wd + j - haar_wd])<<1);
			}
			temp.push_back(0);
		}

		haar_wd = 1;
		haar_ht = 2;

		for (i = 0; i < ht + 1 - haar_ht; ++i)
		{
			for (j = 0; j < wd + 1 - haar_wd; ++j)
			{
				temp.push_back(eachImg[i * wd + j] + eachImg[(i + 1) * wd + j]);
			}
		}
		for (j = 0; j < wd; ++j)
			temp.push_back(0);
		for (j = 0; j < haar_ht * wd; ++j)
			temp.push_back(0);
		for (i = haar_ht; i < ht + 1 - haar_ht; ++i)
		{
			for (j = 0; j < haar_wd; ++j)
				temp.push_back(0);
			for (j = haar_wd; j < wd + 1 - haar_wd; ++j)
			{
				temp.push_back(abs(temp[(i + 60) * wd + j] + temp[((i + 60) - haar_ht) * wd + j] - temp[(i + 60) * wd + j - haar_wd] - temp[((i + 60) - haar_ht) * wd + j - haar_wd])<<1);
			}
		}
		for (j = 0; j < wd; ++j)
			temp.push_back(0);
		for (j = 0; j < haar_ht * wd; ++j)
			temp.push_back(0);
		for (i = haar_ht; i < ht + 1 - haar_ht; ++i)
		{
			for (j = 0; j < haar_wd; ++j)
				temp.push_back(0);
			for (j = haar_wd; j < wd + 1 - haar_wd; ++j)
			{
				temp.push_back(abs(temp[(i + 60) * wd + j] - temp[((i + 60) - haar_ht) * wd + j] + temp[(i + 60) * wd + j - haar_wd] - temp[((i + 60) - haar_ht) * wd + j - haar_wd])<<1);
			}
		}
		for (j = 0; j < wd; ++j)
			temp.push_back(0);
		block_sum_data.push_back(temp);
	}

}

//获得6通道的LAB特征图像
void Learning::GetLABData_All(vector < vector <int>> &block_sum_data, vector < vector <int>> &total_lab_data)
{
	int ii, i, j, k;
	int cenArray;
	int haar_ht, haar_wd;
	int n, r;

	int lab_feat;
	for (auto eachImg:block_sum_data)
	{
		vector<int> temp;
		for (ii = 0; ii < LAB_block_size_num; ++ii)
		{
			if (ii == 0)
			{
				haar_wd = 2;
				haar_ht = 1;
			}
			else
			{
				haar_wd = 1;
				haar_ht = 2;
			}

			int temp_multi_data = haar_ht*wd;
			for (r = 0; r < 3; ++r)
			{
				for (i = 0; i < ht + 1 - 3 * haar_ht; ++i)
				{

					for (j = 0; j < wd + 1 - 3 * haar_wd; ++j)
					{

							n = i * wd + j + haar_wd + temp_multi_data + r * wd * ht;
					

						cenArray = eachImg[n];
						lab_feat = 0;
						lab_feat += (cenArray < eachImg[n - temp_multi_data - haar_wd]) ? 1 : 0;
						lab_feat += (cenArray < eachImg[n - temp_multi_data]) ? 2 : 0;
						lab_feat += (cenArray < eachImg[n - temp_multi_data + haar_wd]) ? 4 : 0;
						lab_feat += (cenArray < eachImg[n + haar_wd]) ? 8 : 0;
						lab_feat += (cenArray < eachImg[n + temp_multi_data + haar_wd]) ? 16 : 0;
						lab_feat += (cenArray < eachImg[n + temp_multi_data]) ? 32 : 0;
						lab_feat += (cenArray < eachImg[n + temp_multi_data - haar_wd]) ? 64 : 0;
						lab_feat += (cenArray < eachImg[n - haar_wd]) ? 128 : 0;
						temp.push_back(lab_feat);
					}
					for (j = 0; j < 3 * haar_wd - 1; ++j)
						temp.push_back(0);
				}
				for (i = 0; i < 3 * haar_ht - 1; ++i)
					for (j = 0; j < wd; ++j)
						temp.push_back(0);
			}
		}
		total_lab_data.push_back(temp);
	}
}

void Learning::AdaBoost()
{
	initSample();
	//auto test = samp[5];
	//for (auto i : test.img)
	//	cout << i << " ";
	//cout << endl;
	//cout << test.polar << endl;
	//cout << test.weight;
	int T = 1;//AdaBoost弱分类器个数
	double totalW;
	for (int i = 0; i < T; ++i)
	{
		int pmax, pmin;
		int nper;
		pair<int, double> best;
		//归一化权重
		totalW = 0;
		for (auto s : samp)
			totalW += s.weight;
		for (auto s : samp)
			s.weight /= totalW;

		for (int j = 0; j < 2400; ++j)//对每个特征，即6*20*20个特征
		{
			vector <int> tempPosData;
			vector <int> tempNegData;

			for (int k = 0; k < 256;++k)
			{
				int l = 0;
				//获得正负样本这个特征的值
				for (;; ++l)
				{
					if (!samp[l].polar)
						break;
					tempPosData.push_back(samp[l].img[j]);
				}
				for (; l < samp.size(); ++l)
					tempNegData.push_back(samp[l].img[j]);
				//将所有特征值分正负样本进行排序
				sort(tempPosData.begin(), tempPosData.end());
				sort(tempNegData.begin(), tempNegData.end());
				//for (auto p : tempPosData)
				//	cout << p << " ";
				//cout << endl;
				//for (auto n : tempNegData)
				//	cout << n << " ";
			}
		}
	}

}

void Learning::initSample()
{

	double p, n;//正负样本数量
	double pp, nn;//正负样本初始权重

	p = pos_total_lab_data.size();
	n = neg_total_lab_data.size();

	//计算初始权重
	pp = 1 / (2 * p);
	nn = 1 / (2 * n);

	cout << n << " " << p << " " << nn << " " << pp << endl;
	for (auto v : pos_total_lab_data)
		samp.push_back(sample(v, true, pp));
	for (auto v : neg_total_lab_data)
		samp.push_back(sample(v, false, nn));

}