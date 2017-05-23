#include "Learning.h"
#include <opencv.hpp>

using namespace cv;

int main()
{
	int i, j;
	Model tempModel;
	string s;
	IplImage *src;
	vector < vector <int>> posimg;
	vector < vector <int>> negimg;
	int width = 20;
	int height = 20;
	char c[6];
	unsigned char graybuff;
	//image = (int *) malloc (sizeof ())

	//读取正例
	for (i = 0; i < POSNUM; ++i)
	{
		vector <int> temp;
		sprintf(c, "%05d", i+1);
		string buff = c;
		s = "dataset\\pos\\face" + buff;
		s += ".bmp";
		
		IplImage *src = cvLoadImage(s.c_str());
		//cvShowImage(s.c_str(),src);
		//waitKey();
		IplImage *gray = cvCreateImage(cvSize(width,height),8,1);
		if (src == NULL)
			break;
		cvCvtColor(src, gray, CV_BGR2GRAY);
		//cvShowImage(s.c_str(), gray);
		//waitKey();
		for (j = 0; j < width*height; ++j)
		{
			graybuff = gray->imageData[j];
			temp.push_back(int(graybuff));
		}
		posimg.push_back(temp);

	}
	//读取反例
	for (i = 0; i < NEGNUM; ++i)
	{
		vector <int> temp;
		sprintf(c, "%05d", i + 1);
		string buff = c;
		s = "dataset\\neg\\B1_" + buff;
		s += ".bmp";

		IplImage *src = cvLoadImage(s.c_str());
		//cvShowImage(s.c_str(),src);
		//waitKey();
		IplImage *gray = cvCreateImage(cvSize(width, height), 8, 1);
		if (src == NULL)
			break;
		cvCvtColor(src, gray, CV_BGR2GRAY);
		//cvShowImage(s.c_str(), gray);
		//waitKey();
		for (j = 0; j < width*height; ++j)
		{
			graybuff = gray->imageData[j];
			temp.push_back(int(graybuff));
		}
		negimg.push_back(temp);

	}

	//读取部分检查过没有问题

	Learning learning;
	learning.init();
	learning.setParam();
	//处理正例
	learning.GetBlockSumHaarData_All(posimg,learning.pos_block_sum_data);
	learning.GetLABData_All(learning.pos_block_sum_data,learning.pos_total_lab_data);
	//处理反例
	learning.GetBlockSumHaarData_All(negimg, learning.neg_block_sum_data);
	learning.GetLABData_All(learning.neg_block_sum_data, learning.neg_total_lab_data);



	//vector <vector <int>> temp = learning.pos_total_lab_data;
	learning.AdaBoost();

	
	while (1);
	//ofstream output("b.txt");
	//int k;
	//for (auto eachImg:learning.pos_total_lab_data)
	//{
	//	for (i = 0; i < 120; i++)
	//	{
	//		for (j = 0; j < 20; j++)
	//			output << setw(5) <<eachImg[i * 20 + j] << " ";
	//		output << endl;
	//	}
	//}

}

//int main()
//{
//	int i = 11;
//	string str;
//	char c[5];
//	sprintf(c, "%05d", i);
//	cout << c << endl;
//	while (1);
//}