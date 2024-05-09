#include "Frame.h"
#include <thread>

namespace BoW3D
{
    long unsigned int Frame::nNextId = 0;
   
    Frame::Frame(LinK3D_Extractor* pLink3dExtractor, pcl::PointCloud<pcl::PointXYZ>::Ptr pLaserCloudIn):mpLink3dExtractor(pLink3dExtractor)
    {
        //更新当前帧的编号
        mnId = nNextId++; 
        //调用括号重载函数
        (*mpLink3dExtractor)(pLaserCloudIn, mvAggregationKeypoints, mDescriptors, mClusterEdgeKeypoints);
    }

                
}