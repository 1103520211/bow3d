#include "BoW3D.h"
#include <fstream>

using namespace std;


namespace BoW3D
{
    BoW3D::BoW3D(LinK3D_Extractor* pLinK3D_Extractor, float thr_, int thf_, int num_add_retrieve_features_): 
            mpLinK3D_Extractor(pLinK3D_Extractor), 
            thr(thr_), 
            thf(thf_), 
            num_add_retrieve_features(num_add_retrieve_features_)
    {
       N_nw_ofRatio = std::make_pair(0, 0); 
    }
    //为什么只考虑每一帧的前五个描述子? 
    //如果特征值对应的位置信息不存在，则创建新的位置信息，并将其存储在特征数据库中。
    //如果特征值对应的位置信息已存在，则将当前位置信息添加到已存在的位置集合中(位置集合代表的是某个描述子列数一样和值一样  但是帧的id或者描述子的当前帧行数不一样组成的集合)

    //可以理解成对于每一帧来讲，每一个描述子每一位对应的都是一个特征，有距离和维度（列数代表他的扇区，扇区就代表他的方向信息），
    //然后我们提取的就是他每一位的信息，就有了距离和方向
    //实时创建的词袋,根据当前帧的描述子信息更新特征数据库  单词存放first（距离，列数j）,second（帧id，描述子行数）
    void BoW3D::update(Frame* pCurrentFrame)
    {
        mvFrames.emplace_back(pCurrentFrame);
        //把当前帧的ID和描述子赋值
        cv::Mat descriptors = pCurrentFrame->mDescriptors;
        long unsigned int frameId = pCurrentFrame->mnId;
        //描述子个数
        size_t numFeature = descriptors.rows;
        //判断描述子的个数是否小于某个阈值
        if(numFeature < (size_t)num_add_retrieve_features) 
        {
            //遍历每一行
            for(size_t i = 0; i < numFeature; i++)
            {
                float *p = descriptors.ptr<float>(i);
                //搜索每一个描述子的每一列
                for(size_t j = 0; j < (size_t)descriptors.cols; j++)
                {
                    //判断此描述子当前列的值是否为0
                    if(p[j] != 0)
                    {
                        //声明了一个能够遍历键为 pair<float, int> 类型、值为 unordered_set<pair<int, int>, pair_hash> 
                        //类型的 unordered_map 的迭代器
                        unordered_map<pair<float, int>, unordered_set<pair<int, int>, pair_hash>, pair_hash>::iterator it; 
                        //键 描述子的当前列的值  和当前列
                        pair<float, int> word= make_pair(p[j], j);
                        //查找当前特征值的位置信息是否已经存在
                        it = this->find(word);
                        //找不到对应的特征值，则将当前特征值的位置信息插入到 std::unordered_map 中
                        if(it == this->end())
                        {
                            //存储该特征值对应的位置信息
                            unordered_set<pair<int,int>, pair_hash> place;
                            place.insert(make_pair(frameId, i));
                            //键和值对应存放  this指向当前对象  存放在bow3d对象里面
                            (*this)[word] = place;

                            N_nw_ofRatio.first++;//一共有多少个键值
                            N_nw_ofRatio.second++;//一共有多少个位置条目  指的是描述子中不为0的个数
                        }
                        else
                        {
                            //当前的键 word 已经存在于 std::unordered_map 中，说明这是一个已知的特征值，
                            //只需将新的位置信息添加到对应的 std::unordered_set 中即可。

                            //std::unordered_set<pair<int,int>, pair_hash>，它是一个无序集合，可以存储多个值。
                            //在这个分支中，代码是向这个值中插入了一个新的 pair<int, int> 对，而不是直接修改了已经存在的键的值。所以即使键是重复的，但是值是一个集合
                            (*it).second.insert(make_pair(frameId, i));
                            //一共有多少个位置条目 指的是所有不为0位，不仅仅是特征数量
                            N_nw_ofRatio.second++;
                        }

                    }
                }
            }
        }
        //大于5
        else
        {
            //
            for(size_t i = 0; i < (size_t)num_add_retrieve_features; i++)
            {
                float *p = descriptors.ptr<float>(i);
                for(size_t j = 0; j < (size_t)descriptors.cols; j++)
                {
                    if(p[j] != 0)
                    {
                        unordered_map<pair<float, int>, unordered_set<pair<int, int>, pair_hash>, pair_hash>::iterator it; 

                        pair<float, int> word= make_pair(p[j], j);
                        it = this->find(word);

                        if(it == this->end())
                        {
                            unordered_set<pair<int,int>, pair_hash> place;
                            place.insert(make_pair(frameId, i));
                            (*this)[word] = place;

                            N_nw_ofRatio.first++;
                            N_nw_ofRatio.second++;
                        }
                        else
                        {
                            (*it).second.insert(make_pair(frameId, i));

                            N_nw_ofRatio.second++;
                        }
                    }
                }
            }
        }
    }
       
    //回环检测
    //比较描述子中的特征  得到帧序号  然后再去比较该帧出现的频率 是否大于某个阈值  大于则进行闭环匹配
    void BoW3D::retrieve(Frame* pCurrentFrame, int &loopFrameId, Eigen::Matrix3d &loopRelR, Eigen::Vector3d &loopRelt)
    {        
        int frameId = pCurrentFrame->mnId;

        cv::Mat descriptors = pCurrentFrame->mDescriptors;      
        //记录描述子个数
        size_t rowSize = descriptors.rows;       
        
        //map<int, int>mScoreFrameID;
        std::multimap<int, int> mScoreFrameID;//速度好像慢了很多,
        //判断当前帧的描述子是否小于给定值
        if(rowSize < (size_t)num_add_retrieve_features) 
        {   //遍历每一个描述子
            for(size_t i = 0; i < rowSize; i++)
            {
                unordered_map<pair<int, int>, int, pair_hash> mPlaceScore;                
                                
                float *p = descriptors.ptr<float>(i);

                int countValue = 0;

                for(size_t j = 0; j < (size_t)descriptors.cols; j++)
                {
                    countValue++;

                    if(p[j] != 0)
                    {                   
                        pair<float, int> word = make_pair(p[j], j); 
                        //在实时的词袋里面搜索特征 
                        auto wordPlacesIter = this->find(word);//指向的是词袋中单词的位置
                        //
                        if(wordPlacesIter == this->end())
                        {
                            continue;
                        }
                        //搜到了对应特征
                        else
                        {
                            //N_nw_ofRatio.second代表一共位置条目  N_nw_ofRatio.first代表的是总的特征数
                            double averNumInPlaceSet = N_nw_ofRatio.second / N_nw_ofRatio.first;
                            //当前单词中的条目数量  也就是多少个位置看到了该特征
                            int curNumOfPlaces = (wordPlacesIter->second).size();
                            
                            double ratio = curNumOfPlaces / averNumInPlaceSet;
                            //说明这个特征太多帧看到了
                            if(ratio > thr)
                            {
                                continue;
                            }
                            
                            //对于其中一个描述子来讲，遍历了每一位，记录当前列中对应的帧对，统计到mPlaceScore，次数是针对当前，描述子下的 .对每一个描述子处理了之后，mPlaceScore自动释放，
                            for(auto placesIter = (wordPlacesIter->second).begin(); placesIter != (wordPlacesIter->second).end(); placesIter++)
                            {
                                //The interval between the loop and the current frame should be at least 300.
                                //循环与当前帧之间的间隔至少应为300
                                if(frameId - (*placesIter).first < 300) 
                                {
                                    continue;
                                }

                                auto placeNumIt = mPlaceScore.find(*placesIter);                    
                                //如果当前位置对在 mPlaceScore 中不存在，则将其插入，并将计数器初始化为 1；
                                //如果已经存在，则将计数器加一。这样可以统计满足条件的位置对出现的次数。
                                if(placeNumIt == mPlaceScore.end())
                                {                                
                                    mPlaceScore[*placesIter] = 1;
                                }
                                else
                                {
                                    mPlaceScore[*placesIter]++;                                    
                                }                                                              
                            }                       
                        }                            
                    }                    
                }
                //处理了一个描述子之后，统计当前描述子中满足阈值的帧对
                //遍历 mPlaceScore 中的每个位置对，检查其出现次数是否超过阈值 thf。如果超过了阈值，
                //程序会将该位置对的出现次数作为键，对应的帧的索引作为值，插入到 mScoreFrameID 中。
                for(auto placeScoreIter = mPlaceScore.begin(); placeScoreIter != mPlaceScore.end(); placeScoreIter++)
                {
                    //当前位置对的出现次数超过阈值 thf
                    if((*placeScoreIter).second > thf) 
                    {  ///会不会导致位置对应的出现次数一样的帧数被覆盖了? //速度好像慢了很多,怎么解决这个重复问题
                       //mScoreFrameID[(*placeScoreIter).second] = ((*placeScoreIter).first).first;//((*placeScoreIter).first).first帧索引
                       int count = 0;
                       count = mScoreFrameID.count(((*placeScoreIter).first).first);
                       if (count == 0) {
                            mScoreFrameID.insert(std::make_pair((*placeScoreIter).second, ((*placeScoreIter).first).first));
                       }
                    }
                }                                   
            }//遍历所有的描述子，得到满足阈值帧对                  
        }
        else
        {
            for(size_t i = 0; i < (size_t)num_add_retrieve_features; i++) 
            {
                unordered_map<pair<int, int>, int, pair_hash> mPlaceScore;
                
                float *p = descriptors.ptr<float>(i);

                int countValue = 0;

                for(size_t j = 0; j < (size_t)descriptors.cols; j++)
                {
                    countValue++;

                    if(p[j] != 0)
                    {                   
                        pair<float, int> word = make_pair(p[j], j);    

                        auto wordPlacesIter = this->find(word);

                        if(wordPlacesIter == this->end())
                        {
                            continue;
                        }
                        else
                        {
                            double averNumInPlaceSet = (double) N_nw_ofRatio.second / N_nw_ofRatio.first;
                            int curNumOfPlaces = (wordPlacesIter->second).size();

                            double ratio = curNumOfPlaces / averNumInPlaceSet;

                            if(ratio > thr)
                            {
                                continue;
                            }

                            for(auto placesIter = (wordPlacesIter->second).begin(); placesIter != (wordPlacesIter->second).end(); placesIter++)
                            {
                                //The interval between the loop and the current frame should be at least 300.
                                if(frameId - (*placesIter).first < 300) 
                                {
                                    continue;
                                }

                                auto placeNumIt = mPlaceScore.find(*placesIter);                    
                                
                                if(placeNumIt == mPlaceScore.end())
                                {                                
                                    mPlaceScore[*placesIter] = 1;
                                }
                                else
                                {
                                    mPlaceScore[*placesIter]++;                                    
                                }                                                              
                            }                       
                        }                            
                    }
                }

                for(auto placeScoreIter = mPlaceScore.begin(); placeScoreIter != mPlaceScore.end(); placeScoreIter++)
                {
                    if((*placeScoreIter).second > thf) 
                    {
                       //第一项 ((*placeScoreIter).second) 是特征数量 值((*placeScoreIter).first).first（第二项）表示的是帧的序号
                       //mScoreFrameID[(*placeScoreIter).second] = ((*placeScoreIter).first).first;
                       //可否加一个判断传入的该帧已经存好了
                       int count = 0;
                       count = mScoreFrameID.count(((*placeScoreIter).first).first);
                       if (count == 0) {
                            mScoreFrameID.insert(std::make_pair((*placeScoreIter).second, ((*placeScoreIter).first).first));
                       }
                    }
                }                                   
            }                           
        }     

        if(mScoreFrameID.size() == 0)
        {
            return;
        }
        //遍历前面得到的所有可能是闭环帧的序号  并没有比较哪帧好坏 哪帧先满足条件就输出了? 先使用的是帧数出现多的
        for(auto it = mScoreFrameID.rbegin(); it != mScoreFrameID.rend(); it++)
        {          
            //可以加个判断语句，来判断当前帧是否判断了

            int loopId = (*it).second;

            //获得循环帧
            Frame* pLoopFrame = mvFrames[loopId];
            vector<pair<int, int>> vMatchedIndex;  
            //帧间匹配 计算匹配分数  当前帧与闭环帧之间的关键点配对
            mpLinK3D_Extractor->match(pCurrentFrame->mvAggregationKeypoints, pLoopFrame->mvAggregationKeypoints, pCurrentFrame->mDescriptors, pLoopFrame->mDescriptors, vMatchedIndex);               

            int returnValue = 0;
            Eigen::Matrix3d loopRelativeR;
            Eigen::Vector3d loopRelativet;
                                
            returnValue = loopCorrection(pCurrentFrame, pLoopFrame, vMatchedIndex, loopRelativeR, loopRelativet);

            //The distance between the loop and the current should less than 3m.    哪帧先满足条件就输出了?               
            if(returnValue != -1 && loopRelativet.norm() < 3 && loopRelativet.norm() > 0) 
            {
                loopFrameId = (*it).second;
                loopRelR = loopRelativeR;
                loopRelt = loopRelativet;                         
                
                return;
            }     
        } 
    }


    int BoW3D::loopCorrection(Frame* currentFrame, Frame* matchedFrame, vector<pair<int, int>> &vMatchedIndex, Eigen::Matrix3d &R, Eigen::Vector3d &t)
    {
        //匹配的点对数少于30
        if(vMatchedIndex.size() <= 30)
        {
            return -1;
        }

        ScanEdgePoints currentFiltered;
        ScanEdgePoints matchedFiltered;
        //对输入的点云聚类进行低曲率点的过滤，筛选出每个聚类中曲率最大的点或者只包含一个点的子聚类
        mpLinK3D_Extractor->filterLowCurv(currentFrame->mClusterEdgeKeypoints, currentFiltered);
        mpLinK3D_Extractor->filterLowCurv(matchedFrame->mClusterEdgeKeypoints, matchedFiltered);

        vector<std::pair<PointXYZSCA, PointXYZSCA>> matchedEdgePt;
        //找到对应的边缘关键点
        mpLinK3D_Extractor->findEdgeKeypointMatch(currentFiltered, matchedFiltered, vMatchedIndex, matchedEdgePt);
        
        pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>());

        pcl::CorrespondencesPtr corrsPtr (new pcl::Correspondences()); 

        for(int i = 0; i < (int)matchedEdgePt.size(); i++)
        {
            std::pair<PointXYZSCA, PointXYZSCA> matchPoint = matchedEdgePt[i];

            pcl::PointXYZ sourcePt(matchPoint.first.x, matchPoint.first.y, matchPoint.first.z);            
            pcl::PointXYZ targetPt(matchPoint.second.x, matchPoint.second.y, matchPoint.second.z);
            
            source->push_back(sourcePt);
            target->push_back(targetPt);
            //用于表示两个点云或特征之间的对应关系。它通常用于保存匹配点的索引以及它们之间的距离或相似性分数
            pcl::Correspondence correspondence(i, i, 0);
            corrsPtr->push_back(correspondence);
        }

        pcl::Correspondences corrs;
        //用于执行 RANSAC 算法来剔除不符合模型的匹配点对
        pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZ> Ransac_based_Rejection;
        Ransac_based_Rejection.setInputSource(source);
        Ransac_based_Rejection.setInputTarget(target);
        //设置 RANSAC 算法的阈值
        double sac_threshold = 0.4;
        Ransac_based_Rejection.setInlierThreshold(sac_threshold);
        Ransac_based_Rejection.getRemainingCorrespondences(*corrsPtr, corrs);
        //
        if(corrs.size() <= 100)
        {
            return -1;
        } 

        //svd分解 
        
        Eigen::Vector3d p1 = Eigen::Vector3d::Zero();
        Eigen::Vector3d p2 = p1;
        int corrSize = (int)corrs.size();
        //计算所有匹配点对的坐标之和，分别对应于源点云和目标点云
        for(int i = 0; i < corrSize; i++)
        {  
            pcl::Correspondence corr = corrs[i];         
            p1(0) += source->points[corr.index_query].x;
            p1(1) += source->points[corr.index_query].y;
            p1(2) += source->points[corr.index_query].z; 

            p2(0) += target->points[corr.index_match].x;
            p2(1) += target->points[corr.index_match].y;
            p2(2) += target->points[corr.index_match].z;
        }
        //计算源点云和目标点云的质心坐标 center1 和 center2，即将所有点的坐标之和除以匹配点对的数量
        Eigen::Vector3d center1 = Eigen::Vector3d(p1(0)/corrSize, p1(1)/corrSize, p1(2)/corrSize);
        Eigen::Vector3d center2 = Eigen::Vector3d(p2(0)/corrSize, p2(1)/corrSize, p2(2)/corrSize);
        //存储去中心化后的匹配点对。
        vector<Eigen::Vector3d> vRemoveCenterPt1, vRemoveCenterPt2;
        //对每个匹配点对进行去中心化，即将每个点的坐标减去相应的质心坐标
        for(int i = 0; i < corrSize; i++)
        {
            pcl::Correspondence corr = corrs[i];
            pcl::PointXYZ sourcePt = source->points[corr.index_query];
            pcl::PointXYZ targetPt = target->points[corr.index_match];

            Eigen::Vector3d removeCenterPt1 = Eigen::Vector3d(sourcePt.x - center1(0), sourcePt.y - center1(1), sourcePt.z - center1(2));
            Eigen::Vector3d removeCenterPt2 = Eigen::Vector3d(targetPt.x - center2(0), targetPt.y - center2(1), targetPt.z - center2(2));
        
            vRemoveCenterPt1.emplace_back(removeCenterPt1);
            vRemoveCenterPt2.emplace_back(removeCenterPt2);
        }
        //创建了一个零矩阵 w，用于存储去中心化后的匹配点对的乘积之和
        Eigen::Matrix3d w = Eigen::Matrix3d::Zero();

        for(int i = 0; i < corrSize; i++)
        {
            w += vRemoveCenterPt1[i] * vRemoveCenterPt2[i].transpose();
        }      

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(w, Eigen::ComputeFullU|Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        
        R = V * U.transpose();
        t = center2 - R * center1;

        return 1;
    }

}
