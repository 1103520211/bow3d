#include <ros/ros.h>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <eigen3/Eigen/Dense>
#include <pcl/filters/extract_indices.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "LinK3D_Extractor.h"
#include "BoW3D.h"
#include <map>


using namespace std;
using namespace BoW3D;


//Parameters of LinK3D
int nScans = 64; //Number of LiDAR scan lines
float scanPeriod = 0.1; 
float minimumRange = 0.1;
float distanceTh = 0.4;
int matchTh = 6;

//Parameters of BoW3D
float thr = 3.5;//3.5 大于该值就不会对当前帧下某个描述子的特征处理
int thf = 5;//5  对检索时间影响很大
int num_add_retrieve_features = 5;//5
//loopclosure
int LoopIndex = 0;
//endIndex
int endIndex = 0;
//map<pair<int, int>, int> ground_Map;
map<pair<int, int>, int> bow3d_Map;
vector<int> F_loop;



//Here, the KITTI's point cloud with '.bin' format is read.
vector<float> read_lidar_data(const std::string lidar_data_path)
{
    std::ifstream lidar_data_file;
    lidar_data_file.open(lidar_data_path, std::ifstream::in | std::ifstream::binary);//two way

    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char*>(&lidar_data_buffer[0]), num_elements*sizeof(float));
    return lidar_data_buffer;
}

//对真值处理
// 定义关键帧类
struct KeyFrame {
    int FrameID_;
    int BinID_;
    Eigen::Quaterniond rotation_;  // 旋转姿态
    Eigen::Vector3d translation_;  // 平移向量
};

// 计算两个姿态之间的旋转角度
double computeRotationAngle(const Eigen::Quaterniond& q1, const Eigen::Quaterniond& q2) {
    Eigen::Quaterniond delta_q = q2 * q1.inverse();
    double angle = 2.0 * acos(delta_q.w()) * 180.0 / M_PI; // 角度表示
    return angle;
}

// 计算两个姿态之间的距离
double computeTranslationDistance(const Eigen::Vector3d& t1, const Eigen::Vector3d& t2) {
    return (t2 - t1).norm();
}

// 关键帧检测
void dynamicKeyFrameDetection(const vector<pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses, vector<KeyFrame>& keyFrames, double rotationThreshold, double translationThreshold) {
     
    keyFrames.push_back(KeyFrame{0, 0, poses[0].first, poses[0].second});  // 第一帧作为初始关键帧

    // 从第二帧开始遍历
    int key_frame_id = 1;
    for (int i = 1; i < poses.size(); ++i) {
        const auto& currentPose = poses[i];
        const auto& lastKeyFramePose = keyFrames.back();

        // 计算当前帧与上一个关键帧之间的旋转角度和距离
        double rotationAngle = computeRotationAngle(lastKeyFramePose.rotation_, currentPose.first);
        double distance = computeTranslationDistance(lastKeyFramePose.translation_, currentPose.second);

        // 如果超过阈值，则将当前帧定义为新的关键帧并存储起来 && (i - lastKeyFramePose.getFrameID()) > 10
        if ((rotationAngle > rotationThreshold || distance > translationThreshold)) {
            keyFrames.push_back(KeyFrame{key_frame_id++, i, currentPose.first, currentPose.second});  // 存储当前帧作为关键帧
            //cout << "Frame " << i << " is a keyframe." << endl;
        }
    }
    // 输出关键帧的数量
    for (const auto& FramesID : keyFrames) {
        //cout << "ID of keyFrames: " << FramesID.getFrameID() << endl;
    }
    cout << "Number of keyframes: " << keyFrames.size() << endl;
}

// 从TUM文件中读取姿态数据并存储到向量中
vector<pair<Eigen::Quaterniond, Eigen::Vector3d>> readTUMFile(const string& filename) {
    vector<pair<Eigen::Quaterniond, Eigen::Vector3d>> poses;
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        istringstream iss(line);
        vector<double> data;
        double value;
        while (iss >> value) {
            data.push_back(value);
        }
        if (data.size() >= 8) {
            Eigen::Quaterniond rotation(data[7], data[4], data[5], data[6]);
            Eigen::Vector3d translation(data[1], data[2], data[3]);
            poses.emplace_back(rotation.normalized(), translation); // 将四元数归一化后存储
        }
    }
    return poses;
}
//处理完毕
//检测真实的闭环点(关键帧)
// map<pair<int, int>, int> findLoopClosures(const vector<KeyFrame>& keyFrames, const vector<pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses, double rotationThreshold, double translationThreshold) {
//     map<pair<int, int>, int> loopClosures;  // 使用map存储闭环点对，键为点对，值为当前帧的索引

//     // 遍历所有关键帧
//     for (size_t i = 0; i < keyFrames.size(); ++i) {
//         const auto& keyFrame = keyFrames[i];

//         // 获取当前关键帧的姿态
//         const auto& keyFrameRotation = keyFrame.getRotation();
//         const auto& keyFrameTranslation = keyFrame.getTranslation();

//         // 遍历所有帧以查找可能的闭环
//         for (size_t j = 0; j < poses.size(); ++j) {
            
//                 const auto& pose = poses[j];

//                 // 计算当前帧与关键帧之间的旋转角度和距离
//                 double rotationAngle = computeRotationAngle(keyFrameRotation, pose.first);
//                 double distance = computeTranslationDistance(keyFrameTranslation, pose.second);
//                 if (abs(keyFrame.getFrameID() - (int)j) < 300) {
//                     continue;
//                 }
//                 // 如果超过阈值，则认为可能存在闭环
//                 if (rotationAngle < rotationThreshold && distance < translationThreshold) {
//                     // 存储闭环帧对
//                     if (keyFrame.getFrameID() > j) {
//                         loopClosures[{keyFrame.getFrameID(), j}] = keyFrame.getFrameID(); // 存储闭环点对，值为当前帧的索引
//                     }
//                     if (keyFrame.getFrameID() < j) {
//                         loopClosures[{j, keyFrame.getFrameID()}] = keyFrame.getFrameID(); // 存储闭环点对，值为当前帧的索引
//                     }                   
//                 }          
//         }
//     }   
//     // 输出闭环点对和对应的当前帧索引
//     for (const auto& entry : loopClosures) {
//         //cout << "Loop closure: (" << entry.first.first << ", " << entry.first.second << ") -> " << entry.second << endl;
//     }

//     return loopClosures;
// }
std::vector<set<int>> findLoopClosures(const vector<KeyFrame>& keyFrames, double translationThreshold) {
    int n = keyFrames.size();
    std::vector<set<int>> loopClosures(n);  // 使用map存储闭环点对，键为关键帧的ID，值为包含所有闭环点对的容器

    // 遍历所有关键帧
    for (size_t i = 0; i < n; ++i) {
        const auto& keyFrame = keyFrames[i];
        // 遍历所有帧以查找可能的闭环
        for (size_t j = 0; (i - j) > 300; ++j) {
            const auto& keyFramej = keyFrames[j];

            // 计算当前帧与关键帧之间的旋转角度和距离
            double distance = computeTranslationDistance(keyFrame.translation_, keyFramej.translation_);
            
            // 认为可能存在闭环
            if (distance < translationThreshold) {
                // 存储闭环帧对，以关键帧的ID为键
                loopClosures[i].insert(j);
            }   
        }
    }   

    return loopClosures;
}

// map<pair<int, int>, int> findPoesLoop(const vector<pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses, double rotationThreshold, double translationThreshold) {
//     map<pair<int, int>, int> loopClosures;  // 使用map存储闭环点对，键为点对，值为当前帧的索引
//     for (int i = 301; i < poses.size(); i++) {
//         const auto& crunFrame = poses[i];
//         for (int j = 0; j < i - 300; j++) {
//             const auto& loopFrame = poses[j];
//             double rotationAngle = computeRotationAngle(crunFrame.first, loopFrame.first);
//             double distance = computeTranslationDistance(crunFrame.second, loopFrame.second);
//             if (rotationAngle < rotationThreshold && distance < translationThreshold) {
//                 loopClosures[]
//             }
//         }
//     }
// }
//TP FP FN
vector<float> caculate_TP_TF(map<pair<int, int>, int>& bow_map, map<int, vector<pair<int, int>>>& truth_map, vector<int> F_loop) {
    int TP = 0;
    int FP = 0;
    int FN = 0;
    vector<float> temp;
    map<int,int> temp_match;
    //set<size_t> processed_key_bows; // 用于跟踪已经处理过的 key_bow
    for (auto it_bow = bow_map.begin(); it_bow != bow_map.end(); it_bow++) {
        size_t loop_bow = (*it_bow).first.second;
        size_t key_bow = (*it_bow).second;
        //bool found = false;
        // 检查当前的 key_bow 是否已经处理过，如果已经处理过，则跳过
        // if (processed_key_bows.count(key_bow) > 0) {
        //     continue;
        // }
        auto key = truth_map.find(key_bow);
        if (key != truth_map.end()) {
            for (const auto& it_item : key -> second) {
                if (loop_bow == it_item.second) {
                    TP++;
                    break;
                }
            }      
        }
        // for (const auto& it : truth_map) {
        //     int truth_frame_id = it.first; // 获取闭环帧的ID
        //     for(const auto& pair : it.second) {
        //         if (pair.second == key_bow) {
        //             found = true;
        //             temp_match[truth_frame_id] = key_bow; // 将找到的匹配关键帧的ID存储起来
        //         }
        //     }
        //     if (found) {
        //         TP++;
        //         break;
        //     }
        // }
        // 将当前 key_bow 标记为已处理
        //processed_key_bows.insert(key_bow);
    }

    double F1;
    cout << "truth_loop: " << truth_map.size() << endl;
    FP = bow3d_Map.size() - TP;
    FN = truth_map.size() - TP;
    cout << "TP: " << TP << endl;
    cout << "bow3d_loop: " << bow3d_Map.size() << endl;
    // cout << "FP: " << FP << endl;
    // cout << "FN: " << FN << endl;
    float pre = 0.0;
    float rec = 0.0;
    if (bow3d_Map.size() != 0) {
        pre = static_cast<float>(TP) / bow3d_Map.size();
    }
    if (truth_map.size() != 0) {
        rec = static_cast<float>(TP) / truth_map.size();
    }
    F1 = 2 * (pre * rec) / (pre + rec);
    temp.push_back(pre);
    temp.push_back(rec);
    temp.push_back(F1);
    return temp;
}
// 写入姿态数据到 .txt 文件
void writePosesToFile(const string& filename, const vector<pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses) {
    ofstream outfile(filename, ios::app); // 打开文件以追加的方式写入
    if (!outfile.is_open()) {
        cerr << "Error: Failed to open the file for writing." << endl;
        return;
    }

    for (const auto& pose : poses) {
        const auto& translation = pose.second;
        outfile << fixed << setprecision(6) << translation.x() << " " << translation.y() << " " << translation.z() << endl;
    }

    outfile.close();
}
// 将 bow_map 中对应的姿态数据写入到同一个 .txt 文件中
void writeBowMapPosesToFile(const string& filename, const map<pair<int, int>, int>& bow_map, const vector<pair<Eigen::Quaterniond, Eigen::Vector3d>>& poses) {
    ofstream outfile(filename);
    if (!outfile.is_open()) {
        cerr << "Error: Failed to open the file for writing." << endl;
        return;
    }

    for (const auto& entry : bow_map) {
        //int frame_id = entry.first.second;
        int frame_id = entry.first.first;
        if (frame_id >= 0 && frame_id < poses.size()) {
            const auto& translation = poses[frame_id].second;
            outfile << fixed << setprecision(6) << translation.x() << " " << translation.y() << " " << translation.z() << endl;
        }
    }

    outfile.close();
}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "BoW3D");
    ros::NodeHandle nh;  

    /*Please replace the dataset folder path with the path in your computer. KITTI's 00, 02, 05, 06, 07, 08 have loops*/
    string dataset_folder;
    string filename = "/home/zhihui/data/odom_evo/tum_02_gt.txt";//jia
    dataset_folder = "/home/zhihui/data/KITTI/dataset/sequences/02/velodyne/"; //The last '/' should be added
    vector<pair<Eigen::Quaterniond, Eigen::Vector3d>> poses = readTUMFile(filename);
    
    // 进行关键帧检测
    double rotationThreshold = 10.0;  // 设定旋转角度阈值为10度
    double translationThreshold = 1.0; // 设定距离阈值为3单位
    double loop_thres = 3.0; // 设定距离阈值为3单位
    vector<KeyFrame> mkeyFrames;
    dynamicKeyFrameDetection(poses, mkeyFrames, rotationThreshold, translationThreshold);
    auto ground_loop = findLoopClosures(mkeyFrames, loop_thres);

    // 写入所有关键帧的姿态数据到同一个 .txt 文件
    ofstream ground_loop_poses_ofs("/home/zhihui/data/odom_evo/ground_loop_poses.txt");
    ofstream ground_loop_id_ofs("/home/zhihui/data/odom_evo/ground_loop_id.txt");
    int truth_loop_cnt = 0;
    for (int i = 0; i < ground_loop.size(); ++i) {
        if(!ground_loop[i].empty()){
            auto& translation = mkeyFrames[i].translation_;
            ground_loop_poses_ofs << fixed << setprecision(6) << translation.x() << " " << translation.y() << " " << translation.z() << endl;
            truth_loop_cnt++;
        }
        for(auto j : ground_loop[i]){
            ground_loop_id_ofs << j << " ";
        }
        ground_loop_id_ofs << endl;
    }
    ground_loop_poses_ofs.close();
    ground_loop_id_ofs.close();
    
    //
    BoW3D::LinK3D_Extractor* pLinK3dExtractor = new BoW3D::LinK3D_Extractor(nScans, scanPeriod, minimumRange, distanceTh, matchTh); 
    BoW3D::BoW3D* pBoW3D = new BoW3D::BoW3D(pLinK3dExtractor, thr, thf, num_add_retrieve_features);

    int T = truth_loop_cnt;
    int F = ground_loop.size() - T;
    int P = 0;
    int TP = 0;
    vector<double> times_Int;
    ofstream bow3d_loop_id_ofs("/home/zhihui/data/odom_evo/bow3d_loop_id.txt");
    ofstream bow3d_loop_poses_ofs("/home/zhihui/data/odom_evo/bow3d_loop_poses.txt");
    ofstream missed_loop_id_ofs("/home/zhihui/data/odom_evo/missed_loop_id.txt");
    ofstream missed_loop_poses_ofs("/home/zhihui/data/odom_evo/missed_loop_poses.txt");
    ofstream error_loop_id_ofs("/home/zhihui/data/odom_evo/error_loop_id.txt");
    ofstream bow_truth_poses_ofs("/home/zhihui/data/odom_evo/bow_truth_poses.txt");

    for (int i = 0; i < mkeyFrames.size(); ++i) {
    // for (int i = 0; i < 1600; ++i) {
        const auto& key_frame = mkeyFrames[i];

        std::stringstream lidar_data_path;
        lidar_data_path << dataset_folder << std::setfill('0') << std::setw(6) << key_frame.BinID_ << ".bin";
        vector<float> lidar_data = read_lidar_data(lidar_data_path.str());
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for(std::size_t i = 0; i < lidar_data.size(); i += 4){            
            pcl::PointXYZ point;
            point.x = lidar_data[i];
            point.y = lidar_data[i + 1];
            point.z = lidar_data[i + 2];
            current_cloud->push_back(point);
        }

        //将当前帧的描述子  关键点算出来 会更新他的帧的ID
        Frame* pCurrentFrame = new Frame(pLinK3dExtractor, current_cloud); 
        //当前帧的 ID 小于 2，那么就会执行一些更新操作
        //此时应该是系统刚启动
        if(pCurrentFrame->mnId < 2)
        {
            pBoW3D->update(pCurrentFrame);
        }
        //启动了之后的检测
        else
        {                
            int loopFrameId = -1;
            Eigen::Matrix3d loopRelR;
            Eigen::Vector3d loopRelt;

            clock_t start, end;
            double time;       
            start = clock();
            //回环检测
            pBoW3D->retrieve(pCurrentFrame, loopFrameId, loopRelR, loopRelt); 

            end = clock();
            time = ((double) (end - start)) / CLOCKS_PER_SEC;
            
            pBoW3D->update(pCurrentFrame);               

            if(loopFrameId == -1)
            {
                cout << "-------------------------" << endl;
                cout << "Detection Time: " << time << "s" << endl;
                times_Int.push_back(time);
                cout << "Frame" << pCurrentFrame->mnId << " Has No Loop..." << endl;
                if (!ground_loop[i].empty()) {
                    missed_loop_id_ofs << pCurrentFrame->mnId << " ";
                    auto& miss_translation = mkeyFrames[i].translation_;
                    missed_loop_poses_ofs << fixed << setprecision(6) << miss_translation.x() << " " << miss_translation.y() << " " << miss_translation.z() << endl;
                }
            }
            else
            {
                LoopIndex++;
                cout << "--------------------------------------" << endl;
                cout << "Detection Time: " << time << "s" << endl;
                cout << "Frame" << pCurrentFrame->mnId << " Has Loop Frame" << loopFrameId << endl;
                times_Int.push_back(time);
                cout << "Loop Relative R: " << endl;
                cout << loopRelR << endl;
                                
                cout << "Loop Relative t: " << endl;                
                cout << "   " << loopRelt.x() << " " << loopRelt.y() << " " << loopRelt.z() << endl;
                bow3d_Map[{pCurrentFrame->mnId,loopFrameId}] = pCurrentFrame->mnId;

                bow3d_loop_id_ofs << loopFrameId << " ";
                auto& bow3d_translation = mkeyFrames[i].translation_;
                bow3d_loop_poses_ofs << fixed << setprecision(6) << bow3d_translation.x() << " " << bow3d_translation.y() << " " << bow3d_translation.z() << endl;
                P++;
                if(ground_loop[i].find(loopFrameId) != ground_loop[i].end()){
                    TP++;
                    auto& bow_truth_translation = mkeyFrames[i].translation_;
                    bow_truth_poses_ofs << fixed << setprecision(6) << bow_truth_translation.x() << " " << bow_truth_translation.y() << " " << bow_truth_translation.z() << endl;
                }
                if(ground_loop[i].find(loopFrameId) == ground_loop[i].end()){
                    error_loop_id_ofs << i << " " << loopFrameId << endl;
                }
            }
        }
        missed_loop_id_ofs << std::endl;
        bow3d_loop_id_ofs << std::endl;
    }
    error_loop_id_ofs.close();
    missed_loop_id_ofs.close();
    bow3d_loop_id_ofs.close();
    missed_loop_poses_ofs.close();
    bow_truth_poses_ofs.close();
    

    double precision = TP / (double)P;
    double recall = TP / (double)T;
    double average_time = 0;
    double num_time = 0;
    for (int i = 0; i < times_Int.size(); ++i) {
        num_time += times_Int[i];
    }
    average_time = num_time / times_Int.size();
    sort(times_Int.begin(),times_Int.end());
    std::cout << "T F: " << T << ", " << ground_loop.size() - T << std::endl;
    std::cout << "P N: " << P << ", " << ground_loop.size() - P << std::endl;
    cout << "TP: " << TP << endl;
    std::cout << "Precision: " << precision << std::endl;
    std::cout << "Recall: " << recall << std::endl;
    std::cout << "F1: " << (precision * recall) * 2.0 / (precision + recall)<< std::endl;
    std::cout << "num_time: " << num_time << endl;
    std::cout << "average_time: " << average_time << endl;
    std::cout << "time_max: " << times_Int[times_Int.size() - 1] << endl;

    return 0;
}

