#include <ros/ros.h>
#include <string>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>
#include <eigen3/Eigen/Dense>
#include <pcl/filters/extract_indices.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include "Scancontext.h"
#include <map>

using namespace std;

map<pair<int, int>, int> scan_Map;
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
void PointXYZ2PointXYZI(pcl::PointCloud<pcl::PointXYZ>::Ptr inCloud, pcl::PointCloud<pcl::PointXYZI>::Ptr outCloud) {
    for(auto p : inCloud->points){
        pcl::PointXYZI point;
        point.x = p.x;
        point.y = p.y;
        point.z = p.z;
        point.intensity = 0;
        outCloud->push_back(point);
    }

}
int main(int argc, char** argv)
{
    ros::init(argc, argv, "BoW3D");
    ros::NodeHandle nh;  

    /*Please replace the dataset folder path with the path in your computer. KITTI's 00, 02, 05, 06, 07, 08 have loops*/
    string dataset_folder;
    string filename = "/home/zhihui/data/odom_evo/tum_08_gt.txt";//jia
    dataset_folder = "/home/zhihui/data/KITTI/dataset/sequences/08/velodyne/"; //The last '/' should be added
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
    // BoW3D::LinK3D_Extractor* pLinK3dExtractor = new BoW3D::LinK3D_Extractor(nScans, scanPeriod, minimumRange, distanceTh, matchTh); 
    // BoW3D::BoW3D* pBoW3D = new BoW3D::BoW3D(pLinK3dExtractor, thr, thf, num_add_retrieve_features);
    SCManager scManager;
    int T = truth_loop_cnt;
    int F = ground_loop.size() - T;
    int P = 0;
    int TP = 0;
    vector<double> times_Int;
    ofstream scan_loop_id_ofs("/home/zhihui/data/odom_evo/scan_loop_id_ofs.txt");
    ofstream scan_loop_poses_ofs("/home/zhihui/data/odom_evo/scan_loop_poses_ofs.txt");
    ofstream missed_loop_id_ofs("/home/zhihui/data/odom_evo/missed_loop_id.txt");
    ofstream missed_loop_poses_ofs("/home/zhihui/data/odom_evo/missed_loop_poses.txt");
    ofstream error_loop_id_ofs("/home/zhihui/data/odom_evo/error_loop_id.txt");
    ofstream scan_truth_poses_ofs("/home/zhihui/data/odom_evo/scan_truth_poses_ofs.txt");

    for (int i = 0; i < mkeyFrames.size(); ++i) {
    // for (int i = 0; i < 1600; ++i) {
        const auto& key_frame = mkeyFrames[i];

        std::stringstream lidar_data_path;
        lidar_data_path << dataset_folder << std::setfill('0') << std::setw(6) << key_frame.BinID_ << ".bin";
        vector<float> lidar_data = read_lidar_data(lidar_data_path.str());
        pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for(std::size_t i = 0; i < lidar_data.size(); i += 4){            
            pcl::PointXYZ point;
            point.x = lidar_data[i];
            point.y = lidar_data[i + 1];
            point.z = lidar_data[i + 2];
            raw_cloud->push_back(point);
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(new pcl::PointCloud<pcl::PointXYZI>());
        PointXYZ2PointXYZI(raw_cloud,current_cloud);  
        scManager.makeAndSaveScancontextAndKeys(*current_cloud);
        int SCclosestHistoryFrameID = -1; // init with -1
        int pCurrentFrameID = i;

        clock_t start, end;
        double time;       
        start = clock();
        //回环检测
        auto detectResult = scManager.detectLoopClosureID(); // first: nn index, second: yaw diff 
        SCclosestHistoryFrameID = detectResult.first;
        end = clock();
        time = ((double) (end - start)) / CLOCKS_PER_SEC;            

        if(SCclosestHistoryFrameID == -1)
        {
            cout << "-------------------------" << endl;
            cout << "Detection Time: " << time << "s" << endl;
            times_Int.push_back(time);
            cout << "Frame" << pCurrentFrameID << " Has No Loop..." << endl;
            if (!ground_loop[i].empty()) {
                missed_loop_id_ofs << pCurrentFrameID << " ";
                auto& miss_translation = mkeyFrames[i].translation_;
                missed_loop_poses_ofs << fixed << setprecision(6) << miss_translation.x() << " " << miss_translation.y() << " " << miss_translation.z() << endl;
            }
        }
        else
        {
            cout << "--------------------------------------" << endl;
            cout << "Detection Time: " << time << "s" << endl;
            cout << "Frame" << pCurrentFrameID << " Has Loop Frame" << SCclosestHistoryFrameID << endl;
            times_Int.push_back(time);                                         
            scan_Map[{pCurrentFrameID,SCclosestHistoryFrameID}] = pCurrentFrameID;

            scan_loop_id_ofs << SCclosestHistoryFrameID << " ";
            auto& scan_translation = mkeyFrames[i].translation_;
            scan_loop_poses_ofs << fixed << setprecision(6) << scan_translation.x() << " " << scan_translation.y() << " " << scan_translation.z() << endl;
            P++;
            if(ground_loop[i].find(SCclosestHistoryFrameID) != ground_loop[i].end()){
                TP++;
                auto& bow_truth_translation = mkeyFrames[i].translation_;
                scan_truth_poses_ofs << fixed << setprecision(6) << bow_truth_translation.x() << " " << bow_truth_translation.y() << " " << bow_truth_translation.z() << endl;
            }
            if(ground_loop[i].find(SCclosestHistoryFrameID) == ground_loop[i].end()){
                error_loop_id_ofs << i << " " << SCclosestHistoryFrameID << endl;
            }
        }
    
    missed_loop_id_ofs << std::endl;
    scan_loop_id_ofs << std::endl;
    }
    error_loop_id_ofs.close();
    missed_loop_id_ofs.close();
    scan_loop_id_ofs.close();
    missed_loop_poses_ofs.close();
    scan_truth_poses_ofs.close();
    

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