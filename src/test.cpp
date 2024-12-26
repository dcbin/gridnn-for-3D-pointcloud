#include "gridnn.hpp"
#include <pcl/io/pcd_io.h>
#include <chrono>

int main(int argc, char** argv)
{
    // 初始化Glog
    google::InitGoogleLogging(argv[0]);

    // 读取点云文件
    std::string first_path = "/home/dcbin/cpp_ws/gridnn/data/first.pcd";
    std::string second_path = "/home/dcbin/cpp_ws/gridnn/data/second.pcd";
    gridnn::CloudPtr cloud_query(new gridnn::CloudType);
    gridnn::CloudPtr cloud_soure(new gridnn::CloudType);
    pcl::io::loadPCDFile(first_path, *cloud_query);
    pcl::io::loadPCDFile(second_path, *cloud_soure);
    if (cloud_query -> empty() || cloud_soure -> empty())
    {
        LOG(ERROR) << "点云读取错误!未读取到点云!";
        return 0;
    }

    // 栅格大小取0.1m
    gridnn::GridNN gridnn_entity(0.1, gridnn::GridNN::NearbyType::NEARBY6);

    // 存放匹配结果
    std::vector<std::pair<size_t, size_t>> matches;
    gridnn_entity.GenerateHashTable(cloud_soure);
    auto t1 = std::chrono::high_resolution_clock::now();
    gridnn_entity.GetClosetCloudMt(cloud_query, matches);
    auto t2 = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() * 1000;
    LOG(ERROR) << "用时:" << total_time << "ms";
    return 0;
}