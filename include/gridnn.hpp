/**
 * @ref 大部分内容参考高翔博士的<<自动驾驶与机器人中的SLAM技术>>.
 * https://github.com/gaoxiang12/slam_in_autonomous_driving.git
 */

#ifndef GRIDNN_HPP
#define GRIDNN_HPP

#include <unordered_map>
#include <glog/logging.h>
#include <bfnn.hpp>


using Vec3i = Eigen::Vector3i;
using PointType = pcl::PointXYZI;
using CloudType = pcl::PointCloud<PointType>;
using CloudPtr = CloudType::Ptr;


/**
 * @brief 定义向量哈希的哈希映射关系:向量的每个分量与分别与不同的大质数相乘再取异或,最后对结果除个大整数取模.
 *        哈希函数的运算结果并不是直接决定key对应的value,而是用于定位key-value对在哈希表中的存储位置.
 *        具体来说,哈希函数的作用是将key映射为一个整数值(哈希值),这个整数值进一步被用来定位哈希表中的存储桶（bucket）。
 */
struct hash_vec
{
    inline size_t operator()(const Eigen::Matrix<int, 3, 1>& vec) const;
};


/**
 * @brief 1.定义转化函数,把点云中的点转化到栅格坐标中,做法是:(point * resolution_inv).cast<int>
 *        2.对点云中的每个点,调用步骤1中的函数,把栅格坐标作为Key,点在点云中的索引作为Value,哈希函数采用
 *          向量哈希,存储到std::unordered_map中.
 *        3.根据用户需要的近邻关系(0、6、14近邻),从原始点云中取出
 */
class GridNN
{
public:
    // 哈希表的键值类型,向量哈希的键应该是整型的向量
    using KeyType = Eigen::Matrix<int, 3, 1>;
    enum class NearbyType{
        CENTER = 0,
        NEARBY6,
        NEARBY14
    };

private:
    // 栅格分辨率,单位为m
    float resolution_ = 0.2;
    // 分辨率倒数,方便求某个点在栅格坐标系下的位置
    float resolution_inv_ = 1.0 / resolution_;
    // 存放某个点的近邻栅格或体素信息
    std::vector<KeyType> nearby_grids_;
    NearbyType nearby_type_ = NearbyType::NEARBY6;

    // 存放目标点云
    CloudPtr cloud_;

    /**
     * @brief 存储栅格坐标-点在点云中的索引的哈希表
     *        三个参数分别是键类型、值类型、哈希函数.
     *        具体来说,这里的Key应该是整数栅格坐标,Value应该是点在点云中的索引.
     */
    std::unordered_map <KeyType, std::vector<size_t>, hash_vec> grids_; 

public:

    explicit GridNN(float resolution = 0.1, NearbyType nearby_type = NearbyType::NEARBY6)
        :resolution_(resolution), nearby_type_(nearby_type){
        resolution_inv_ = 1.0 / resolution;
        if(nearby_type_ != NearbyType::NEARBY6 ||
           nearby_type_ != NearbyType::NEARBY14 ||
           nearby_type_ != NearbyType::CENTER){
            LOG(ERROR) << "三维栅格只能使用0、6、14近邻,将使用6近邻.";
            nearby_type_ = NearbyType::NEARBY6;
        }
        GenerateNearbyGrids();
    };

    /**
     * @brief 将空间坐标转换为栅格坐标
     * @param point 待转换的扫描点坐标、
     * @return 该扫描点的栅格坐标,将作为哈希表的Key值
     */
    Eigen::Matrix<int, 3, 1> pos2grid(const Eigen::Matrix<float, 3, 1>& point);
    void GenerateHashTable(const CloudPtr& ptr);
    void GenerateNearbyGrids();
    bool GetClosetPoint(const PointType& point, PointType& closet_point, int& idx);
    bool GetClosetCloudMt(const CloudPtr cloud_query,
                          std::vector<std::pair<size_t, size_t>>& matches);

};

// 把点云中的所有点的空间坐标转换为栅格坐标,栅格坐标就是哈希表的Key.
void GridNN::GenerateHashTable(const CloudPtr& cloud){
    std::vector<size_t> index(cloud->size());
    std::for_each(index.begin(), index.end(), [idx = 0](auto& i) mutable {i = idx++;});

    std::for_each(index.begin(), index.end(), [&cloud, this](auto& idx){
        PointType pcl_point = cloud -> points[idx];
        KeyType Key = pos2grid(pcl_point.getVector3fMap());

        // 如果哈希表中Key还未被创建,则创建该Key并插入点的索引.
        if(grids_.find(Key) == grids_.end()){
            grids_.insert({Key, {idx}});
        }
        // 如果哈希表中Key已经存在,则向该Key对应的Value中插入索引idx.
        // 注意,实际上Key-Value是一一对应的,但这里的Value是一个Vector,所以可以向Value的末尾插入元素.
        else{
            // 按照前面对grids_的定义,这里grids_[Key]实际上就是Key对应的向量
            grids_[Key].emplace_back(idx);
        }
    });
    cloud_ = cloud;
    LOG(INFO) << "grids size:" << grids_.size();
}


bool GridNN::GetClosetPoint(const PointType& point, PointType& closet_point, int& idx){
    // 存放需要查找的栅格中所有点的索引
    std::vector<size_t>idx_to_check;
    // 当前查找点的Key
    auto Key = pos2grid(point.getVector3fMap());

    if (grids_.find(Key) != grids_.end()){
        // 对于每一个候选栅格,把它们在哈希表中对应的索引存入到idx_to_check中
        std::for_each(nearby_grids_.begin(), nearby_grids_.end(), [&Key, &idx_to_check, this](const Vec3i& idx_delta){
            auto dKey = Key + idx_delta;
            auto iter = grids_.find(dKey);
            if (iter != grids_.end()){
                idx_to_check.insert(idx_to_check.end(), iter->second.begin(), iter->second.end());}});
    }

    // 根据近邻栅格和点云构建暴力搜索的目标点云
    CloudPtr nearby_cloud(new CloudType);

    // 用来存待搜索的点云nearby_cloud中的每一个点,在原本的完整点云中的索引
    std::vector<size_t> nearby_index;
    for(const auto& idx : idx_to_check){
        nearby_cloud -> points.emplace_back(cloud_ -> points[idx]);
        nearby_index.emplace_back(idx);
    }

    if(nearby_cloud -> empty()){return false;}

    // 这里得到的是最近邻点在构建的nearby_cloud中的索引
    size_t closet_index = bfnn::bfnn_point(nearby_cloud, point.getVector3fMap());

    // 根据上一步的索引查找最近邻点在完整点云中的索引
    idx = nearby_index[closet_index];
    closet_point = cloud_ -> points[idx];
    return true;
}


bool GridNN::GetClosetCloudMt(const CloudPtr cloud_query,
                              std::vector<std::pair<size_t, size_t>>& matches)
{
    matches.resize(cloud_query -> size());
    std::vector<size_t> index(cloud_query -> size());
    std::for_each(index.begin(), index.end(), [i = 0](auto& idx)mutable{idx = i++;});
    
    PointType closet_point;
    size_t closet_idx;
    std::for_each(std::execution::par_unseq, index.begin(), index.end(),
                 [&matches, &cloud_query, this](auto& idx)
    {
        if(GetClosetPoint(cloud_query -> points[idx], closet_point, closet_idx)){
            matches.emplace_back(idx, closet_idx);
        }
    });
    return true;
}


void GridNN::GenerateNearbyGrids(){
    if (this -> nearby_type_ == NearbyType::CENTER){
        this -> nearby_grids_ = {Vec3i(0, 0, 0)};
    }
    else if (this -> nearby_type_ == NearbyType::NEARBY6){
        this -> nearby_grids_ = {Vec3i(0, 0, 0), Vec3i(0, 0, 1), Vec3i(0, 1, 0),
                                 Vec3i(1, 0, 0), Vec3i(0, 0, -1), Vec3i(0, -1, 0), 
                                 Vec3i(-1, 0, 0)};
    }
    else if (this -> nearby_type_ == NearbyType::NEARBY14){
        this -> nearby_grids_ = {Vec3i(0, 0, 0), Vec3i(0, 0, 1), Vec3i(0, 1, 0),
                                 Vec3i(1, 0, 0), Vec3i(0, 0, -1), Vec3i(0, -1, 0), 
                                 Vec3i(-1, 0, 0), Vec3i(-1, 0, 0), Vec3i(1, 1, 1),
                                 Vec3i(1, 1, -1), Vec3i(1, -1, 1), Vec3i(-1, 1, 1),
                                 Vec3i(1, -1, -1), Vec3i(-1, -1, 1), Vec3i(-1, 1, -1)};
    }
    else{
        LOG(ERROR) << "近邻栅格参数错误.";
    }
}

Eigen::Matrix<int, 3, 1> GridNN::pos2grid(const Eigen::Matrix<float, 3, 1>& point){
    return (point * this -> resolution_inv_).cast<int>();
}

inline size_t hash_vec::operator()(const Eigen::Matrix<int, 3, 1>& vec) const{
    return (vec[0, 0] * 73856093) ^ (vec[1, 0] * 471943) ^ (vec[2, 0] * 83492791) % 10000000;
}
#endif