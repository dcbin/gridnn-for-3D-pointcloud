#ifndef BFNN_HPP
#define BFNN_HPP
#include <pcl-1.10/pcl/point_cloud.h>
#include <pcl-1.10/pcl/point_types.h>
#include <pcl-1.10/pcl/impl/point_types.hpp>
#include <eigen3/Eigen/Dense>
#include <algorithm>
#include <vector>
#include <execution>
#include <iostream>


namespace bfnn{

using PointType = pcl::PointXYZI;
using PointCloudType = pcl::PointCloud<PointType>;
using CloudPtr = PointCloudType::Ptr;
using Vec3f = Eigen::Vector3f;


/*******************************************/
/*****************函数声明*******************/
/*******************************************/

/**
 * @brief 单个点的暴力匹配算法,寻找目标点距离给定点云中的最近点的索引
 * @param cloud PointCloudType::Ptr;目标点云的指针
 * @param point Eigen::Vector3f;单个点云
 * @return 返回在cloud中与point欧式距离最小的点的索引,是一个int值
 */
int bfnn_point(CloudPtr cloud, const Vec3f& point);

/**
 * @brief 找到目标点在点云数据中的的k个近邻点的索引
 * @param cloud1
 * @param cloud2
 * @param k
 * @return std::vector<int>;k个最近点的索引
 */
std::vector<int> bfnn_point_k(CloudPtr cloud, const Vec3f& point, const int& k);

/**
 * @brief 多线程点云匹配,寻找cloud2中的每个点在cloud1中的最近邻点.
 * 需要注意的是:
 * 1.cloud2中的每个点都应该在cloud1中存在一个最近邻点.
 * 2.cloud1和cloud2的size不一定相同(几乎不可能相同).
 * 3.cloud2中每个点只存在一个最近邻点,
 *   但cloud1中每个点可能是cloud2中许多个点云的最近点,
 *   也可能不是任何点的最近点.
 * @param cloud1 目标点云
 * @param cloud2 源点云
 * @param matches std::vector <std::pair <size_t, size_t>>.匹配结果,存储在vector中.每个元素包含两个整型数据,
 *                分别表示两个匹配点在各自点云中的索引.std::pair中的第二个元素表示cloud2每个点的索引,
 *                第一个元素表示在cloud1中的最近点索引
 */
void bfnn_cloud_mt(CloudPtr cloud1, CloudPtr cloud2,
                   std::vector <std::pair <size_t, size_t>>& matches);

/**
 * @brief 寻找点云cloud2中的每个点在点云cloud1中的k个最近邻点,匹配结果存入matches中
 * @param cloud1 CloudPtr
 * @param cloud2 CloudPtr
 * @param matches std::vector<std::pair<size_t, size_t>>& matches
 * @param k int
 */
void bfnn_cloud_k_mt(CloudPtr cloud1, CloudPtr cloud2,
                     std::vector<std::pair<size_t, size_t>>& matches, const int& k);



/***********************************/
/*************函数实现***************/
/***********************************/


int bfnn_point(CloudPtr cloud, const Vec3f& point)
{
    return std::min_element(cloud->points.begin(), cloud->points.end(),
                            [&point](const PointType& pt1, const PointType& pt2) -> bool
                            {
                                return (pt1.getVector3fMap() - point).squaredNorm() <
                                       (pt2.getVector3fMap() - point).squaredNorm();
                            }) - cloud->points.begin();
}


std::vector<int> bfnn_point_k(CloudPtr cloud, const Vec3f& point, const int& k)
{
    // 存储点云中每个元素的索引以及与目标点的欧式距离
    struct IndexAndDis2
    {
        IndexAndDis2(){}
        IndexAndDis2(int index, double dis2):index_(index), Dis2_(dis2){}
        int index_;
        double Dis2_;
    };
    std::vector<IndexAndDis2> index_and_dis2;
    for(int i = 0; i < cloud->size(); ++i)
    {
        index_and_dis2[i] = {i, (cloud->points[i].getVector3fMap() - point).squaredNorm()};
    }

    std::sort(index_and_dis2.begin(), index_and_dis2.end(),
        [](const IndexAndDis2& var1, const IndexAndDis2& var2) -> bool
        {return var1.Dis2_ < var2.Dis2_;});
    
    std::vector<int> index;
    std::transform(index_and_dis2.begin(), index_and_dis2.begin() + k,
                    std::back_inserter(index), [](const auto& var) -> auto
                    {return var.index_;});
    return index;
}

/**
 * @brief 多线程点云匹配,寻找cloud2中的每个点在cloud1中的最近邻点.
 * @param cloud1 目标点云
 * @param cloud2 源点云
 * @param matches std::vector <std::pair <size_t, size_t>>.匹配结果,存储在vector中.每个元素包含两个整型数据,
 *                分别表示两个匹配点在各自点云中的索引.std::pair中的第二个元素表示cloud2每个点的索引,
 *                第一个元素表示在cloud1中的最近点索引
 */
void bfnn_cloud_mt(CloudPtr cloud1, CloudPtr cloud2,
                   std::vector <std::pair <size_t, size_t>>& matches)
{
    std::vector<size_t> index(cloud2 -> size());
    std::for_each(index.begin(), index.end(), [idx = 0](auto& i) mutable {i = idx++;});

    // 确保存放匹配结果的向量的尺度与cloud2的点云数量一致
    matches.resize(index.size());
    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](size_t idx){
        matches[idx].second = idx;
        matches[idx].first = bfnn_point(cloud1, cloud2 -> points[idx].getVector3fMap());
        std::cout << "完成1次计算." << std::endl;
    });
}


void bfnn_cloud_k_mt(CloudPtr cloud1, CloudPtr cloud2,
                     std::vector<std::pair<size_t, size_t>>& matches, const int& k)
{
    std::vector<size_t> index(cloud2 -> size());
    matches.resize(index.size() * k);
    std::for_each(index.begin(), index.end(), [idx = 0](auto& i) mutable {i = idx++;});

    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](size_t idx){
        auto vec = bfnn_point_k(cloud1, cloud2 -> points[idx].getVector3fMap(), k);
        for(int i = 0; i < vec.size(); ++i)
        {
            matches[k * idx + i].first = vec[i];
            matches[k * idx + i].second = idx;
        }
    });
}
} //namespace bfnn
#endif