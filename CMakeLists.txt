# 最低cmake版本需求
cmake_minimum_required(VERSION 3.16)

project(gridnn_test)
set(CMAKE_CXX_STANDARD 17)

find_package(PCL 1.10 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED)

# 指定可执行文件
add_executable(gridnn_test src/test.cpp)

# 链接到库
target_link_libraries(gridnn_test ${PCL_LIBRARIES})
target_link_libraries(gridnn_test glog)
target_link_libraries(gridnn_test TBB::tbb)

# 包含头文件
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

# 因为0.4.0的glog没有办法用find_package找到，所以只能手动指定
include_directories(/usr/include/glog)
link_directories(/usr/lib/x86_64-linux-gnu)
