cmake_minimum_required(VERSION 3.14)
project(bevfusion)

find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()
add_definitions(-DROS_DISTRO_GALACTIC)

# set environment variables
set(TensorRT_Lib /usr/local/include/TensorRT-8.5.1.7/lib)
set(TensorRT_Inc /usr/local/include/TensorRT-8.5.1.7/include)
set(TensorRT_Bin /usr/local/include/TensorRT-8.5.1.7/bin)
set(CUDA_Lib /usr/local/cuda/lib64)
set(CUDA_Inc /usr/local/cuda/include)
set(CUDA_Bin /usr/local/cuda/bin)
set(CUDA_HOME /usr/local/cuda)
set(CUDNN_Lib /usr/lib/aarch64-linux-gnu)
set(arch ${CMAKE_HOST_SYSTEM_PROCESSOR})

find_package(CUDA REQUIRED)
find_package(Protobuf REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)

# For cuda-12
# set(CMAKE_CXX_FLAGS_RELEASE "-std=c++17 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
# set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++17 -O0 -g -DENABLE_TEXT_BACKEND_STB")
# set(spconv_root bevfusion/lib/libraries/3DSparseConvolution/libspconv_cuda12)

# For cuda-11
# set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -Wextra -Wall -Wno-missing-field-initializers -Wno-deprecated-declarations -O3 -DENABLE_TEXT_BACKEND_STB")
# set(CMAKE_CXX_FLAGS_DEBUG   "-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB")
add_compile_options(-std=c++14 -O0 -g -DENABLE_TEXT_BACKEND_STB)
set(spconv_root bevfusion/lib/libraries/3DSparseConvolution/libspconv_cuda11)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
  -gencode arch=compute_80,code=compute_80
)


include_directories(
  ${Protobuf_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDA_Inc}
  ${TensorRT_Inc}
  ${Python_Inc}
  ${YAML_CPP_INCLUDE_DIRS}
  bevfusion/lib/dependencies/stb
  bevfusion/lib/dependencies/pybind11/include
  bevfusion/lib/libraries/cuOSD/src
  ${spconv_root}/include
  bevfusion/src/common
  bevfusion/src/
  include
)

link_directories(
  ${CUDA_Lib}
  ${TensorRT_Lib}
  ${spconv_root}/lib/${arch}
)

file(GLOB_RECURSE CORE_FILES
  bevfusion/src/bevfusion/*.cu 
  bevfusion/src/bevfusion/*.cpp
  bevfusion/src/onnx/*.cpp
  bevfusion/src/common/tensor.cu
  bevfusion/src/common/tensorrt.cpp
)

file(GLOB_RECURSE VISUALIZE_FILES 
  bevfusion/lib/libraries/cuOSD/src/*.cpp
  bevfusion/lib/libraries/cuOSD/src/*.cu
  bevfusion/src/common/visualize.cu
)

####################### libbevfusion_core.so ##########################
cuda_add_library(bevfusion_core SHARED
  ${CORE_FILES}
  ${VISUALIZE_FILES}
)

target_link_libraries(bevfusion_core
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libspconv.so
  ${Protobuf_LIBRARIES}
  libnvinfer_plugin.so
)
#######################################################################

# ###################### custom_layernorm.so ############################
# cuda_add_library(custom_layernorm SHARED
#   bevfusion/src/plugins/custom_layernorm.cu
# )

# target_link_libraries(custom_layernorm
#   libnvinfer.so
#   libnvinfer_plugin.so
# )
# #######################################################################

###################### bevfusion_model.so ############################
cuda_add_library(bevfusion_model SHARED
  src/model.cpp
)

target_link_libraries(bevfusion_model
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libnvinfer_plugin.so
  libspconv.so
  bevfusion_core
  ${Protobuf_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
)
#######################################################################


ament_auto_add_library(bevfusion_component SHARED
  src/model.cpp
  src/node.cpp
)

target_link_libraries(bevfusion_component
  libcudart.so
  libcublasLt.so
  libnvinfer.so
  libnvinfer_plugin.so
  libspconv.so
  bevfusion_core
  ${Protobuf_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  yaml-cpp
)

rclcpp_components_register_node(bevfusion_component
  PLUGIN "BEVFusionNode"
  EXECUTABLE bevfusion_node
)

ament_auto_package(INSTALL_TO_SHARE
  launch
  configs
  model
)
