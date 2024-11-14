#ifndef __MODEL_H__
#define __MODEL_H__

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <dlfcn.h>
#include <yaml-cpp/yaml.h>
#include <cuda_runtime.h>
#include "bevfusion/bevfusion.hpp"
#include "common/check.hpp"
#include "common/tensor.hpp"
#include "common/timer.hpp"


class Model {
public:

  std::shared_ptr<bevfusion::Core> model_;
  cudaStream_t stream;

  bool use_camera_;
  std::vector<std::string> class_names_;
  size_t num_camera_;
  std::vector<std::string> cam_names_;
  size_t image_height_;
  size_t image_width_;
  size_t image_input_height_;
  size_t image_input_width_;
  size_t image_feat_height_;
  size_t image_feat_width_;
  float resize_lim_;
  std::vector<float> mean_;
  std::vector<float> std_;
  std::vector<float> pc_range_;
  std::vector<float> voxel_size_;
  size_t max_points_per_voxel_;
  size_t max_points_;
  size_t max_voxels_;
  size_t num_feature_;
  std::string model_point_encoder_;
  std::string model_lidar_feature_encoder_;
  std::string model_camera_backbone_;
  std::string model_camera_vtransform_;
  std::string model_fuser_;
  std::string model_head_;
  std::string precision_;
  std::vector<float> xbound_;
  std::vector<float> ybound_;
  std::vector<float> zbound_;
  std::vector<float> dbound_;
  std::vector<size_t> geometry_dim_;
  size_t out_size_factor_;
  std::vector<float> post_center_range_;
  float confidence_threshold_;
  std::vector<float> cam_intrinsics_;
  std::vector<float> camera2lidars_;
  std::vector<float> lidar2cameras_;
  std::vector<float> lidar2images_;
  std::vector<float> img_aug_matrixs_;

  std::shared_ptr<bevfusion::Core> create_core();
 
  Model(const std::string &config_file);
  ~Model(void);

  void doInfer(const unsigned char** images_data, 
              nv::Tensor& points,
              std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes);

  void initParams(const std::string &config_file);
};


#endif
