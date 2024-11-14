#include <iostream>
#include "model.hpp"


std::shared_ptr<bevfusion::Core> Model::create_core() {

  bevfusion::CoreParameter param;
  param.use_camera_ = use_camera_;

  if (use_camera_) {
    bevfusion::camera::NormalizationParameter normalization;
    normalization.image_width = image_width_;
    normalization.image_height = image_height_;
    normalization.output_width = image_input_width_;
    normalization.output_height = image_input_height_;
    normalization.num_camera = num_camera_;
    normalization.resize_lim = resize_lim_;
    normalization.interpolation = bevfusion::camera::Interpolation::Bilinear;
    float mean[3] = {mean_[0], mean_[1], mean_[2]};
    float std[3] = {std_[0], std_[1], std_[2]};
    normalization.method = bevfusion::camera::NormMethod::mean_std(mean, std, 1 / 255.0f, 0.0f);

    bevfusion::camera::GeometryParameter geometry;
    geometry.xbound = nvtype::Float3(xbound_[0], xbound_[1], xbound_[2]);
    geometry.ybound = nvtype::Float3(ybound_[0], ybound_[1], ybound_[2]);
    geometry.zbound = nvtype::Float3(zbound_[0], zbound_[1], zbound_[2]);
    geometry.dbound = nvtype::Float3(dbound_[0], dbound_[1], dbound_[2]);
    geometry.image_width = image_input_width_;
    geometry.image_height = image_input_height_;
    geometry.feat_width = image_feat_width_;
    geometry.feat_height = image_feat_height_;
    geometry.num_camera = num_camera_;
    geometry.geometry_dim = nvtype::Int3(geometry_dim_[0], geometry_dim_[1], geometry_dim_[2]);

    param.camera_model = model_camera_backbone_;
    param.normalize = normalization;
    param.geometry = geometry;
    param.camera_vtransform = model_camera_vtransform_;
  }
  
  bevfusion::lidar::VoxelizationParameter voxelization;
  voxelization.min_range = nvtype::Float3(pc_range_[0], pc_range_[1], pc_range_[2]);
  voxelization.max_range = nvtype::Float3(pc_range_[3], pc_range_[4], pc_range_[5]);
  voxelization.voxel_size = nvtype::Float3(voxel_size_[0], voxel_size_[1], voxel_size_[2]);
  voxelization.grid_size =
      voxelization.compute_grid_size(voxelization.max_range, voxelization.min_range, voxelization.voxel_size);
  voxelization.max_voxels = max_voxels_;
  voxelization.max_points_per_voxel = max_points_per_voxel_;
  voxelization.max_points = max_points_;
  voxelization.num_feature = num_feature_;

  bevfusion::head::transbbox::TransBBoxParameter transbbox;
  transbbox.out_size_factor = out_size_factor_;
  transbbox.pc_range = {pc_range_[0], pc_range_[1]};
  transbbox.post_center_range_start = {post_center_range_[0], post_center_range_[1], post_center_range_[2]};
  transbbox.post_center_range_end = {post_center_range_[3], post_center_range_[4], post_center_range_[5]};
  transbbox.voxel_size = {voxel_size_[0], voxel_size_[1]};
  transbbox.model = model_head_;
  transbbox.confidence_threshold = confidence_threshold_;
  transbbox.sorted_bboxes = true;
  
  param.voxelization = voxelization;
  param.point_encoder = model_point_encoder_;
  param.lidar_feature_encoder = model_lidar_feature_encoder_;
  param.transfusion = model_fuser_;
  param.transbbox = transbbox;
  return bevfusion::create_core(param);
}


Model::Model(const std::string &config_file){
  initParams(config_file);

  // Init model
  dlopen("libcustom_layernorm.so", RTLD_NOW);

  model_ = create_core();
  if (model_ == nullptr) {
    printf("Core has been failed.\n");
    return;
  }

  cudaStreamCreate(&stream);
 
  model_->print();
  model_->set_timer(true); 
  if (use_camera_){
    model_->update(camera2lidars_.data(), cam_intrinsics_.data(), lidar2images_.data(), img_aug_matrixs_.data(), stream);
  }
}


Model::~Model(void){
  checkRuntime(cudaStreamDestroy(stream));
  model_.reset();
}


void Model::doInfer(const unsigned char** images_data, 
                    nv::Tensor& points,
                    std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes){
  //Inference
  bboxes = model_->forward(images_data, points.ptr<nvtype::half>(), points.size(0), stream);
}


void Model::initParams(const std::string &config_file){
  YAML::Node config = YAML::LoadFile(config_file);
  use_camera_ = config["model"]["use_camera"].as<bool>();
  class_names_ = config["model"]["class_names"].as<std::vector<std::string>>();
  model_point_encoder_ = config["model"]["point_encoder_file"].as<std::string>();
  model_lidar_feature_encoder_ = config["model"]["lidar_feature_encoder_file"].as<std::string>();
  model_fuser_ = config["model"]["fuser_file"].as<std::string>();
  model_head_ = config["model"]["head_file"].as<std::string>();
  precision_ = config["model"]["model_precision"].as<std::string>();
  geometry_dim_ = config["model"]["geometry_dim"].as<std::vector<size_t>>();
  out_size_factor_ = config["model"]["out_size_factor"].as<int>();
  post_center_range_ = config["model"]["post_center_range"].as<std::vector<float>>();
  confidence_threshold_ = config["model"]["confidence_threshold"].as<float>();

  pc_range_ = config["lidar"]["pc_range"].as<std::vector<float>>();
  voxel_size_ = config["lidar"]["voxel_size"].as<std::vector<float>>();
  max_points_per_voxel_ = config["lidar"]["max_points_per_voxel"].as<int>();
  max_points_ = config["lidar"]["max_points"].as<int>();
  max_voxels_ = config["lidar"]["max_voxels"].as<int>();
  num_feature_ = config["lidar"]["num_features"].as<int>();

  if (use_camera_){
    model_camera_backbone_ = config["model"]["camera_backbone_file"].as<std::string>();
    model_camera_vtransform_ = config["model"]["camera_vtranform_file"].as<std::string>();
    xbound_ = config["camera"]["xbound"].as<std::vector<float>>();
    ybound_ = config["camera"]["ybound"].as<std::vector<float>>();
    zbound_ = config["camera"]["zbound"].as<std::vector<float>>();
    dbound_ = config["camera"]["dbound"].as<std::vector<float>>();
    num_camera_ = config["camera"]["Ncams"].as<int>();
    cam_names_ = config["camera"]["cam_names"].as<std::vector<std::string>>();
    image_height_ = config["camera"]["image_size"][0].as<int>();
    image_width_ = config["camera"]["image_size"][1].as<int>();
    image_input_height_ = config["camera"]["input_size"][0].as<int>();
    image_input_width_ = config["camera"]["input_size"][1].as<int>();
    image_feat_height_ = config["model"]["feat_size"][0].as<int>();
    image_feat_width_ = config["model"]["feat_size"][1].as<int>();
    resize_lim_ = config["camera"]["resize_rate"].as<float>();
    mean_ = config["camera"]["mean"].as<std::vector<float>>();
    std_ = config["camera"]["std"].as<std::vector<float>>();

    cam_intrinsics_.clear();
    camera2lidars_.clear();
    lidar2cameras_.clear();
    lidar2images_.clear();
    img_aug_matrixs_.clear();
    for(std::string name : cam_names_){
      std::vector<float> cam_intrinsic = config["camera"][name]["cam_intrinsic"].as<std::vector<float>>();
      std::vector<float> camera2lidar = config["camera"][name]["camera2lidar"].as<std::vector<float>>();
      std::vector<float> lidar2camera = config["camera"][name]["lidar2camera"].as<std::vector<float>>();
      std::vector<float> lidar2image = config["camera"][name]["lidar2image"].as<std::vector<float>>();
      std::vector<float> img_aug_matrix = config["camera"][name]["img_aug_matrix"].as<std::vector<float>>();
      for(size_t i = 0; i < cam_intrinsic.size(); i++){
        cam_intrinsics_.push_back(cam_intrinsic[i]);
        camera2lidars_.push_back(camera2lidar[i]);
        lidar2cameras_.push_back(lidar2camera[i]);
        lidar2images_.push_back(lidar2image[i]);
        img_aug_matrixs_.push_back(img_aug_matrix[i]);
      }
    }
  }
}
