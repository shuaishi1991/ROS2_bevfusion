/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include "bevfusion.hpp"
#include <numeric>

#include "common/check.hpp"
#include "common/timer.hpp"
#include "common/dtype.hpp"
#include "common/tensor.hpp"

namespace bevfusion {

class CoreImplement : public Core {
 public:
  virtual ~CoreImplement() {
    if (lidar_points_device_) checkRuntime(cudaFree(lidar_points_device_));
    if (lidar_points_host_) checkRuntime(cudaFreeHost(lidar_points_host_));
  }

  bool init(const CoreParameter& param) {
    param_ = param;

    lidar_voxelization_ = lidar::create_voxelization(param_.voxelization);
    if (lidar_voxelization_ == nullptr) {
        printf("Failed to create lidar voxelization.\n");
        return false;
    }

    point_encoder_ = point_encoder::create_point_encoder(param_.point_encoder);
    if (point_encoder_ == nullptr) {
      printf("Failed to create point_encoder.\n");
      return false;
    }

    lidar_feature_encoder_ = lidar_feature_encoder::create_lidar_feature_encoder(param_.lidar_feature_encoder);
    if (lidar_feature_encoder_ == nullptr) {
      printf("Failed to create lidar_feature_encoder.\n");
      return false;
    }

    std::vector<int> pillar_scatter_output_shape = lidar_feature_encoder_->spatial_feature_shape();
    pillar_scatter_ = scatter::create_pillar_scatter(pillar_scatter_output_shape);
    if (point_encoder_ == nullptr) {
      printf("Failed to create pillar_scatter.\n");
      return false;
    }

    if(param_.use_camera_){
      camera_backbone_ = camera::create_backbone(param_.camera_model);
      if (camera_backbone_ == nullptr) {
        printf("Failed to create camera backbone.\n");
        return false;
      }

      camera_bevpool_ =
          camera::create_bevpool(camera_backbone_->camera_shape(), param_.geometry.geometry_dim.x, param_.geometry.geometry_dim.y);
      if (camera_bevpool_ == nullptr) {
        printf("Failed to create camera bevpool.\n");
        return false;
      }

      camera_vtransform_ = camera::create_vtransform(param_.camera_vtransform);
      if (camera_vtransform_ == nullptr) {
        printf("Failed to create camera vtransform.\n");
        return false;
      }

      normalizer_ = camera::create_normalization(param_.normalize);
      if (normalizer_ == nullptr) {
        printf("Failed to create normalizer.\n");
        return false;
      }

      camera_depth_ = camera::create_depth(param_.normalize.output_width, param_.normalize.output_height, param_.normalize.num_camera);
      if (camera_depth_ == nullptr) {
        printf("Failed to create depth.\n");
        return false;
      }

      camera_geometry_ = camera::create_geometry(param_.geometry);
      if (camera_geometry_ == nullptr) {
        printf("Failed to create geometry.\n");
        return false;
      }
    }

    transfusion_ = fuser::create_transfusion(param_.transfusion, param_.use_camera_);
    if (transfusion_ == nullptr) {
      printf("Failed to create transfusion.\n");
      return false;
    }

    transbbox_ = head::transbbox::create_transbbox(param_.transbbox);
    if (transbbox_ == nullptr) {
      printf("Failed to create head transbbox.\n");
      return false;
    }
    
    capacity_points_ = param_.voxelization.max_points;
    bytes_capacity_points_ = capacity_points_ * param_.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMalloc(&lidar_points_device_, bytes_capacity_points_));
    checkRuntime(cudaMallocHost(&lidar_points_host_, bytes_capacity_points_));
    return true;
  }

  std::vector<head::transbbox::BoundingBox> forward_only(const void* camera_images, const nvtype::half* lidar_points,
                                                         int num_points, void* stream, bool do_normalization) {
    int cappoints = static_cast<int>(capacity_points_);
    if (num_points > cappoints) {
      printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
    }

    num_points = std::min(cappoints, num_points);

    printf("==================BEVFusion===================\n");
    std::vector<float> times;
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    size_t bytes_points = num_points * param_.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
    checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));
    
    const nvtype::half* camera_bevfeat = nullptr;
    if (param_.use_camera_) {
      nvtype::half* normed_images = (nvtype::half*)camera_images;
      if (do_normalization) {
        normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), stream);
      }
      const nvtype::half* depth = this->camera_depth_->forward(lidar_points_device_, num_points, 5, stream);
      this->camera_backbone_->forward(normed_images, depth, stream);
      const nvtype::half* camera_bev = this->camera_bevpool_->forward(
          this->camera_backbone_->feature(), this->camera_backbone_->depth(), this->camera_geometry_->indices(),
          this->camera_geometry_->intervals(), this->camera_geometry_->num_intervals(), stream);
      camera_bevfeat = camera_vtransform_->forward(camera_bev, stream);
    }

    this->lidar_voxelization_->forward(lidar_points_device_, num_points, stream);
    const nvtype::half* point_feature = this->point_encoder_->forward(this->lidar_voxelization_->long_features(), stream);
    const nvtype::half* spatial_feature = this->pillar_scatter_->forward(point_feature, this->lidar_voxelization_->indices(), this->lidar_voxelization_->num_voxels_addr(), stream);
    const nvtype::half* lidar_feature = this->lidar_feature_encoder_->forward(spatial_feature, stream);
    const nvtype::half* fusion_feature = this->transfusion_->forward(camera_bevfeat, lidar_feature, stream);
    auto output = this->transbbox_->forward(fusion_feature, param_.transbbox.confidence_threshold, stream, param_.transbbox.sorted_bboxes);
    printf("=============================================\n");
    return output;
  }

  std::vector<head::transbbox::BoundingBox> forward_timer(const void* camera_images, const nvtype::half* lidar_points,
                                                          int num_points, void* stream, bool do_normalization) {
    int cappoints = static_cast<int>(capacity_points_);
    if (num_points > cappoints) {
      printf("If it exceeds %d points, the default processing will simply crop it out.\n", cappoints);
    }

    num_points = std::min(cappoints, num_points);

    printf("==================BEVFusion===================\n");
    std::vector<float> times;
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    
    timer_.start(_stream);
    size_t bytes_points = num_points * param_.voxelization.num_feature * sizeof(nvtype::half);
    checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_points, cudaMemcpyHostToHost, _stream));
    checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_, bytes_points, cudaMemcpyHostToDevice, _stream));
    timer_.stop("[NoSt] CopyLidar");

    const nvtype::half* camera_bevfeat = nullptr;
    if (param_.use_camera_) {
      nvtype::half* normed_images = (nvtype::half*)camera_images;
      if (do_normalization) {
        timer_.start(_stream);
        normed_images = (nvtype::half*)this->normalizer_->forward((const unsigned char**)(camera_images), stream);
        timer_.stop("[NoSt] ImageNrom");
      }

      timer_.start(_stream);
      const nvtype::half* depth = this->camera_depth_->forward(lidar_points_device_, num_points, 5, stream);
      times.emplace_back(timer_.stop("Camera Depth"));

      timer_.start(_stream);
      this->camera_backbone_->forward(normed_images, depth, stream);
      times.emplace_back(timer_.stop("Camera Backbone"));

      timer_.start(_stream);
      const nvtype::half* camera_bev = this->camera_bevpool_->forward(
          this->camera_backbone_->feature(), this->camera_backbone_->depth(), this->camera_geometry_->indices(),
          this->camera_geometry_->intervals(), this->camera_geometry_->num_intervals(), stream);
      times.emplace_back(timer_.stop("Camera Bevpool"));

      timer_.start(_stream);
      camera_bevfeat = camera_vtransform_->forward(camera_bev, stream);
      times.emplace_back(timer_.stop("VTransform"));
    }

    timer_.start(_stream);
    this->lidar_voxelization_->forward(lidar_points_device_, num_points, stream);
    times.emplace_back(timer_.stop("Voxelization"));

    timer_.start(_stream);
    const nvtype::half* point_feature = this->point_encoder_->forward(this->lidar_voxelization_->long_features(), stream);
    times.emplace_back(timer_.stop("PointEncoder"));

    timer_.start(_stream);
    const nvtype::half* spatial_feature = this->pillar_scatter_->forward(point_feature, this->lidar_voxelization_->indices(), this->lidar_voxelization_->num_voxels_addr(), stream);
    times.emplace_back(timer_.stop("PillarScatter"));

    timer_.start(_stream);
    const nvtype::half* lidar_feature = this->lidar_feature_encoder_->forward(spatial_feature, stream);
    times.emplace_back(timer_.stop("LidarFeatureEncoder"));

    timer_.start(_stream);
    const nvtype::half* fusion_feature = this->transfusion_->forward(camera_bevfeat, lidar_feature, stream);
    times.emplace_back(timer_.stop("Transfusion"));

    timer_.start(_stream);
    auto output =
        this->transbbox_->forward(fusion_feature, param_.transbbox.confidence_threshold, stream, param_.transbbox.sorted_bboxes);
    times.emplace_back(timer_.stop("Head BoundingBox"));

    float total_time = std::accumulate(times.begin(), times.end(), 0.0f, std::plus<float>{});
    printf("Total: %.3f ms\n", total_time);
    printf("=============================================\n");
    return output;
  }

  virtual std::vector<head::transbbox::BoundingBox> forward(const unsigned char** camera_images, const nvtype::half* lidar_points,
                                                            int num_points, void* stream) override {
    if (enable_timer_) {
      return this->forward_timer(camera_images, lidar_points, num_points, stream, true);
    } else {
      return this->forward_only(camera_images, lidar_points, num_points, stream, true);
    }
  }

  virtual std::vector<head::transbbox::BoundingBox> forward_no_normalize(const nvtype::half* camera_normed_images_device,
                                                                         const nvtype::half* lidar_points, int num_points,
                                                                         void* stream) override {
    if (enable_timer_) {
      return this->forward_timer(camera_normed_images_device, lidar_points, num_points, stream, false);
    } else {
      return this->forward_only(camera_normed_images_device, lidar_points, num_points, stream, false);
    }
  }

  virtual void set_timer(bool enable) override { enable_timer_ = enable; }

  virtual void print() override {
    point_encoder_->print();
    lidar_feature_encoder_->print();
    if (param_.use_camera_){
      camera_backbone_->print();
      camera_vtransform_->print();
    }
    transfusion_->print();
    transbbox_->print();
  }

  virtual void update(const float* camera2lidar, const float* camera_intrinsics, const float* lidar2image,
                      const float* img_aug_matrix, void* stream) override {
    camera_depth_->update(img_aug_matrix, lidar2image, stream);
    camera_geometry_->update(camera2lidar, camera_intrinsics, img_aug_matrix, stream);
  }

  virtual void free_excess_memory() override { camera_geometry_->free_excess_memory(); }

 private:
  CoreParameter param_;
  nv::EventTimer timer_;
  nvtype::half* lidar_points_device_ = nullptr;
  nvtype::half* lidar_points_host_ = nullptr;
  size_t capacity_points_ = 0;
  size_t bytes_capacity_points_ = 0;

  std::shared_ptr<lidar::Voxelization> lidar_voxelization_;
  std::shared_ptr<point_encoder::PointEncoder> point_encoder_;
  std::shared_ptr<scatter::PillarScatter> pillar_scatter_;
  std::shared_ptr<lidar_feature_encoder::LidarFeatureEncoder> lidar_feature_encoder_;
  std::shared_ptr<camera::Normalization> normalizer_;
  std::shared_ptr<camera::Backbone> camera_backbone_;
  std::shared_ptr<camera::BEVPool> camera_bevpool_;
  std::shared_ptr<camera::VTransform> camera_vtransform_;
  std::shared_ptr<camera::Depth> camera_depth_;
  std::shared_ptr<camera::Geometry> camera_geometry_;
  std::shared_ptr<fuser::Transfusion> transfusion_;
  std::shared_ptr<head::transbbox::TransBBox> transbbox_;
  float confidence_threshold_ = 0;
  bool enable_timer_ = false;
};

std::shared_ptr<Core> create_core(const CoreParameter& param) {
  std::shared_ptr<CoreImplement> instance(new CoreImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace bevfusion
