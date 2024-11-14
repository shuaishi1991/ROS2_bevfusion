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

#include <algorithm>
#include <numeric>
#include <iostream>
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"
#include "transfusion.hpp"

namespace bevfusion {
namespace fuser {

class TransfusionImplement : public Transfusion {
 public:
  virtual ~TransfusionImplement() {
    if (output_) checkRuntime(cudaFree(output_));
  }

  virtual bool init(const std::string& model, bool use_camera) {

    use_camera_ = use_camera;

    engine_ = TensorRT::load(model);
    if (engine_ == nullptr) return false;

    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported.\n");
      return false;
    }

    int output_binding;
    if (use_camera_){
      output_binding = 2;
    } else {
      output_binding = 1;
    }
    auto shape = engine_->static_dims(output_binding);
    Asserts(engine_->dtype(output_binding) == TensorRT::DType::HALF, "Invalid binding data type.");

    size_t volumn = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    checkRuntime(cudaMalloc(&output_, volumn * sizeof(half)));
    return true;
  }

  virtual void print() override { engine_->print("Transfusion"); }

  virtual nvtype::half* forward(const nvtype::half* camera_bev, const nvtype::half* lidar_bev, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    if (use_camera_) {
      engine_->forward({/* input  */ camera_bev, lidar_bev,
                            /* output */ output_},
                          _stream);
    } else {
      engine_->forward({/* input  */ lidar_bev,
                      /* output */ output_},
                     _stream);
    }
    return output_;
  }

 private:
  bool use_camera_ = false;
  std::shared_ptr<TensorRT::Engine> engine_;
  nvtype::half* output_ = nullptr;
  std::vector<std::vector<int>> bindshape_;
};

std::shared_ptr<Transfusion> create_transfusion(const std::string& param, bool use_camera) {
  std::shared_ptr<TransfusionImplement> instance(new TransfusionImplement());
  if (!instance->init(param, use_camera)) {
    instance.reset();
  }
  return instance;
}

};  // namespace fuser
};  // namespace bevfusion
