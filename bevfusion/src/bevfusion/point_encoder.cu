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
#include "point_encoder.hpp"

namespace bevfusion {
namespace point_encoder {

class PointEncoderImplement : public PointEncoder {
 public:
  virtual ~PointEncoderImplement() {
    if (output_) checkRuntime(cudaFree(output_));
  }

  virtual bool init(const std::string& model) {
    engine_ = TensorRT::load(model);
    if (engine_ == nullptr) return false;

    if (engine_->has_dynamic_dim()) {
      printf("Dynamic shapes are not supported.\n");
      return false;
    }

    int output_binding = 1;
    auto shape = engine_->static_dims(output_binding);
    output_shape_ = engine_->static_dims(output_binding);
    Asserts(engine_->dtype(output_binding) == TensorRT::DType::HALF, "Invalid binding data type.");

    size_t volumn = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    checkRuntime(cudaMalloc(&output_, volumn * sizeof(half)));
    return true;
  }

  virtual void print() override { engine_->print("PointEncoder"); }

  virtual nvtype::half* forward(const nvtype::half* voxels, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    engine_->forward({/* input  */ voxels,
                      /* output */ output_},
                     _stream);
    return output_;
  }

  virtual std::vector<int> point_feature_shape() override { return output_shape_; }

 private:
  std::shared_ptr<TensorRT::Engine> engine_;
  nvtype::half* output_ = nullptr;
  nvtype::half* spatial_feature = nullptr;
  std::vector<std::vector<int>> bindshape_;
  std::vector<int> output_shape_;
};

std::shared_ptr<PointEncoder> create_point_encoder(const std::string& param) {
  std::shared_ptr<PointEncoderImplement> instance(new PointEncoderImplement());
  if (!instance->init(param)) {
    instance.reset();
  }
  return instance;
}

};  // namespace point_encoder
};  // namespace bevfusion
