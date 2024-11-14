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
#include "pillar_scatter.hpp"

namespace bevfusion {
namespace scatter {

class PillarScatterImplement : public PillarScatter {
 public:
  virtual ~PillarScatterImplement() {
    if (output_) checkRuntime(cudaFree(output_));
  }

  virtual bool init(std::vector<int>& output_shape) {

    output_shape_ = output_shape;
    size_t volumn = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
    checkRuntime(cudaMalloc(&output_, volumn * sizeof(half)));
    return true;
  }

  virtual nvtype::half* forward(const nvtype::half* point_feature, const unsigned int* voxel_idxs, const unsigned int* voxel_num, void* stream) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);

    size_t volumn = std::accumulate(output_shape_.begin(), output_shape_.end(), 1, std::multiplies<int>());
    checkRuntime(cudaMemsetAsync(output_, 0, volumn * sizeof(half), _stream));

    int status = -1;
    status = pillarScatterHalfKernelLaunch(
        (half*)point_feature,
        voxel_idxs,
        voxel_num,
        output_shape_[2],
        output_shape_[3],
        (half *)output_,
        _stream
        );
    assert(status == 0);

    return output_;
  }

  virtual std::vector<int> spatial_feature_shape() override { return output_shape_; }

 private:
  nvtype::half* output_ = nullptr;
  std::vector<int> output_shape_;
};

std::shared_ptr<PillarScatter> create_pillar_scatter(std::vector<int>& output_shape) {
  std::shared_ptr<PillarScatterImplement> instance(new PillarScatterImplement());
  if (!instance->init(output_shape)) {
    instance.reset();
  }
  return instance;
}

};  // namespace scatter
};  // namespace bevfusion
