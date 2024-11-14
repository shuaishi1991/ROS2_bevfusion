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

#ifndef __LIDAR_FEATURE_ENCODER_HPP__
#define __LIDAR_FEATURE_ENCODER_HPP__

#include <memory>
#include <string>
#include <vector>

#include "common/dtype.hpp"

namespace bevfusion {
namespace lidar_feature_encoder {

class LidarFeatureEncoder {
 public:
  virtual nvtype::half* forward(const nvtype::half* spatial_feature, void* stream) = 0;
  virtual void print() = 0;
  virtual std::vector<int> spatial_feature_shape() = 0;
};

std::shared_ptr<LidarFeatureEncoder> create_lidar_feature_encoder(const std::string& model);

};  // namespace lidar_feature_encoder
};  // namespace bevfusion

#endif  // __LIDAR_FEATURE_ENCODER_HPP__