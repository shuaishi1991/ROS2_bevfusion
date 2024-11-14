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

#ifndef __PILLAR_SCATTER_HPP__
#define __PILLAR_SCATTER_HPP__

#include <memory>
#include <string>
#include <vector>
#include "pillarscatter-kernel.hpp"
#include "common/dtype.hpp"

namespace bevfusion {
namespace scatter {

class PillarScatter {
 public:
  virtual nvtype::half* forward(const nvtype::half* point_feature, const unsigned int* voxel_idxs, const unsigned int* voxel_num, void* stream) = 0;
  virtual std::vector<int> spatial_feature_shape() = 0;
};

std::shared_ptr<PillarScatter> create_pillar_scatter(std::vector<int>& output_shape);

};  // namespace scatter
};  // namespace bevfusion

#endif  // __PILLAR_SCATTER_HPP__