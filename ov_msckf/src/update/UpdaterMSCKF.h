/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef OV_MSCKF_UPDATER_MSCKF_H
#define OV_MSCKF_UPDATER_MSCKF_H

#include <Eigen/Eigen>
#include <memory>

#include "feat/FeatureInitializerOptions.h"

#include "UpdaterOptions.h"

namespace ov_core {
class Feature;
class FeatureInitializer;
} // namespace ov_core

namespace ov_msckf {

class State;

/**
 * @brief 计算稀疏特征的系统并更新滤波器。
 *
 * 该类负责为所有将用于更新的特征计算完整的线性系统。
 * 这遵循了原始的 MSCKF 方法：首先对特征进行三角化，然后对特征雅可比矩阵进行零空间投影。
 * 之后，我们将所有观测压缩，以实现高效的更新，并对状态进行更新。
 */
class UpdaterMSCKF {

public:
  /**
   * @brief MSCKF 更新器的默认构造函数
   *
   * 我们的更新器包含一个特征初始化器，用于在需要时初始化特征。
   * 同时，options 允许用户调整不同的更新参数。
   *
   * @param options 更新器选项（包括测量噪声值）
   * @param feat_init_options 特征初始化器选项
   */
  UpdaterMSCKF(UpdaterOptions &options, ov_core::FeatureInitializerOptions &feat_init_options);

  /**
   * @brief 给定跟踪到的特征，将尝试使用它们来更新状态。
   *
   * @param state 滤波器的状态
   * @param feature_vec 可用于更新的特征集合
   */
  void update(std::shared_ptr<State> state, std::vector<std::shared_ptr<ov_core::Feature>> &feature_vec);

protected:
  /// 更新过程中使用的选项
  UpdaterOptions _options;

  /// 特征初始化器类对象
  std::shared_ptr<ov_core::FeatureInitializer> initializer_feat;

  /// 卡方95百分位表（查找结果将是残差的大小）
  std::map<int, double> chi_squared_table;
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_MSCKF_H
