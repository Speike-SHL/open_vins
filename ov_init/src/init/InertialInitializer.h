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

#ifndef OV_INIT_INERTIALINITIALIZER_H
#define OV_INIT_INERTIALINITIALIZER_H

#include "init/InertialInitializerOptions.h"

namespace ov_core {
class FeatureDatabase;
struct ImuData;
} // namespace ov_core
namespace ov_type {
class Type;
class IMU;
} // namespace ov_type

namespace ov_init {

class StaticInitializer;
class DynamicInitializer;

/**
 * @brief 视觉惯性系统初始化器。
 *
 * 这将尝试对状态进行动态和静态初始化。
 * 用户可以请求等待IMU读数跳跃（即设备被拿起）或尽快初始化。
 * 对于静态初始化，用户需要预先指定校准，否则总是使用动态初始化。
 * 逻辑如下：
 * 1. 尝试执行状态元素的动态初始化。
 * 2. 如果这失败了并且我们有校准，那么我们可以尝试做静态初始化
 * 3. 如果设备是静止的并且我们在等待一个抖动，就返回，否则初始化状态！
 *
 * 动态系统基于对工作[未知相机-IMU校准的视觉辅助惯性导航中的估计器初始化]
 * (https://ieeexplore.ieee.org/document/6386235) @cite Dong2012IROS的实现和扩展，该工作通过首先创建一个
 * 用于恢复相机到IMU旋转的线性系统，然后用于速度、重力和特征位置，最后进行完全优化以允许协方差恢复来解决初始化问题。
 * 另一篇读者可能感兴趣的论文是[视觉惯性系统IMU初始化问题的解析解]
 * (https://ieeexplore.ieee.org/abstract/document/9462400)，其中有一些关于尺度恢复和加速度计偏置的详细实验。
 */
class InertialInitializer {

public:
  /**
   * @brief 默认构造函数
   * @param params_ 从ROS或命令行加载的参数
   * @param db 包含所有特征的特征跟踪数据库
   */
  explicit InertialInitializer(InertialInitializerOptions &params_, std::shared_ptr<ov_core::FeatureDatabase> db);

  /**
   * @brief 惯性数据输入函数
   * @param message 包含我们的时间戳和惯性信息
   * @param oldest_time 我们可以丢弃测量值之前的时间
   */
  void feed_imu(const ov_core::ImuData &message, double oldest_time = -1);

  /**
   * @brief 尝试获取初始化的系统
   *
   *
   * @m_class{m-note m-warning}
   *
   * @par 处理成本
   * 这是一个串行过程，可能需要数秒钟才能完成。
   * 如果您是实时应用程序，那么您可能希望从异步线程调用此函数，
   * 这允许在后台进行处理。
   * 使用的特征是从特征数据库克隆的，因此应该是线程安全的，
   * 可以继续向数据库添加新的特征轨迹。
   *
   * @param[out] timestamp 我们初始化状态的时间戳
   * @param[out] covariance 返回状态的计算协方差
   * @param[out] order 协方差矩阵的顺序
   * @param[out] t_imu 我们的IMU类型（需要有正确的ID）
   * @param wait_for_jerk 如果为true，我们将等待一个"抖动"
   * @return 如果我们成功初始化了系统则返回True
   */
  bool initialize(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<ov_type::Type>> &order,
                  std::shared_ptr<ov_type::IMU> t_imu, bool wait_for_jerk = true);

protected:
  /// 初始化参数
  InertialInitializerOptions params;

  /// 包含所有特征的特征跟踪数据库
  std::shared_ptr<ov_core::FeatureDatabase> _db;

  /// 我们的IMU消息历史（时间、角度、线性）
  std::shared_ptr<std::vector<ov_core::ImuData>> imu_data;

  /// 静态初始化辅助类
  std::shared_ptr<StaticInitializer> init_static;

  /// 动态初始化辅助类
  std::shared_ptr<DynamicInitializer> init_dynamic;
};

} // namespace ov_init

#endif // OV_INIT_INERTIALINITIALIZER_H
