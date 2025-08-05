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

#ifndef OV_MSCKF_STATE_H
#define OV_MSCKF_STATE_H

#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "StateOptions.h"
#include "cam/CamBase.h"
#include "types/IMU.h"
#include "types/Landmark.h"
#include "types/PoseJPL.h"
#include "types/Type.h"
#include "types/Vec.h"

namespace ov_msckf {

/**
 * @brief 滤波器的状态
 *
 * 此状态包含滤波器的所有当前估计值。
 * 该系统仿照 MSCKF 滤波器建模，因此我们有一个滑动窗口的克隆状态。
 * 此外，我们还增加了用于在线标定和 SLAM 特征估计的参数。
 * 同时还包含了系统的协方差矩阵，应该通过 StateHelper 类进行管理。
 */
class State {

public:
  /**
   * @brief 默认构造函数（将变量初始化为默认值）
   * @param options_ 包含滤波器选项的结构体
   */
  State(StateOptions &options_);

  ~State() {}

  /**
   * @brief 返回下一个将要边缘化的时间步。（所有IMU clone中的最早时间）
   * 目前，由于我们使用的是滑动窗口，这是最旧的克隆。
   * 但如果你想实现关键帧系统，也可以选择性地边缘化克隆。
   * @return 将要边缘化的克隆的时间步
   */
  double margtimestep() {
    std::lock_guard<std::mutex> lock(_mutex_state);
    double time = INFINITY;
    for (const auto &clone_imu : _clones_IMU) {
      if (clone_imu.first < time) {
        time = clone_imu.first;
      }
    }
    return time;
  }

  /**
   * @brief 计算当前协方差矩阵的最大尺寸
   * @return 当前协方差矩阵的尺寸
   */
  int max_covariance_size() { return (int)_Cov.rows(); }

  /**
   * @brief 陀螺仪和加速度计的内参矩阵（比例不完美和轴不对齐）
   *
   * 如果是 kalibr 模型，则使用矩阵的下三角部分
   * 如果是 rpng 模型，则使用矩阵的上三角部分
   *
   * @return 当前 IMU 陀螺仪/加速度计内参的 3x3 矩阵
   */
  static Eigen::Matrix3d Dm(StateOptions::ImuModel imu_model, const Eigen::MatrixXd &vec) {
    assert(vec.rows() == 6);
    assert(vec.cols() == 1);
    Eigen::Matrix3d D_matrix = Eigen::Matrix3d::Identity();
    if (imu_model == StateOptions::ImuModel::KALIBR) {
      D_matrix << vec(0), 0, 0, vec(1), vec(3), 0, vec(2), vec(4), vec(5);
    } else {
      D_matrix << vec(0), vec(1), vec(3), 0, vec(2), vec(4), 0, 0, vec(5);
    }
    return D_matrix;
  }

  /**
   * @brief 陀螺仪重力敏感性
   *
   * 对于 kalibr 和 rpng 两种模型，这都是按列填充的 3x3 矩阵。
   *
   * @return 当前重力敏感性的 3x3 矩阵
   */
  static Eigen::Matrix3d Tg(const Eigen::MatrixXd &vec) {
    assert(vec.rows() == 9);
    assert(vec.cols() == 1);
    Eigen::Matrix3d Tg = Eigen::Matrix3d::Zero();
    Tg << vec(0), vec(3), vec(6), vec(1), vec(4), vec(7), vec(2), vec(5), vec(8);
    return Tg;
  }

  /**
   * @brief 计算 IMU 内参的误差状态大小。
   *
   * 该函数用于构建我们的状态转移矩阵，其大小取决于我们是否在估计标定参数。
   * 如果估计内参则为 15，若还估计重力敏感性则再加 9。
   *
   * @return 误差状态的大小
   */
  int imu_intrinsic_size() const {
    int sz = 0;
    if (_options.do_calib_imu_intrinsics) {
      sz += 15;
      if (_options.do_calib_imu_g_sensitivity) {
        sz += 9;
      }
    }
    return sz;
  }

  /// 用于锁定状态访问的互斥量
  std::mutex _mutex_state;

  /// 当前时间戳（应为相机时钟帧下的最后更新时间！）
  double _timestamp = -1;

  /// 包含滤波器选项的结构体
  StateOptions _options;

  /// 指向“活动”IMU 状态的指针（q_GtoI, p_IinG, v_IinG, bg, ba）
  std::shared_ptr<ov_type::IMU> _imu;

  /// 成像时间与克隆位姿（q_GtoIi, p_IiinG）之间的映射
  std::map<double, std::shared_ptr<ov_type::PoseJPL>> _clones_IMU;

  /// 当前 SLAM 特征集（3D 位置）
  std::unordered_map<size_t, std::shared_ptr<ov_type::Landmark>> _features_SLAM;

  /// IMU 到相机的时间偏移（t_imu = t_cam + t_off）
  std::shared_ptr<ov_type::Vec> _calib_dt_CAMtoIMU;

  /// 每个相机的标定位姿（R_ItoC, p_IinC）
  std::unordered_map<size_t, std::shared_ptr<ov_type::PoseJPL>> _calib_IMUtoCAM;

  /// 相机内参
  std::unordered_map<size_t, std::shared_ptr<ov_type::Vec>> _cam_intrinsics;

  /// 相机内参相机对象
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> _cam_intrinsics_cameras;

  /// 陀螺仪 IMU 内参（比例不完美和轴不对齐）
  std::shared_ptr<ov_type::Vec> _calib_imu_dw;

  /// 加速度计 IMU 内参（比例不完美和轴不对齐）
  std::shared_ptr<ov_type::Vec> _calib_imu_da;

  /// 陀螺仪重力敏感性
  std::shared_ptr<ov_type::Vec> _calib_imu_tg;

  /// 从陀螺仪坐标系到“IMU”加速度计坐标系的旋转（kalibr 模型）
  std::shared_ptr<ov_type::JPLQuat> _calib_imu_GYROtoIMU;

  /// 从加速度计到“IMU”陀螺仪坐标系的旋转（rpng 模型）
  std::shared_ptr<ov_type::JPLQuat> _calib_imu_ACCtoIMU;

private:
  // 定义 StateHelper 类为此类的友元
  // 使其能够访问下方通常不应被调用的函数
  // 避免开发者误以为"插入克隆"操作会正确添加到协方差矩阵
  friend class StateHelper;

  /// 所有活跃状态变量的协方差矩阵
  Eigen::MatrixXd _Cov;

  /// 所有活跃状态变量的指针
  std::vector<std::shared_ptr<ov_type::Type>> _variables;
};

} // namespace ov_msckf

#endif // OV_MSCKF_STATE_H
