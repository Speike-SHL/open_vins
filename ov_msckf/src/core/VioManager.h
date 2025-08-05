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

#ifndef OV_MSCKF_VIOMANAGER_H
#define OV_MSCKF_VIOMANAGER_H

#include <Eigen/StdVector>
#include <algorithm>
#include <atomic>
#include <boost/filesystem.hpp>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

#include "VioManagerOptions.h"

namespace ov_core {
struct ImuData;
struct CameraData;
class TrackBase;
class FeatureInitializer;
} // namespace ov_core
namespace ov_init {
class InertialInitializer;
} // namespace ov_init

namespace ov_msckf {

class State;
class StateHelper;
class UpdaterMSCKF;
class UpdaterSLAM;
class UpdaterZeroVelocity;
class Propagator;

/**
 * @brief 管理整个系统的核心类
 *
 * 该类包含了 MSCKF 工作所需的状态和其他算法。
 * 我们将测量数据输入到该类，并将其发送到相应的算法中。
 * 如果有需要传播或更新的测量数据，该类会调用我们的状态进行处理。
 */
class VioManager {

public:
  /**
   * @brief 默认构造函数，将加载所有配置变量。
   *
   * 初始化视觉-惯性里程计（VIO）系统的核心管理器，
   * 加载配置参数并构建所有必要的算法模块（如状态估计器、特征跟踪器、初始化器等）。
   *
   * @param params_ 从 ROS 或命令行加载的参数
   */
  VioManager(VioManagerOptions &params_);

  /**
   * @brief 用于输入惯性数据的函数
   * @param message 包含时间戳和惯性信息
   */
  void feed_measurement_imu(const ov_core::ImuData &message);

  /**
   * @brief 用于输入相机测量数据的函数
   * @param message 包含时间戳、图像和相机ID的信息
   */
  void feed_measurement_camera(const ov_core::CameraData &message) { track_image_and_update(message); }

  /**
   * @brief 用于同步模拟相机的输入函数
   * @param timestamp 该图像采集的时间
   * @param camids 我们拥有模拟测量的相机ID
   * @param feats 原始的uv模拟测量值
   */
  void feed_measurement_simulation(double timestamp, const std::vector<int> &camids,
                                   const std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> &feats);

  /**
   * @brief 给定一个状态，将初始化我们的IMU状态。
   * @param imustate MSCKF顺序的状态: [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
   */
  void initialize_with_gt(Eigen::Matrix<double, 17, 1> imustate);

  /// 是否已经初始化
  bool initialized() { return is_initialized_vio && timelastupdate != -1; }

  /// 系统初始化时的时间戳
  double initialized_time() { return startup_time; }

  /// 获取当前系统参数
  VioManagerOptions get_params() { return params; }

  /// 获取当前状态
  std::shared_ptr<State> get_state() { return state; }

  /// 获取当前状态传播器
  std::shared_ptr<Propagator> get_propagator() { return propagator; }

  /// 获取当前跟踪特征的可视化图像
  cv::Mat get_historical_viz_image();

  /// 返回全局坐标系下的三维SLAM特征点
  std::vector<Eigen::Vector3d> get_features_SLAM();

  /// 返回全局坐标系下的三维ARUCO特征点
  std::vector<Eigen::Vector3d> get_features_ARUCO();

  /// 返回上次更新中使用的全局三维特征点
  std::vector<Eigen::Vector3d> get_good_features_MSCKF() { return good_features_MSCKF; }

  /// 返回用于投影当前活动特征的图像
  void get_active_image(double &timestamp, cv::Mat &image) {
    timestamp = active_tracks_time;
    image = active_image;
  }

  /// 返回当前帧中被跟踪的活动特征点
  void get_active_tracks(double &timestamp, std::unordered_map<size_t, Eigen::Vector3d> &feat_posinG,
                         std::unordered_map<size_t, Eigen::Vector3d> &feat_tracks_uvd) {
    timestamp = active_tracks_time;
    feat_posinG = active_tracks_posinG;
    feat_tracks_uvd = active_tracks_uvd;
  }

protected:
  /**
   * @brief 给定一组新的相机图像，将对其进行特征跟踪。
   *
   * 如果是双目跟踪，则调用双目跟踪函数。
   * 否则，将对传入的每一张图像分别进行特征跟踪。
   *
   * @param message 包含时间戳、图像和相机ID的信息
   */
  void track_image_and_update(const ov_core::CameraData &message);

  /**
   * @brief 该函数将对状态进行传播和特征更新
   * @param message 包含时间戳、图像和相机ID的信息
   */
  void do_feature_propagate_update(const ov_core::CameraData &message);

  /**
   * @brief 此函数将尝试初始化状态。
   *
   * 该函数应调用我们的初始化器并尝试初始化状态。
   * 未来我们应该在这里调用结构光束法（structure-from-motion）代码。
   * 此函数也可以用于系统失效后的重新初始化。
   *
   * @param message 包含时间戳、图像和相机ID的信息
   * @return 如果成功初始化则返回true
   */
  bool try_to_initialize(const ov_core::CameraData &message);

  /**
   * @brief 此函数将对当前帧中的所有特征点进行重新三角化
   *
   * 对于系统当前正在跟踪的所有特征点，将对它们进行重新三角化。
   * 这对于需要当前点云的下游应用（如回环检测）非常有用。
   * 该函数会尝试对*所有*点进行三角化，而不仅仅是那些在更新中使用过的点。
   *
   * @param message 包含时间戳、图像和相机ID的信息
   */
  void retriangulate_active_tracks(const ov_core::CameraData &message);

  /// 管理器参数
  VioManagerOptions params;

  /// 主状态对象
  std::shared_ptr<State> state;

  /// 状态传播器
  std::shared_ptr<Propagator> propagator;

  /// 稀疏特征跟踪器（KLT或描述子）
  std::shared_ptr<ov_core::TrackBase> trackFEATS;

  /// ARUCO跟踪器
  std::shared_ptr<ov_core::TrackBase> trackARUCO;

  /// 状态初始化器
  std::shared_ptr<ov_init::InertialInitializer> initializer;

  /// 是否已经初始化
  bool is_initialized_vio = false;

  /// MSCKF特征更新器
  std::shared_ptr<UpdaterMSCKF> updaterMSCKF;

  /// SLAM/ARUCO特征更新器
  std::shared_ptr<UpdaterSLAM> updaterSLAM;

  /// 零速更新器
  std::shared_ptr<UpdaterZeroVelocity> updaterZUPT;

  /// 这是自开始初始化以来收到的测量时间队列
  /// 初始化后，我们希望能够快速传播和更新到最新的时间戳
  std::vector<double> camera_queue_init;
  std::mutex camera_queue_init_mtx;

  // 统计文件和相关变量
  std::ofstream of_statistics;
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

  // 跟踪我们已经行进的距离
  double timelastupdate = -1;
  double distance = 0;

  // 滤波器的启动时间
  double startup_time = -1;

  // 线程及其原子变量
  std::atomic<bool> thread_init_running, thread_init_success;

  // 是否进行了零速更新
  bool did_zupt_update = false;
  bool has_moved_since_zupt = false;

  // 上次更新中使用的优秀特征点（用于可视化）
  std::vector<Eigen::Vector3d> good_features_MSCKF;

  // 从当前帧中看到的重新三角特征3d位置（用于可视化）
  // 对于每个特征，我们有一个线性系统A*p_FinG=b，我们创建并增加它们的成本
  double active_tracks_time = -1;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;
  cv::Mat active_image;
  std::map<size_t, Eigen::Matrix3d> active_feat_linsys_A;
  std::map<size_t, Eigen::Vector3d> active_feat_linsys_b;
  std::map<size_t, int> active_feat_linsys_count;
};

} // namespace ov_msckf

#endif // OV_MSCKF_VIOMANAGER_H
