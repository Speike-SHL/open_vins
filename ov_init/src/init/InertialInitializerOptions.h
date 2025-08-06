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

#ifndef OV_INIT_INERTIALINITIALIZEROPTIONS_H
#define OV_INIT_INERTIALINITIALIZEROPTIONS_H

#include <Eigen/Eigen>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "feat/FeatureInitializerOptions.h"
#include "track/TrackBase.h"
#include "utils/colors.h"
#include "utils/opencv_yaml_parse.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

namespace ov_init {

/**
 * @brief 存储状态估计所需所有选项的结构体。
 *
 * 这被分为几个不同的部分：估计器、跟踪器和仿真。
 * 如果您要在此处添加参数，则需要将其添加到解析器中。
 * 您还需要将其添加到每个部分底部的打印语句中。
 */
struct InertialInitializerOptions {

  /**
   * @brief 此函数将加载系统的非仿真参数并打印。
   * @param parser 如果不为空，此解析器将用于加载我们的参数
   */
  void print_and_load(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    print_and_load_initializer(parser);
    print_and_load_noise(parser);
    print_and_load_state(parser);
  }

  // INITIALIZATION ============================

  /// 我们将进行初始化的时间长度（秒）
  double init_window_time = 1.0;

  /// 加速度方差阈值，用于判断是否在运动
  double init_imu_thresh = 1.0;

  /// 认为设备静止的最大视差
  double init_max_disparity = 1.0;

  /// 我们应该尝试跟踪的特征点数量
  int init_max_features = 50;

  /// 是否应该执行动态初始化
  bool init_dyn_use = false;

  /// 是否应该在MLE中优化和恢复标定参数
  bool init_dyn_mle_opt_calib = false;

  /// 动态初始化的最大MLE迭代次数
  int init_dyn_mle_max_iter = 20;

  /// 动态初始化的最大MLE线程数
  int init_dyn_mle_max_threads = 20;

  /// MLE优化的最大时间（秒）
  double init_dyn_mle_max_time = 5.0;

  /// 初始化期间使用的姿态数量（最大应为相机频率 * 窗口）
  int init_dyn_num_pose = 5;

  /// 尝试初始化前需要旋转的最小角度（范数和）
  double init_dyn_min_deg = 45.0;

  /// 膨胀初始方向协方差的幅度
  double init_dyn_inflation_orientation = 10.0;

  /// 膨胀初始速度协方差的幅度
  double init_dyn_inflation_velocity = 10.0;

  /// 膨胀陀螺仪偏差初始协方差的幅度
  double init_dyn_inflation_bias_gyro = 100.0;

  /// 膨胀加速度计偏差初始协方差的幅度
  double init_dyn_inflation_bias_accel = 100.0;

  /// 协方差恢复可接受的最小倒数条件数（min_sigma / max_sigma <
  /// sqrt(min_reciprocal_condition_number)）
  double init_dyn_min_rec_cond = 1e-15;

  /// 动态初始化的初始IMU陀螺仪偏差值（将被优化）
  Eigen::Vector3d init_dyn_bias_g = Eigen::Vector3d::Zero();

  /// 动态初始化的初始IMU加速度计偏差值（将被优化）
  Eigen::Vector3d init_dyn_bias_a = Eigen::Vector3d::Zero();

  /**
   * @brief 此函数将加载并打印所有已加载的初始化器设置。
   * 这允许可视化检查是否从ROS/CMD解析器正确加载了所有内容。
   *
   * @param parser 如果不为空，此解析器将用于加载我们的参数
   */
  void print_and_load_initializer(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    PRINT_DEBUG("INITIALIZATION SETTINGS:\n");
    if (parser != nullptr) {
      parser->parse_config("init_window_time", init_window_time);
      parser->parse_config("init_imu_thresh", init_imu_thresh);
      parser->parse_config("init_max_disparity", init_max_disparity);
      parser->parse_config("init_max_features", init_max_features);
      parser->parse_config("init_dyn_use", init_dyn_use);
      parser->parse_config("init_dyn_mle_opt_calib", init_dyn_mle_opt_calib);
      parser->parse_config("init_dyn_mle_max_iter", init_dyn_mle_max_iter);
      parser->parse_config("init_dyn_mle_max_threads", init_dyn_mle_max_threads);
      parser->parse_config("init_dyn_mle_max_time", init_dyn_mle_max_time);
      parser->parse_config("init_dyn_num_pose", init_dyn_num_pose);
      parser->parse_config("init_dyn_min_deg", init_dyn_min_deg);
      parser->parse_config("init_dyn_inflation_ori", init_dyn_inflation_orientation);
      parser->parse_config("init_dyn_inflation_vel", init_dyn_inflation_velocity);
      parser->parse_config("init_dyn_inflation_bg", init_dyn_inflation_bias_gyro);
      parser->parse_config("init_dyn_inflation_ba", init_dyn_inflation_bias_accel);
      parser->parse_config("init_dyn_min_rec_cond", init_dyn_min_rec_cond);
      std::vector<double> bias_g = {0, 0, 0};
      std::vector<double> bias_a = {0, 0, 0};
      parser->parse_config("init_dyn_bias_g", bias_g);
      parser->parse_config("init_dyn_bias_a", bias_a);
      init_dyn_bias_g << bias_g.at(0), bias_g.at(1), bias_g.at(2);
      init_dyn_bias_a << bias_a.at(0), bias_a.at(1), bias_a.at(2);
    }
    PRINT_DEBUG("  - init_window_time: %.2f\n", init_window_time);
    PRINT_DEBUG("  - init_imu_thresh: %.2f\n", init_imu_thresh);
    PRINT_DEBUG("  - init_max_disparity: %.2f\n", init_max_disparity);
    PRINT_DEBUG("  - init_max_features: %.2f\n", init_max_features);
    if (init_max_features < 15) {
      PRINT_ERROR(RED "number of requested feature tracks to init not enough!!\n" RESET);
      PRINT_ERROR(RED "  init_max_features = %d\n" RESET, init_max_features);
      std::exit(EXIT_FAILURE);
    }
    if (init_imu_thresh <= 0.0 && !init_dyn_use) {
      PRINT_ERROR(RED "need to have an IMU threshold for static initialization!\n" RESET);
      PRINT_ERROR(RED "  init_imu_thresh = %.3f\n" RESET, init_imu_thresh);
      PRINT_ERROR(RED "  init_dyn_use = %d\n" RESET, init_dyn_use);
      std::exit(EXIT_FAILURE);
    }
    if (init_max_disparity <= 0.0 && !init_dyn_use) {
      PRINT_ERROR(RED "need to have an DISPARITY threshold for static initialization!\n" RESET);
      PRINT_ERROR(RED "  init_max_disparity = %.3f\n" RESET, init_max_disparity);
      PRINT_ERROR(RED "  init_dyn_use = %d\n" RESET, init_dyn_use);
      std::exit(EXIT_FAILURE);
    }
    PRINT_DEBUG("  - init_dyn_use: %d\n", init_dyn_use);
    PRINT_DEBUG("  - init_dyn_mle_opt_calib: %d\n", init_dyn_mle_opt_calib);
    PRINT_DEBUG("  - init_dyn_mle_max_iter: %d\n", init_dyn_mle_max_iter);
    PRINT_DEBUG("  - init_dyn_mle_max_threads: %d\n", init_dyn_mle_max_threads);
    PRINT_DEBUG("  - init_dyn_mle_max_time: %.2f\n", init_dyn_mle_max_time);
    PRINT_DEBUG("  - init_dyn_num_pose: %d\n", init_dyn_num_pose);
    PRINT_DEBUG("  - init_dyn_min_deg: %.2f\n", init_dyn_min_deg);
    PRINT_DEBUG("  - init_dyn_inflation_ori: %.2e\n", init_dyn_inflation_orientation);
    PRINT_DEBUG("  - init_dyn_inflation_vel: %.2e\n", init_dyn_inflation_velocity);
    PRINT_DEBUG("  - init_dyn_inflation_bg: %.2e\n", init_dyn_inflation_bias_gyro);
    PRINT_DEBUG("  - init_dyn_inflation_ba: %.2e\n", init_dyn_inflation_bias_accel);
    PRINT_DEBUG("  - init_dyn_min_rec_cond: %.2e\n", init_dyn_min_rec_cond);
    if (init_dyn_num_pose < 4) {
      PRINT_ERROR(RED "number of requested frames to init not enough!!\n" RESET);
      PRINT_ERROR(RED "  init_dyn_num_pose = %d (4 min)\n" RESET, init_dyn_num_pose);
      std::exit(EXIT_FAILURE);
    }
    PRINT_DEBUG("  - init_dyn_bias_g: %.2f, %.2f, %.2f\n", init_dyn_bias_g(0), init_dyn_bias_g(1), init_dyn_bias_g(2));
    PRINT_DEBUG("  - init_dyn_bias_a: %.2f, %.2f, %.2f\n", init_dyn_bias_a(0), init_dyn_bias_a(1), init_dyn_bias_a(2));
  }

  // NOISE / CHI2 ============================

  /// 陀螺仪白噪声（rad/s/sqrt(hz)）
  double sigma_w = 1.6968e-04;

  /// 陀螺仪随机游走（rad/s^2/sqrt(hz)）
  double sigma_wb = 1.9393e-05;

  /// 加速度计白噪声（m/s^2/sqrt(hz)）
  double sigma_a = 2.0000e-3;

  /// 加速度计随机游走（m/s^3/sqrt(hz)）
  double sigma_ab = 3.0000e-03;

  /// 原始像素测量的噪声标准差
  double sigma_pix = 1;

  /**
   * @brief 此函数将加载并打印所有已加载的噪声参数。
   * 这允许可视化检查是否从ROS/CMD解析器正确加载了所有内容。
   *
   * @param parser 如果不为空，此解析器将用于加载我们的参数
   */
  void print_and_load_noise(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    PRINT_DEBUG("NOISE PARAMETERS:\n");
    if (parser != nullptr) {
      parser->parse_external("relative_config_imu", "imu0", "gyroscope_noise_density", sigma_w);
      parser->parse_external("relative_config_imu", "imu0", "gyroscope_random_walk", sigma_wb);
      parser->parse_external("relative_config_imu", "imu0", "accelerometer_noise_density", sigma_a);
      parser->parse_external("relative_config_imu", "imu0", "accelerometer_random_walk", sigma_ab);
      parser->parse_config("up_slam_sigma_px", sigma_pix);
    }
    PRINT_DEBUG("  - gyroscope_noise_density: %.6f\n", sigma_w);
    PRINT_DEBUG("  - accelerometer_noise_density: %.5f\n", sigma_a);
    PRINT_DEBUG("  - gyroscope_random_walk: %.7f\n", sigma_wb);
    PRINT_DEBUG("  - accelerometer_random_walk: %.6f\n", sigma_ab);
    PRINT_DEBUG("  - sigma_pix: %.2f\n", sigma_pix);
  }

  // STATE DEFAULTS ==========================

  /// 全局坐标系中的重力大小（通常应为9.81）
  double gravity_mag = 9.81;

  /// 我们将在其中观察特征的不同相机数量
  int num_cameras = 1;

  /// 是否应该将两个相机作为立体或双目处理。如果是双目，我们对每个图像进行单目特征跟踪。
  bool use_stereo = true;

  /// 将使所有跟踪图像的分辨率减半（如果同时启用了dowsize_aruco，aruco将是1/4而不是减半）
  bool downsample_cameras = false;

  /// 相机和IMU之间的时间偏移（t_imu = t_cam + t_off）
  double calib_camimu_dt = 0.0;

  /// 相机ID和相机内参之间的映射（fx, fy, cx, cy, d1...d4, cam_w, cam_h）
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> camera_intrinsics;

  /// 相机ID和相机外参之间的映射（q_ItoC, p_IinC）。
  std::map<size_t, Eigen::VectorXd> camera_extrinsics;

  /**
   * @brief 此函数将加载并打印所有状态参数（例如传感器外参）
   * 这允许可视化检查是否从ROS/CMD解析器正确加载了所有内容。
   *
   * @param parser 如果不为空，此解析器将用于加载我们的参数
   */
  void print_and_load_state(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    if (parser != nullptr) {
      parser->parse_config("gravity_mag", gravity_mag);
      parser->parse_config("max_cameras", num_cameras); // might be redundant
      parser->parse_config("use_stereo", use_stereo);
      parser->parse_config("downsample_cameras", downsample_cameras);
      for (int i = 0; i < num_cameras; i++) {

        // 时间偏移（使用第一个）
        // TODO: 支持相机之间的多个时间偏移
        if (i == 0) {
          parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "timeshift_cam_imu", calib_camimu_dt, false);
        }

        // 畸变模型
        std::string dist_model = "radtan";
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "distortion_model", dist_model);

        // 畸变参数
        std::vector<double> cam_calib1 = {1, 1, 0, 0};
        std::vector<double> cam_calib2 = {0, 0, 0, 0};
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "intrinsics", cam_calib1);
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "distortion_coeffs", cam_calib2);
        Eigen::VectorXd cam_calib = Eigen::VectorXd::Zero(8);
        cam_calib << cam_calib1.at(0), cam_calib1.at(1), cam_calib1.at(2), cam_calib1.at(3), cam_calib2.at(0), cam_calib2.at(1),
            cam_calib2.at(2), cam_calib2.at(3);
        cam_calib(0) /= (downsample_cameras) ? 2.0 : 1.0;
        cam_calib(1) /= (downsample_cameras) ? 2.0 : 1.0;
        cam_calib(2) /= (downsample_cameras) ? 2.0 : 1.0;
        cam_calib(3) /= (downsample_cameras) ? 2.0 : 1.0;

        // 视场角 / 分辨率
        std::vector<int> matrix_wh = {1, 1};
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "resolution", matrix_wh);
        matrix_wh.at(0) /= (downsample_cameras) ? 2.0 : 1.0;
        matrix_wh.at(1) /= (downsample_cameras) ? 2.0 : 1.0;

        // 外参
        Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
        parser->parse_external("relative_config_imucam", "cam" + std::to_string(i), "T_imu_cam", T_CtoI);

        // 将这些加载到我们的状态中
        Eigen::Matrix<double, 7, 1> cam_eigen;
        cam_eigen.block(0, 0, 4, 1) = ov_core::rot_2_quat(T_CtoI.block(0, 0, 3, 3).transpose());
        cam_eigen.block(4, 0, 3, 1) = -T_CtoI.block(0, 0, 3, 3).transpose() * T_CtoI.block(0, 3, 3, 1);

        // 创建内参模型
        if (dist_model == "equidistant") {
          camera_intrinsics.insert({i, std::make_shared<ov_core::CamEqui>(matrix_wh.at(0), matrix_wh.at(1))});
          camera_intrinsics.at(i)->set_value(cam_calib);
        } else {
          camera_intrinsics.insert({i, std::make_shared<ov_core::CamRadtan>(matrix_wh.at(0), matrix_wh.at(1))});
          camera_intrinsics.at(i)->set_value(cam_calib);
        }
        camera_extrinsics.insert({i, cam_eigen});
      }
    }
    PRINT_DEBUG("STATE PARAMETERS:\n");
    PRINT_DEBUG("  - gravity_mag: %.4f\n", gravity_mag);
    PRINT_DEBUG("  - gravity: %.3f, %.3f, %.3f\n", 0.0, 0.0, gravity_mag);
    PRINT_DEBUG("  - num_cameras: %d\n", num_cameras);
    PRINT_DEBUG("  - use_stereo: %d\n", use_stereo);
    PRINT_DEBUG("  - downsize cameras: %d\n", downsample_cameras);
    if (num_cameras != (int)camera_intrinsics.size() || num_cameras != (int)camera_extrinsics.size()) {
      PRINT_ERROR(RED "[SIM]: camera calib size does not match max cameras...\n" RESET);
      PRINT_ERROR(RED "[SIM]: got %d but expected %d max cameras (camera_intrinsics)\n" RESET, (int)camera_intrinsics.size(), num_cameras);
      PRINT_ERROR(RED "[SIM]: got %d but expected %d max cameras (camera_extrinsics)\n" RESET, (int)camera_extrinsics.size(), num_cameras);
      std::exit(EXIT_FAILURE);
    }
    PRINT_DEBUG("  - calib_camimu_dt: %.4f\n", calib_camimu_dt);
    for (int n = 0; n < num_cameras; n++) {
      std::stringstream ss;
      ss << "cam_" << n << "_fisheye:" << (std::dynamic_pointer_cast<ov_core::CamEqui>(camera_intrinsics.at(n)) != nullptr) << std::endl;
      ss << "cam_" << n << "_wh:" << std::endl << camera_intrinsics.at(n)->w() << " x " << camera_intrinsics.at(n)->h() << std::endl;
      ss << "cam_" << n << "_intrinsic(0:3):" << std::endl
         << camera_intrinsics.at(n)->get_value().block(0, 0, 4, 1).transpose() << std::endl;
      ss << "cam_" << n << "_intrinsic(4:7):" << std::endl
         << camera_intrinsics.at(n)->get_value().block(4, 0, 4, 1).transpose() << std::endl;
      ss << "cam_" << n << "_extrinsic(0:3):" << std::endl << camera_extrinsics.at(n).block(0, 0, 4, 1).transpose() << std::endl;
      ss << "cam_" << n << "_extrinsic(4:6):" << std::endl << camera_extrinsics.at(n).block(4, 0, 3, 1).transpose() << std::endl;
      Eigen::Matrix4d T_CtoI = Eigen::Matrix4d::Identity();
      T_CtoI.block(0, 0, 3, 3) = ov_core::quat_2_Rot(camera_extrinsics.at(n).block(0, 0, 4, 1)).transpose();
      T_CtoI.block(0, 3, 3, 1) = -T_CtoI.block(0, 0, 3, 3) * camera_extrinsics.at(n).block(4, 0, 3, 1);
      ss << "T_C" << n << "toI:" << std::endl << T_CtoI << std::endl << std::endl;
      PRINT_DEBUG(ss.str().c_str());
    }
  }

  // SIMULATOR ===============================

  /// 初始状态的随机种子（即生成地图中随机特征的3D位置）
  int sim_seed_state_init = 0;

  /// 标定扰动的随机种子。如果启用了扰动，请更改此值以通过不同的随机值进行扰动。
  int sim_seed_preturb = 0;

  /// 测量噪声种子。在蒙特卡洛仿真中，每次运行都应该增加此值，以生成相同的真实测量值，
  /// 但产生不同的噪声值。
  int sim_seed_measurements = 0;

  /// 是否应该扰动估计器开始时的标定参数
  bool sim_do_perturbation = false;

  /// 我们将进行B样条插值和仿真的轨迹路径。应为time(s),pos(xyz),ori(xyzw)格式。
  std::string sim_traj_path = "../ov_data/sim/udel_gore.txt";

  /// 我们将在沿B样条移动这么多距离后开始仿真。这防止了静态开始，因为我们在仿真中从真实值初始化。
  double sim_distance_threshold = 1.2;

  /// 我们将仿真相机的频率（Hz）
  double sim_freq_cam = 10.0;

  /// 我们将仿真惯性测量单元的频率（Hz）
  double sim_freq_imu = 400.0;

  /// 我们生成特征的特征距离（最小值）
  double sim_min_feature_gen_distance = 5;

  /// 我们生成特征的特征距离（最大值）
  double sim_max_feature_gen_distance = 10;

  /**
   * @brief 此函数将加载并打印所有仿真参数。
   * 这允许可视化检查是否从ROS/CMD解析器正确加载了所有内容。
   *
   * @param parser 如果不为空，此解析器将用于加载我们的参数
   */
  void print_and_load_simulation(const std::shared_ptr<ov_core::YamlParser> &parser = nullptr) {
    if (parser != nullptr) {
      parser->parse_config("sim_seed_state_init", sim_seed_state_init);
      parser->parse_config("sim_seed_preturb", sim_seed_preturb);
      parser->parse_config("sim_seed_measurements", sim_seed_measurements);
      parser->parse_config("sim_do_perturbation", sim_do_perturbation);
      parser->parse_config("sim_traj_path", sim_traj_path);
      parser->parse_config("sim_distance_threshold", sim_distance_threshold);
      parser->parse_config("sim_freq_cam", sim_freq_cam);
      parser->parse_config("sim_freq_imu", sim_freq_imu);
      parser->parse_config("sim_min_feature_gen_dist", sim_min_feature_gen_distance);
      parser->parse_config("sim_max_feature_gen_dist", sim_max_feature_gen_distance);
    }
    PRINT_DEBUG("SIMULATION PARAMETERS:\n");
    PRINT_WARNING(BOLDRED "  - state init seed: %d \n" RESET, sim_seed_state_init);
    PRINT_WARNING(BOLDRED "  - perturb seed: %d \n" RESET, sim_seed_preturb);
    PRINT_WARNING(BOLDRED "  - measurement seed: %d \n" RESET, sim_seed_measurements);
    PRINT_WARNING(BOLDRED "  - do perturb?: %d\n" RESET, sim_do_perturbation);
    PRINT_DEBUG("  - traj path: %s\n", sim_traj_path.c_str());
    PRINT_DEBUG("  - dist thresh: %.2f\n", sim_distance_threshold);
    PRINT_DEBUG("  - cam feq: %.2f\n", sim_freq_cam);
    PRINT_DEBUG("  - imu feq: %.2f\n", sim_freq_imu);
    PRINT_DEBUG("  - min feat dist: %.2f\n", sim_min_feature_gen_distance);
    PRINT_DEBUG("  - max feat dist: %.2f\n", sim_max_feature_gen_distance);
  }
};

} // namespace ov_init

#endif // OV_INIT_INERTIALINITIALIZEROPTIONS_H