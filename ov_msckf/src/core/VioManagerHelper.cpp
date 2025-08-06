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

#include "VioManager.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "types/LandmarkRepresentation.h"
#include "utils/print.h"

#include "init/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

void VioManager::initialize_with_gt(Eigen::Matrix<double, 17, 1> imustate) {

  // 初始化系统
  state->_imu->set_value(imustate.block(1, 0, 16, 1));
  state->_imu->set_fej(imustate.block(1, 0, 16, 1));

  // 固定全局偏航和位置的规范自由度
  // TODO：为什么这会破坏仿真一致性指标？
  std::vector<std::shared_ptr<ov_type::Type>> order = {state->_imu};
  Eigen::MatrixXd Cov = std::pow(0.02, 2) * Eigen::MatrixXd::Identity(state->_imu->size(), state->_imu->size());
  Cov.block(0, 0, 3, 3) = std::pow(0.017, 2) * Eigen::Matrix3d::Identity(); // q
  Cov.block(3, 3, 3, 3) = std::pow(0.05, 2) * Eigen::Matrix3d::Identity();  // p
  Cov.block(6, 6, 3, 3) = std::pow(0.01, 2) * Eigen::Matrix3d::Identity();  // v (static)
  StateHelper::set_initial_covariance(state, Cov, order);

  // 设置状态时间
  state->_timestamp = imustate(0, 0);
  startup_time = imustate(0, 0);
  is_initialized_vio = true;

  // 清理所有早于初始化时间的特征
  trackFEATS->get_feature_database()->cleanup_measurements(state->_timestamp);
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup_measurements(state->_timestamp);
  }

  // 打印初始化的内容
  PRINT_DEBUG(GREEN "[INIT]: INITIALIZED FROM GROUNDTRUTH FILE!!!!!\n" RESET);
  PRINT_DEBUG(GREEN "[INIT]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, state->_imu->quat()(0), state->_imu->quat()(1),
              state->_imu->quat()(2), state->_imu->quat()(3));
  PRINT_DEBUG(GREEN "[INIT]: bias gyro = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_g()(0), state->_imu->bias_g()(1),
              state->_imu->bias_g()(2));
  PRINT_DEBUG(GREEN "[INIT]: velocity = %.4f, %.4f, %.4f\n" RESET, state->_imu->vel()(0), state->_imu->vel()(1), state->_imu->vel()(2));
  PRINT_DEBUG(GREEN "[INIT]: bias accel = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_a()(0), state->_imu->bias_a()(1),
              state->_imu->bias_a()(2));
  PRINT_DEBUG(GREEN "[INIT]: position = %.4f, %.4f, %.4f\n" RESET, state->_imu->pos()(0), state->_imu->pos()(1), state->_imu->pos()(2));
}

bool VioManager::try_to_initialize(const ov_core::CameraData &message) {


  // 如果初始化线程正在运行，将相机的时间戳加入队列，以备后续初始化完成可以快速追赶到当前时间。
  if (thread_init_running) {
    std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
    camera_queue_init.push_back(message.timestamp);
    return false;
  }

  // 如果线程初始化成功，直接返回成功
  if (thread_init_success) {
    return true;
  }

  // 在另一个线程中运行初始化，这样它可以按需慢慢进行
  thread_init_running = true;
  std::thread thread([&] {
    // 初始化器的返回值
    double timestamp;
    Eigen::MatrixXd covariance;
    std::vector<std::shared_ptr<ov_type::Type>> order;
    auto init_rT1 = boost::posix_time::microsec_clock::local_time();

    // 尝试初始化系统
    // 如果没有启用零速更新，则等待加速度突变
    // 否则可以立即初始化，因为零速会处理静止情况
    bool wait_for_jerk = (updaterZUPT == nullptr);
    bool success = initializer->initialize(timestamp, covariance, order, state->_imu, wait_for_jerk);

    // 如果初始化成功，则设置协方差和状态元素
    // TODO：在此处设置克隆和SLAM特征，以便我们可以立即开始更新...
    if (success) {

      // 设置我们的协方差（状态应已在初始化器中设置）
      StateHelper::set_initial_covariance(state, covariance, order);

      // 设置状态时间
      state->_timestamp = timestamp;
      startup_time = timestamp;

      // 清理所有早于初始化时间的特征
      // 并在估计期间将特征数量增加到期望值
      // NOTE：我们会将总特征数在所有相机间均匀分配
      trackFEATS->get_feature_database()->cleanup_measurements(state->_timestamp);
      trackFEATS->set_num_features(std::floor((double)params.num_pts / (double)params.state_options.num_cameras));
      if (trackARUCO != nullptr) {
        trackARUCO->get_feature_database()->cleanup_measurements(state->_timestamp);
      }

      // 如果我们在移动，则不进行零速更新
      if (state->_imu->vel().norm() > params.zupt_max_velocity) {
        has_moved_since_zupt = true;
      }

      // 否则一切正常，打印统计信息
      auto init_rT2 = boost::posix_time::microsec_clock::local_time();
      PRINT_INFO(GREEN "[init]: successful initialization in %.4f seconds\n" RESET, (init_rT2 - init_rT1).total_microseconds() * 1e-6);
      PRINT_INFO(GREEN "[init]: orientation = %.4f, %.4f, %.4f, %.4f\n" RESET, state->_imu->quat()(0), state->_imu->quat()(1),
                 state->_imu->quat()(2), state->_imu->quat()(3));
      PRINT_INFO(GREEN "[init]: bias gyro = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_g()(0), state->_imu->bias_g()(1),
                 state->_imu->bias_g()(2));
      PRINT_INFO(GREEN "[init]: velocity = %.4f, %.4f, %.4f\n" RESET, state->_imu->vel()(0), state->_imu->vel()(1), state->_imu->vel()(2));
      PRINT_INFO(GREEN "[init]: bias accel = %.4f, %.4f, %.4f\n" RESET, state->_imu->bias_a()(0), state->_imu->bias_a()(1),
                 state->_imu->bias_a()(2));
      PRINT_INFO(GREEN "[init]: position = %.4f, %.4f, %.4f\n" RESET, state->_imu->pos()(0), state->_imu->pos()(1), state->_imu->pos()(2));

      // 初始化成功后，可以取出初始化时刻之后的相机时间戳，方便后续快速追赶到当前时间。
      // 如果初始化耗时较长，可能会发生相机时间戳队列中存在早于初始化时刻的相机时间戳。
      std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
      std::vector<double> camera_timestamps_to_init;
      for (size_t i = 0; i < camera_queue_init.size(); i++) {
        if (camera_queue_init.at(i) > timestamp) {
          camera_timestamps_to_init.push_back(camera_queue_init.at(i));
        }
      }

      // 现在我们已经初始化，利用储存的相机数据将状态传播到当前时刻
      // 只要初始化没有花费太长时间，这通常没问题
      // 如果初始偏置很差，跨越多秒传播会成为问题
      size_t clone_rate = (size_t)((double)camera_timestamps_to_init.size() / (double)params.state_options.max_clone_size) + 1;
      for (size_t i = 0; i < camera_timestamps_to_init.size(); i += clone_rate) {
        propagator->propagate_and_clone(state, camera_timestamps_to_init.at(i));
        StateHelper::marginalize_old_clone(state);
      }
      PRINT_DEBUG(YELLOW "[init]: moved the state forward %.2f seconds\n" RESET, state->_timestamp - timestamp);
      thread_init_success = true;
      camera_queue_init.clear();

    } else {
      auto init_rT2 = boost::posix_time::microsec_clock::local_time();
      PRINT_DEBUG(YELLOW "[init]: failed initialization in %.4f seconds\n" RESET, (init_rT2 - init_rT1).total_microseconds() * 1e-6);
      thread_init_success = false;
      std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
      camera_queue_init.clear();
    }

    thread_init_running = false;
  });

  if (!params.use_multi_threading_subs) {
    thread.join();   // 单线程阻塞等待初始化完成
  } else {
    thread.detach(); // 多线程分离
  }
  return false;
}

void VioManager::retriangulate_active_tracks(const ov_core::CameraData &message) {

  // 开始计时
  boost::posix_time::ptime retri_rT1, retri_rT2, retri_rT3;
  retri_rT1 = boost::posix_time::microsec_clock::local_time();

  // 清除旧的活动跟踪数据
  assert(state->_clones_IMU.find(message.timestamp) != state->_clones_IMU.end());
  active_tracks_time = message.timestamp;
  active_image = cv::Mat();
  trackFEATS->display_active(active_image, 255, 255, 255, 255, 255, 255, " ");
  if (!active_image.empty()) {
    active_image = active_image(cv::Rect(0, 0, message.images.at(0).cols, message.images.at(0).rows));
  }
  active_tracks_posinG.clear();
  active_tracks_uvd.clear();

  // 前端（Frontend）中当前的活动跟踪点
  // TODO: 这里或许应该断言（assert）这些特征点观测对应的是消息的时间戳...
  auto last_obs = trackFEATS->get_last_obs();
  auto last_ids = trackFEATS->get_last_ids();

  // 新的线性系统集合，仅包含最新的跟踪信息
  std::map<size_t, Eigen::Matrix3d> active_feat_linsys_A_new;
  std::map<size_t, Eigen::Vector3d> active_feat_linsys_b_new;
  std::map<size_t, int> active_feat_linsys_count_new;
  std::unordered_map<size_t, Eigen::Vector3d> active_tracks_posinG_new;

  // 为每个相机（图像）附加我们新的观测值
  std::map<size_t, cv::Point2f> feat_uvs_in_cam0; // 在相机0（cam0）中记录的匹配点UV坐标
  for (auto const &cam_id : message.sensor_ids) {

    // IMU的历史克隆位姿（IMU相对于全局坐标系）
    Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(active_tracks_time)->Rot();
    Eigen::Vector3d p_IinG = state->_clones_IMU.at(active_tracks_time)->pos();

    // 当前相机（cam_id）的标定（外参，从IMU到相机）
    Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(cam_id)->Rot();
    Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(cam_id)->pos();

    // 计算当前相机（CAMERA）相对于全局坐标系的位置
    Eigen::Matrix3d R_GtoCi = R_ItoC * R_GtoI;
    Eigen::Vector3d p_CiinG = p_IinG - R_GtoCi.transpose() * p_IinC;

    // 循环处理每个测量值
    assert(last_obs.find(cam_id) != last_obs.end());
    assert(last_ids.find(cam_id) != last_ids.end());
    for (size_t i = 0; i < last_obs.at(cam_id).size(); i++) {

      // 如果该特征点被相机0（cam0）观测到，则记录其去畸变前的UV坐标（pt_d）
      size_t featid = last_ids.at(cam_id).at(i);
      cv::Point2f pt_d = last_obs.at(cam_id).at(i).pt;
      if (cam_id == 0) {
        feat_uvs_in_cam0[featid] = pt_d;
      }

      // 如果该特征点是SLAM特征点（其状态估计具有优先级），则跳过它
      if (state->_features_SLAM.find(featid) != state->_features_SLAM.end()) {
        continue;
      }

      // 获取归一化平面坐标（pt_n）
      cv::Point2f pt_n = state->_cam_intrinsics_cameras.at(cam_id)->undistort_cv(pt_d);
      Eigen::Matrix<double, 3, 1> b_i;
      b_i << pt_n.x, pt_n.y, 1;
      b_i = R_GtoCi.transpose() * b_i;     // 转换到全局坐标系下的方向向量
      b_i = b_i / b_i.norm();              // 归一化
      Eigen::Matrix3d Bperp = skew_x(b_i); // 构建法平面投影矩阵

      // 将当前观测添加到线性系统中
      Eigen::Matrix3d Ai = Bperp.transpose() * Bperp; // 贡献矩阵A
      Eigen::Vector3d bi = Ai * p_CiinG;              // 贡献向量b
      if (active_feat_linsys_A.find(featid) == active_feat_linsys_A.end()) {
        // 新特征点：在映射中插入初始值
        active_feat_linsys_A_new.insert({featid, Ai});
        active_feat_linsys_b_new.insert({featid, bi});
        active_feat_linsys_count_new.insert({featid, 1});
      } else {
        // 已存在特征点：累积贡献
        active_feat_linsys_A_new[featid] = Ai + active_feat_linsys_A[featid];
        active_feat_linsys_b_new[featid] = bi + active_feat_linsys_b[featid];
        active_feat_linsys_count_new[featid] = 1 + active_feat_linsys_count[featid];
      }

      // 对于该特征点，如果我们有足够的观测次数，则恢复其3D位置！
      if (active_feat_linsys_count_new.at(featid) > 3) {

        // 通过求解线性系统恢复特征点估计（p_FinG）
        Eigen::Matrix3d A = active_feat_linsys_A_new[featid];
        Eigen::Vector3d b = active_feat_linsys_b_new[featid];
        Eigen::MatrixXd p_FinG = A.colPivHouseholderQr().solve(b);
        // 将特征点位置转换到当前相机坐标系 (Ci)
        Eigen::MatrixXd p_FinCi = R_GtoCi * (p_FinG - p_CiinG);

        // 检查矩阵A的条件数和p_FinCi的Z分量
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(A);
        Eigen::MatrixXd singularValues;
        singularValues.resize(svd.singularValues().rows(), 1);
        singularValues = svd.singularValues();
        double condA = singularValues(0, 0) / singularValues(singularValues.rows() - 1, 0); // 条件数

        // 如果条件数过差，或者估计值超出范围（太近或太远），或者计算值非法（NaN）
        // 则不使用该估计（保持bad标志 - 通过将active_tracks_posinG_new中的z轴设为nan？此处实现未明说）
        if (std::abs(condA) <= params.featinit_options.max_cond_number && p_FinCi(2, 0) >= params.featinit_options.min_dist &&
            p_FinCi(2, 0) <= params.featinit_options.max_dist && !std::isnan(p_FinCi.norm())) {
          // 条件良好且在有效范围内：保存3D位置
          active_tracks_posinG_new[featid] = p_FinG;
        }
      }
    }
  }
  size_t total_triangulated = active_tracks_posinG.size(); // 记录本次成功三角化的特征点数量

  // 更新活动的线性系统集
  active_feat_linsys_A = active_feat_linsys_A_new;
  active_feat_linsys_b = active_feat_linsys_b_new;
  active_feat_linsys_count = active_feat_linsys_count_new;
  active_tracks_posinG = active_tracks_posinG_new;             // 更新全局坐标系下活动特征点的3D位置集合
  retri_rT2 = boost::posix_time::microsec_clock::local_time(); // 第二计时点（结束线性系统构建与三角化）

  // 如果没有特征点（包括活动跟踪点和SLAM特征点），则提前返回
  if (active_tracks_posinG.empty() && state->_features_SLAM.empty())
    return;

  // 附加我们已有的SLAM特征点
  for (const auto &feat : state->_features_SLAM) {
    Eigen::Vector3d p_FinG = feat.second->get_xyz(false); // 获取特征点位置表示
    if (LandmarkRepresentation::is_relative_representation(feat.second->_feat_representation)) {
      // 断言该特征点存在锚点位姿
      assert(feat.second->_anchor_cam_id != -1);
      // 获取锚点相机的标定参数（从IMU到相机）
      Eigen::Matrix3d R_ItoC = state->_calib_IMUtoCAM.at(feat.second->_anchor_cam_id)->Rot();
      Eigen::Vector3d p_IinC = state->_calib_IMUtoCAM.at(feat.second->_anchor_cam_id)->pos();
      // 锚点位姿（IMU在全局坐标系中的朝向和位置）
      Eigen::Matrix3d R_GtoI = state->_clones_IMU.at(feat.second->_anchor_clone_timestamp)->Rot();
      Eigen::Vector3d p_IinG = state->_clones_IMU.at(feat.second->_anchor_clone_timestamp)->pos();
      // 将相对表示的特征点转换为全局坐标系下的位置 (p_FinG)
      p_FinG = R_GtoI.transpose() * R_ItoC.transpose() * (feat.second->get_xyz(false) - p_IinC) + p_IinG;
    }
    // 添加到活动特征点3D位置集合
    active_tracks_posinG[feat.second->_featid] = p_FinG;
  }

  // 相机0（cam0）的内参和标定
  std::shared_ptr<Vec> distortion = state->_cam_intrinsics.at(0);      // 畸变参数
  std::shared_ptr<PoseJPL> calibration = state->_calib_IMUtoCAM.at(0); // 外参
  Eigen::Matrix<double, 3, 3> R_ItoC = calibration->Rot();
  Eigen::Matrix<double, 3, 1> p_IinC = calibration->pos();

  // 获取当前IMU克隆位姿状态
  std::shared_ptr<PoseJPL> clone_Ii = state->_clones_IMU.at(active_tracks_time);
  Eigen::Matrix3d R_GtoIi = clone_Ii->Rot();
  Eigen::Vector3d p_IiinG = clone_Ii->pos();

  // 4. 接下来我们可以用全局位置更新变量
  //    同时将特征点投影回当前帧
  for (const auto &feat : active_tracks_posinG) {

    // 暂时跳过在当前帧中未被跟踪的特征点（指cam0）
    // TODO: 是否应该发布那些不在cam0中跟踪的其他特征点？？
    if (feat_uvs_in_cam0.find(feat.first) == feat_uvs_in_cam0.end())
      continue;

    // 计算该特征点在当前帧中的深度
    // 将SLAM特征点和非cam0特征点投影回当前参考系
    // 特征点位置在当前IMU坐标系 (Ii)
    Eigen::Vector3d p_FinIi = R_GtoIi * (feat.second - p_IiinG);
    // 特征点位置在当前相机坐标系 (Ci)
    Eigen::Vector3d p_FinCi = R_ItoC * p_FinIi + p_IinC;
    double depth = p_FinCi(2); // 深度（Z坐标）
    Eigen::Vector2d uv_dist;   // 畸变后的图像坐标 (u_d, v_d)
    // 如果该特征点在cam0中有观测，则使用实际观测值
    if (feat_uvs_in_cam0.find(feat.first) != feat_uvs_in_cam0.end()) {
      uv_dist << (double)feat_uvs_in_cam0.at(feat.first).x, (double)feat_uvs_in_cam0.at(feat.first).y;
    } else {
      // 否则，计算理论投影点 (cam0的非SLAM特征点不会进入这里)
      Eigen::Vector2d uv_norm; // 归一化平面坐标
      uv_norm << p_FinCi(0) / depth, p_FinCi(1) / depth;
      uv_dist = state->_cam_intrinsics_cameras.at(0)->distort_d(uv_norm); // 添加畸变
    }

    // 跳过无效点（深度为负或太小(小于0.1)）
    if (depth < 0.1) {
      continue;
    }

    // 跳过无效点（图像坐标超出图像边界）
    int width = state->_cam_intrinsics_cameras.at(0)->w();
    int height = state->_cam_intrinsics_cameras.at(0)->h();
    if (uv_dist(0) < 0 || (int)uv_dist(0) >= width || uv_dist(1) < 0 || (int)uv_dist(1) >= height) {
      // PRINT_DEBUG("feat %zu -> depth = %.2f | u_d = %.2f | v_d = %.2f\n",(*it2)->featid,depth,uv_dist(0),uv_dist(1));
      continue;
    }

    // 最终构造成(UV-D)向量并存储
    Eigen::Vector3d uvd;
    uvd << uv_dist, depth;
    active_tracks_uvd.insert({feat.first, uvd});
  }
  retri_rT3 = boost::posix_time::microsec_clock::local_time(); // 第三计时点（结束投影过程）

  // 输出计时信息
  PRINT_ALL(CYAN "[RETRI-用时]: %.4f 秒用于三角化（%zu个三角化成功 / 共%zu个活动特征点）\n" RESET,
            (retri_rT2 - retri_rT1).total_microseconds() * 1e-6, total_triangulated, active_feat_linsys_A.size());
  PRINT_ALL(CYAN "[RETRI-用时]: %.4f 秒用于重投影回当前帧\n" RESET, (retri_rT3 - retri_rT2).total_microseconds() * 1e-6);
  PRINT_ALL(CYAN "[RETRI-用时]: 总计 %.4f 秒\n" RESET, (retri_rT3 - retri_rT1).total_microseconds() * 1e-6);
}

/**
 * @brief 获取历史特征跟踪的可视化图像
 *
 * 此函数生成包含特征跟踪历史信息的可视化图像，包括SLAM特征点高亮显示和状态覆盖文本。
 * 若系统未初始化或特征跟踪器不可用，则返回空矩阵。
 */
cv::Mat VioManager::get_historical_viz_image() {

  // 如果状态或特征跟踪器未就绪，则返回空矩阵
  if (state == nullptr || trackFEATS == nullptr)
    return cv::Mat();

  // 构建需要高亮显示的SLAM特征点ID列表
  std::vector<size_t> highlighted_ids;
  for (const auto &feat : state->_features_SLAM) {
    highlighted_ids.push_back(feat.first);
  }

  // 根据状态生成覆盖文本：零速更新或初始化标志
  std::string overlay = (did_zupt_update) ? "zvupt" : "";
  overlay = (!is_initialized_vio) ? "init" : overlay;

  // 获取当前活动特征点的历史跟踪图像
  cv::Mat img_history;
  trackFEATS->display_history(img_history, 255, 255, 0, 255, 255, 255, highlighted_ids, overlay);

  // 如果存在ARUCO跟踪器，叠加其历史跟踪信息
  if (trackARUCO != nullptr) {
    trackARUCO->display_history(img_history, 0, 255, 255, 255, 255, 255, highlighted_ids, overlay);
    // trackARUCO->display_active(img_history, 0, 255, 255, 255, 255, 255, overlay); // 备用激活状态显示
  }

  // 返回最终合成的历史图像
  return img_history;
}

/**
 * @brief 获取SLAM系统估计的特征点3D坐标
 *
 * 此函数提取SLAM特征点在全局坐标系中的3D位置，处理了特征点的不同表示方法（相对/绝对）。
 * 注意：过滤了前4倍最大ARUCO特征数量范围内的ID（保留非ARUCO特征）。
 */
std::vector<Eigen::Vector3d> VioManager::get_features_SLAM() {
  std::vector<Eigen::Vector3d> slam_feats;
  for (auto &f : state->_features_SLAM) {
    // 跳过ARUCO特征点（ID小于4倍最大ARUCO特征数量）
    if ((int)f.first <= 4 * state->_options.max_aruco_features)
      continue;

    // 相对表示法处理（需依赖锚点）
    if (ov_type::LandmarkRepresentation::is_relative_representation(f.second->_feat_representation)) {
      // 断言确保特征点存在有效的锚点相机ID
      assert(f.second->_anchor_cam_id != -1);

      // 获取锚点相机的标定参数：IMU到相机的旋转和平移
      Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->pos();

      // 获取锚点时刻IMU位姿：全局到IMU的旋转，IMU在全局中的位置
      Eigen::Matrix<double, 3, 3> R_GtoI = state->_clones_IMU.at(f.second->_anchor_clone_timestamp)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinG = state->_clones_IMU.at(f.second->_anchor_clone_timestamp)->pos();

      // 特征点转换到全局坐标系：
      // 1. 相机坐标系 -> IMU坐标系
      // 2. IMU坐标系 -> 全局坐标系
      slam_feats.push_back(R_GtoI.transpose() * R_ItoC.transpose() * (f.second->get_xyz(false) - p_IinC) + p_IinG);
    }
    // 绝对表示法直接获取坐标
    else {
      slam_feats.push_back(f.second->get_xyz(false));
    }
  }
  return slam_feats;
}

std::vector<Eigen::Vector3d> VioManager::get_features_ARUCO() {
  std::vector<Eigen::Vector3d> aruco_feats;
  for (auto &f : state->_features_SLAM) {
    if ((int)f.first > 4 * state->_options.max_aruco_features)
      continue;
    if (ov_type::LandmarkRepresentation::is_relative_representation(f.second->_feat_representation)) {
      // Assert that we have an anchor pose for this feature
      assert(f.second->_anchor_cam_id != -1);
      // Get calibration for our anchor camera
      Eigen::Matrix<double, 3, 3> R_ItoC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinC = state->_calib_IMUtoCAM.at(f.second->_anchor_cam_id)->pos();
      // Anchor pose orientation and position
      Eigen::Matrix<double, 3, 3> R_GtoI = state->_clones_IMU.at(f.second->_anchor_clone_timestamp)->Rot();
      Eigen::Matrix<double, 3, 1> p_IinG = state->_clones_IMU.at(f.second->_anchor_clone_timestamp)->pos();
      // Feature in the global frame
      aruco_feats.push_back(R_GtoI.transpose() * R_ItoC.transpose() * (f.second->get_xyz(false) - p_IinC) + p_IinG);
    } else {
      aruco_feats.push_back(f.second->get_xyz(false));
    }
  }
  return aruco_feats;
}
