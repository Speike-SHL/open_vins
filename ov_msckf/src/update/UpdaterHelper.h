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

#ifndef OV_MSCKF_UPDATER_HELPER_H
#define OV_MSCKF_UPDATER_HELPER_H

#include <Eigen/Eigen>
#include <memory>
#include <unordered_map>

#include "types/LandmarkRepresentation.h"

namespace ov_type {
class Type;
} // namespace ov_type

namespace ov_msckf {

class State;

/**
 * @brief 为我们的更新器提供辅助函数的类。
 *
 * 可以计算单个特征表示的雅可比矩阵。
 * 这将根据我们状态所处的表示类型来创建雅可比矩阵。
 * 如果我们使用锚点（anchor）表示，则还会有关于锚点状态的额外雅可比矩阵。
 * 还包含诸如零空间投影和完整雅可比矩阵构建等函数。
 * 推导过程请参考 @ref update-feat 页面，其中有详细的方程推导。
 *
 */
class UpdaterHelper {
public:
  /**
   * @brief UpdaterHelper 所使用的特征对象，包含所有测量值和均值
   */
  struct UpdaterHelperFeature {

    /// 该特征的唯一ID
    size_t featid;

    /// 该特征被观测到的UV坐标（按相机ID映射）
    std::unordered_map<size_t, std::vector<Eigen::VectorXf>> uvs;

    /// 该特征被观测到的归一化UV坐标（按相机ID映射）
    std::unordered_map<size_t, std::vector<Eigen::VectorXf>> uvs_norm;

    /// 每个UV测量的时间戳（按相机ID映射）
    std::unordered_map<size_t, std::vector<double>> timestamps;

    /// 特征的表示类型
    ov_type::LandmarkRepresentation::Representation feat_representation;

    /// 特征锚定的相机ID，默认第一个观测为锚点
    int anchor_cam_id = -1;

    /// 锚点克隆的时间戳
    double anchor_clone_timestamp = -1;

    /// 该特征在锚点坐标系下的三角化位置
    Eigen::Vector3d p_FinA;

    /// 该特征在锚点坐标系下的三角化位置（第一估计）
    Eigen::Vector3d p_FinA_fej;

    /// 该特征在全局坐标系下的三角化位置
    Eigen::Vector3d p_FinG;

    /// 该特征在全局坐标系下的三角化位置（第一估计）
    Eigen::Vector3d p_FinG_fej;
  };

  /**
   * @brief 获取关于特征表示的特征和状态雅可比矩阵
   *
   * @param[in] state 滤波器系统的状态
   * @param[in] feature 需要获取雅可比矩阵的特征（必须包含特征均值）
   * @param[out] H_f 关于特征误差状态的雅可比矩阵（对于单一深度为3x1，否则为3x3）
   * @param[out] H_x 关于状态的额外雅可比矩阵（例如锚点位姿）
   * @param[out] x_order 额外雅可比矩阵对应的变量顺序（例如锚点位姿）
   */
  static void get_feature_jacobian_representation(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                                  std::vector<Eigen::MatrixXd> &H_x, std::vector<std::shared_ptr<ov_type::Type>> &x_order);

  /**
   * @brief 为单个特征的所有测量构建“堆叠”的雅可比矩阵
   *
   * @param[in] state 滤波器系统的状态
   * @param[in] feature 需要获取雅可比矩阵的特征（必须包含特征均值）
   * @param[out] H_f 关于特征误差状态的雅可比矩阵
   * @param[out] H_x 关于状态的额外雅可比矩阵（例如锚点位姿）
   * @param[out] res 该特征的测量残差
   * @param[out] x_order 额外雅可比矩阵对应的变量顺序（例如锚点位姿）
   */
  static void get_feature_jacobian_full(std::shared_ptr<State> state, UpdaterHelperFeature &feature, Eigen::MatrixXd &H_f,
                                        Eigen::MatrixXd &H_x, Eigen::VectorXd &res, std::vector<std::shared_ptr<ov_type::Type>> &x_order);

  /**
   * @brief 该函数将 H_f 的左零空间投影到线性系统上。
   *
   * 详细原理请参考 @ref update-null 页面。
   * 这是 MSCKF 的零空间投影操作，用于去除对特征状态的依赖。
   * 注意：该操作是**原地**进行的，所有矩阵在函数调用后都会发生变化。
   *
   * @param H_f 需要进行零空间投影的雅可比矩阵 [res = Hx*(x-xhat)+Hf(f-fhat)+n]
   * @param H_x 状态雅可比矩阵
   * @param res 测量残差
   */
  static void nullspace_project_inplace(Eigen::MatrixXd &H_f, Eigen::MatrixXd &H_x, Eigen::VectorXd &res);

  /**
   * @brief 该函数将执行测量压缩操作
   *
   * 详细原理请参考 @ref update-compress 页面。
   * 注意：该操作是**原地**进行的，所有矩阵在函数调用后都会发生变化。
   *
   * @param H_x 状态雅可比矩阵
   * @param res 测量残差
   */
  static void measurement_compress_inplace(Eigen::MatrixXd &H_x, Eigen::VectorXd &res);
};

} // namespace ov_msckf

#endif // OV_MSCKF_UPDATER_HELPER_H
