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

#ifndef OV_MSCKF_STATE_HELPER_H
#define OV_MSCKF_STATE_HELPER_H

#include <Eigen/Eigen>
#include <memory>

namespace ov_type {
class Type;
} // namespace ov_type

namespace ov_msckf {

class State;

/**
 * @brief 操作 State 及其协方差的辅助类。
 *
 * 通常，该类包含了基于扩展卡尔曼滤波器（EKF）系统的所有核心逻辑。
 * 这里包含了所有更改协方差、以及向状态中添加或移除元素的函数。
 * 所有函数均为静态函数，因此是自包含的，将来可以支持跟踪和更新多个状态。
 * 我们建议您直接查阅该类的代码，以便更清楚地了解我们在每个函数中具体做了什么，以及相关的文档页面。
 */
class StateHelper {

public:
  /**
   * @brief 执行状态协方差的 EKF 传播。
   *
   * 状态的均值应已完成传播，此函数仅将协方差向前推进。
   * 新的状态（order_NEW）应在内存中**连续**存储。
   * 用户只需指定该区块所依赖的子变量。
   * \f[
   * \tilde{\mathbf{x}}' =
   * \begin{bmatrix}
   * \boldsymbol\Phi_1 &
   * \boldsymbol\Phi_2 &
   * \boldsymbol\Phi_3
   * \end{bmatrix}
   * \begin{bmatrix}
   * \tilde{\mathbf{x}}_1 \\
   * \tilde{\mathbf{x}}_2 \\
   * \tilde{\mathbf{x}}_3
   * \end{bmatrix}
   * +
   * \mathbf{n}
   * \f]
   *
   * @param state 状态指针
   * @param order_NEW 需要根据状态转移演化的连续变量
   * @param order_OLD 状态转移中使用的变量顺序
   * @param Phi 状态转移矩阵（尺寸为 order_NEW × order_OLD）
   * @param Q 状态传播过程中的加性噪声矩阵（尺寸为 order_NEW × order_NEW）
   */
  static void EKFPropagation(std::shared_ptr<State> state, const std::vector<std::shared_ptr<ov_type::Type>> &order_NEW,
                             const std::vector<std::shared_ptr<ov_type::Type>> &order_OLD, const Eigen::MatrixXd &Phi,
                             const Eigen::MatrixXd &Q);

  /**
   * @brief 执行状态的 EKF 更新（详见 @ref linear-meas 页面）
   * @param state 状态指针
   * @param H_order 压缩雅可比矩阵中使用的变量顺序
   * @param H 更新测量的压缩雅可比矩阵
   * @param res 更新测量的残差
   * @param R 更新测量的协方差
   */
  static void EKFUpdate(std::shared_ptr<State> state, const std::vector<std::shared_ptr<ov_type::Type>> &H_order, const Eigen::MatrixXd &H,
                        const Eigen::VectorXd &res, const Eigen::MatrixXd &R);

  /**
   * @brief 设置指定状态元素的初始协方差。
   * 同时确保插入正确的交叉协方差项。
   * @param state 状态指针
   * @param covariance 系统状态的协方差矩阵
   * @param order 协方差矩阵的变量顺序
   */
  static void set_initial_covariance(std::shared_ptr<State> state, const Eigen::MatrixXd &covariance,
                                     const std::vector<std::shared_ptr<ov_type::Type>> &order);

  /**
   * @brief 对于给定的一组变量，计算只包含这些变量及其交叉项的较小协方差矩阵。
   *
   * 返回的协方差矩阵仅包含所指定变量的相关项及所有交叉项。
   * 返回矩阵的尺寸等于所有传入变量维度之和。
   * 通常用于更新前的卡方检验（此时无需完整协方差矩阵）。
   *
   * @param state 状态指针
   * @param small_variables 需要获取边缘协方差的变量向量
   * @return 所传变量的边缘协方差矩阵
   */
  static Eigen::MatrixXd get_marginal_covariance(std::shared_ptr<State> state,
                                                 const std::vector<std::shared_ptr<ov_type::Type>> &small_variables);

  /**
   * @brief 获取完整的协方差矩阵。
   *
   * 仅应在仿真过程中使用，因为对该协方差的操作会很慢。
   * 该函数会返回一个副本，因此无法通过此接口修改协方差（这是有意为之）。
   * 如需以编程方式修改协方差，请使用 StateHelper 中的其他接口函数。
   *
   * @param state 状态指针
   * @return 当前状态的协方差矩阵
   */
  static Eigen::MatrixXd get_full_covariance(std::shared_ptr<State> state);

  /**
   * @brief 边缘化一个变量，并正确修改状态中的排序和协方差
   *
   * 此函数可以直接支持任何 Type 变量的边缘化。
   * 目前不支持对子变量/类型的边缘化。
   * 例如，如果你只想边缘化 PoseJPL 的方向部分，这是不支持的。
   * 我们会首先移除与该类型对应的行和列（即执行边缘化）。
   * 然后更新所有类型的 id，以确保它们考虑到协方差矩阵部分缩小后的变化。
   *
   * @param state 状态指针
   * @param marg 需要边缘化的变量指针
   */
  static void marginalize(std::shared_ptr<State> state, std::shared_ptr<ov_type::Type> marg);

  /**
   * @brief 克隆“variable_to_clone”并将其放置在协方差矩阵末尾
   * @param state 状态指针
   * @param variable_to_clone 需要克隆的变量指针
   */
  static std::shared_ptr<ov_type::Type> clone(std::shared_ptr<State> state, std::shared_ptr<ov_type::Type> variable_to_clone);

  /**
   * @brief 初始化新变量到协方差中。
   *
   * 使用 Givens 分解将系统分为更新和初始化部分（因此系统必须为各向同性）。
   * 如果不是各向同性，请先对系统进行白化（TODO：我们应该添加一个辅助函数来实现此功能）。
   * 如果你的 H_L 雅可比已经可逆，直接调用 initialize_invertible() 即可，无需使用本函数。
   * 详细推导请参考 @ref update-delay 页面。
   *
   * @param state 状态指针
   * @param new_variable 需要初始化的变量指针
   * @param H_order 在压缩状态雅可比中变量的顺序
   * @param H_R 初始化测量对 H_order 变量的雅可比
   * @param H_L 初始化测量对新变量的雅可比
   * @param R 初始化测量的协方差（各向同性）
   * @param res 初始化测量的残差
   * @param chi_2_mult 卡方阈值的放大倍数（越大越容易接受更多测量）
   */
  static bool initialize(std::shared_ptr<State> state, std::shared_ptr<ov_type::Type> new_variable,
                         const std::vector<std::shared_ptr<ov_type::Type>> &H_order, Eigen::MatrixXd &H_R, Eigen::MatrixXd &H_L,
                         Eigen::MatrixXd &R, Eigen::VectorXd &res, double chi_2_mult);

  /**
   * @brief 初始化新变量到协方差中（H_L 必须可逆）
   *
   * 详细推导请参考 @ref update-delay 页面。
   * 该函数假设 H_L 可逆（因此为方阵）且噪声为各向同性，仅进行更新操作。
   *
   * @param state 状态指针
   * @param new_variable 需要初始化的变量指针
   * @param H_order 在压缩状态雅可比中变量的顺序
   * @param H_R 初始化测量对 H_order 变量的雅可比
   * @param H_L 初始化测量对新变量的雅可比（需可逆）
   * @param R 初始化测量的协方差
   * @param res 初始化测量的残差
   */
  static void initialize_invertible(std::shared_ptr<State> state, std::shared_ptr<ov_type::Type> new_variable,
                                    const std::vector<std::shared_ptr<ov_type::Type>> &H_order, const Eigen::MatrixXd &H_R,
                                    const Eigen::MatrixXd &H_L, const Eigen::MatrixXd &R, const Eigen::VectorXd &res);

  /**
   * @brief 用当前 IMU 位姿的随机拷贝扩充状态
   *
   * 在传播之后，通常我们会用一个新的克隆（clone）来扩充状态，该克隆对应新的更新时刻。
   * 该扩充操作会克隆 IMU 位姿，并将其添加到状态的克隆映射中。
   * 如果我们在进行时间偏移标定，克隆操作也会成为时间偏移的函数。
   * 时间偏移的逻辑基于 Li 和 Mourikis 的论文 @cite Li2014IJRR。
   *
   * 我们可以将当前克隆在真实 IMU 基准时钟时间下的表达写为：
   * \f{align*}{
   * {}^{I_{t+t_d}}_G\bar{q} &= \begin{bmatrix}\frac{1}{2} {}^{I_{t+\hat{t}_d}}\boldsymbol\omega \tilde{t}_d \\
   * 1\end{bmatrix}\otimes{}^{I_{t+\hat{t}_d}}_G\bar{q} \\
   * {}^G\mathbf{p}_{I_{t+t_d}} &= {}^G\mathbf{p}_{I_{t+\hat{t}_d}} + {}^G\mathbf{v}_{I_{t+\hat{t}_d}}\tilde{t}_d
   * \f}
   * 其中我们假设状态已经传播到当前图像的估计真实成像时刻，
   * \f${}^{I_{t+\hat{t}_d}}\boldsymbol\omega\f$ 是去除偏置后的传播末端角速度。
   * 由于存在较小的误差，为了获得 IMU 基准时钟下的真实成像时刻，我们可以附加一个小的时间偏移误差。
   * 因此，在克隆过程中关于时间偏移的雅可比为：
   * \f{align*}{
   * \frac{\partial {}^{I_{t+t_d}}_G\tilde{\boldsymbol\theta}}{\partial \tilde{t}_d} &= {}^{I_{t+\hat{t}_d}}\boldsymbol\omega \\
   * \frac{\partial {}^G\tilde{\mathbf{p}}_{I_{t+t_d}}}{\partial \tilde{t}_d} &= {}^G\mathbf{v}_{I_{t+\hat{t}_d}}
   * \f}
   *
   * @param state 状态指针
   * @param last_w 克隆时刻的估计角速度（用于估算 imu-cam 时间偏移）
   */
  static void augment_clone(std::shared_ptr<State> state, Eigen::Matrix<double, 3, 1> last_w);

  /**
   * @brief 移除最旧的克隆（clone），如果当前克隆数量超过最大限制
   *
   * 该函数会将最旧的克隆从协方差矩阵中边缘化，并从状态中移除。
   * 主要作为一个辅助函数，可在每次更新后调用。
   * 它会边缘化由 State::margtimestep() 指定的克隆，该函数应返回一个克隆的时间戳。
   *
   * @param state 状态指针
   */
  static void marginalize_old_clone(std::shared_ptr<State> state);

  /**
   * @brief 边缘化不良的 SLAM 特征
   * @param state 状态指针
   */
  static void marginalize_slam(std::shared_ptr<State> state);

private:
  /**
   * 该类中的所有函数均为静态函数。
   * 因此无法创建该类的实例。
   */
  StateHelper() {}
};

} // namespace ov_msckf

#endif // OV_MSCKF_STATE_HELPER_H
