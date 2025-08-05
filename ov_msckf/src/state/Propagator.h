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

#ifndef OV_MSCKF_STATE_PROPAGATOR_H
#define OV_MSCKF_STATE_PROPAGATOR_H

#include <atomic>
#include <memory>
#include <mutex>

#include "utils/sensor_data.h"

#include "utils/NoiseManager.h"

namespace ov_msckf {

class State;

/**
 * @brief Performs the state covariance and mean propagation using imu measurements
 *
 * We will first select what measurements we need to propagate with.
 * We then compute the state transition matrix at each step and update the state and covariance.
 * For derivations look at @ref propagation page which has detailed equations.
 */
class Propagator {
public:
  /**
   * @brief Default constructor
   * @param noises imu noise characteristics (continuous time)
   * @param gravity_mag Global gravity magnitude of the system (normally 9.81)
   */
  Propagator(NoiseManager noises, double gravity_mag) : _noises(noises), cache_imu_valid(false) {
    _noises.sigma_w_2 = std::pow(_noises.sigma_w, 2);
    _noises.sigma_a_2 = std::pow(_noises.sigma_a, 2);
    _noises.sigma_wb_2 = std::pow(_noises.sigma_wb, 2);
    _noises.sigma_ab_2 = std::pow(_noises.sigma_ab, 2);
    last_prop_time_offset = 0.0;
    _gravity << 0.0, 0.0, gravity_mag;
  }

  /**
   * @brief 存储传入的惯性测量数据
   * @param message 包含时间戳和惯性信息
   * @param oldest_time 可以丢弃早于该时间的测量值（IMU时钟）
   */
  void feed_imu(const ov_core::ImuData &message, double oldest_time = -1) {

    // Append it to our vector
    std::lock_guard<std::mutex> lck(imu_data_mtx);
    imu_data.emplace_back(message);

    // Clean old measurements
    // std::cout << "PROP: imu_data.size() " << imu_data.size() << std::endl;
    clean_old_imu_measurements(oldest_time - 0.10);
  }

  /**
   * @brief This will remove any IMU measurements that are older then the given measurement time
   * @param oldest_time Time that we can discard measurements before (in IMU clock)
   */
  void clean_old_imu_measurements(double oldest_time) {
    if (oldest_time < 0)
      return;
    auto it0 = imu_data.begin();
    while (it0 != imu_data.end()) {
      if (it0->timestamp < oldest_time) {
        it0 = imu_data.erase(it0);
      } else {
        it0++;
      }
    }
  }

  /**
   * @brief 使用于 fast propagation 的缓存失效
   */
  void invalidate_cache() { cache_imu_valid = false; }

  /**
   * @brief Propagate state up to given timestamp and then clone
   *
   * This will first collect all imu readings that occured between the
   * *current* state time and the new time we want the state to be at.
   * If we don't have any imu readings we will try to extrapolate into the future.
   * After propagating the mean and covariance using our dynamics,
   * We clone the current imu pose as a new clone in our state.
   *
   * @param state Pointer to state
   * @param timestamp Time to propagate to and clone at (CAM clock frame)
   */
  void propagate_and_clone(std::shared_ptr<State> state, double timestamp);

  /**
   * @brief 获取在给定时间戳下的状态及其协方差
   *
   * 该函数可用于预测“未来”某一时刻的状态，而无需实际推进主状态。
   * 它会对当前IMU状态及其协方差矩阵进行克隆并传播。
   * 通常用于在两次更新之间提供高频率的位姿估计。
   *
   * @param state 指向状态的指针
   * @param timestamp 需要传播到的时间（IMU时钟）
   * @param state_plus 传播后的状态（q_GtoI, p_IinG, v_IinI, w_IinI）
   * @param covariance 传播后的协方差（q_GtoI, p_IinG, v_IinI, w_IinI）
   * @return 如果能够成功传播到当前时间戳则返回true
   */
  bool fast_state_propagate(std::shared_ptr<State> state, double timestamp, Eigen::Matrix<double, 13, 1> &state_plus,
                            Eigen::Matrix<double, 12, 12> &covariance);

  /**
   * @brief 辅助函数，根据当前IMU数据，选择两个时间点之间的IMU读数。
   *
   * 这将创建我们将用于积分的测量值，并在末尾添加一个额外的测量值。
   * 我们使用 @ref interpolate_data() 函数在积分开始和结束时“截断”IMU读数。
   * 传入的时间戳应已考虑时间偏移值。
   *
   * @param imu_data 我们将从中选择测量值的IMU数据
   * @param time0 开始时间戳
   * @param time1 结束时间戳
   * @param warn 如果IMU数据不足以传播时是否发出警告（例如，快速传播会收到警告）
   * @return 测量值的向量（如果我们可以计算它们）
   * @note 但是如果 time0 在所有的 imu 数据前面，则并不会在 time0 处进行插值？
   */
  static std::vector<ov_core::ImuData> select_imu_readings(const std::vector<ov_core::ImuData> &imu_data, double time0, double time1,
                                                           bool warn = true);

  /**
   * @brief 在线性插值两个IMU消息之间的便捷辅助函数。
   *
   * 应优先使用此函数，而不是简单地“截断”包围相机时间的IMU消息。
   * 如果IMU频率较低，使用该函数能获得更好的时间对齐效果，也可以尝试更高阶/样条插值。
   *
   * @param imu_1 插值区间起始的IMU数据
   * @param imu_2 插值区间结束的IMU数据
   * @param timestamp 需要插值到的时间戳
   */
  static ov_core::ImuData interpolate_data(const ov_core::ImuData &imu_1, const ov_core::ImuData &imu_2, double timestamp) {
    // time-distance lambda
    double lambda = (timestamp - imu_1.timestamp) / (imu_2.timestamp - imu_1.timestamp);
    // PRINT_DEBUG("lambda - %d\n", lambda);
    // interpolate between the two times
    ov_core::ImuData data;
    data.timestamp = timestamp;
    data.am = (1 - lambda) * imu_1.am + lambda * imu_2.am;
    data.wm = (1 - lambda) * imu_1.wm + lambda * imu_2.wm;
    return data;
  }

  /**
   * @brief compute the Jacobians for Dw
   *
   * See @ref analytical_linearization_imu for details.
   * \f{align*}{
   * \mathbf{H}_{Dw,kalibr} & =
   *   \begin{bmatrix}
   *   {}^w\hat{w}_1 \mathbf{I}_3  & {}^w\hat{w}_2\mathbf{e}_2 & {}^w\hat{w}_2\mathbf{e}_3 & {}^w\hat{w}_3 \mathbf{e}_3
   *   \end{bmatrix} \\
   *   \mathbf{H}_{Dw,rpng} & =
   *   \begin{bmatrix}
   *   {}^w\hat{w}_1\mathbf{e}_1 & {}^w\hat{w}_2\mathbf{e}_1 & {}^w\hat{w}_2\mathbf{e}_2 & {}^w\hat{w}_3 \mathbf{I}_3
   *   \end{bmatrix}
   * \f}
   *
   * @param state Pointer to state
   * @param w_uncorrected Angular velocity in a frame with bias and gravity sensitivity removed
   */
  static Eigen::MatrixXd compute_H_Dw(std::shared_ptr<State> state, const Eigen::Vector3d &w_uncorrected);

  /**
   * @brief compute the Jacobians for Da
   *
   * See @ref analytical_linearization_imu for details.
   * \f{align*}{
   * \mathbf{H}_{Da,kalibr} & =
   * \begin{bmatrix}
   *   {}^a\hat{a}_1\mathbf{e}_1 & {}^a\hat{a}_2\mathbf{e}_1 & {}^a\hat{a}_2\mathbf{e}_2 & {}^a\hat{a}_3 \mathbf{I}_3
   * \end{bmatrix} \\
   * \mathbf{H}_{Da,rpng} & =
   * \begin{bmatrix}
   *   {}^a\hat{a}_1 \mathbf{I}_3 &  & {}^a\hat{a}_2\mathbf{e}_2 & {}^a\hat{a}_2\mathbf{e}_3 & {}^a\hat{a}_3\mathbf{e}_3
   * \end{bmatrix}
   * \f}
   *
   * @param state Pointer to state
   * @param a_uncorrected Linear acceleration in gyro frame with bias removed
   */
  static Eigen::MatrixXd compute_H_Da(std::shared_ptr<State> state, const Eigen::Vector3d &a_uncorrected);

  /**
   * @brief compute the Jacobians for Tg
   *
   * See @ref analytical_linearization_imu for details.
   * \f{align*}{
   * \mathbf{H}_{Tg} & =
   *  \begin{bmatrix}
   *  {}^I\hat{a}_1 \mathbf{I}_3 & {}^I\hat{a}_2 \mathbf{I}_3 & {}^I\hat{a}_3 \mathbf{I}_3
   *  \end{bmatrix}
   * \f}
   *
   * @param state Pointer to state
   * @param a_inI Linear acceleration with bias removed
   */
  static Eigen::MatrixXd compute_H_Tg(std::shared_ptr<State> state, const Eigen::Vector3d &a_inI);

protected:
  /**
   * @brief 使用IMU数据将状态向前传播，并计算该区间的噪声协方差和状态转移矩阵。
   *
   * 此函数可以被解析/数值积分或其他状态表示的实现所替换。
   * 该函数包含了我们的状态转移矩阵以及噪声随时间的演化方式。
   * 如果除了IMU之外还有其他状态变量需要演化，也应在此添加。
   * 离散模型推导详见 @ref propagation_discrete 页面。
   * 解析模型推导详见 @ref propagation_analytical 页面。
   *
   * @param state 状态指针
   * @param data_minus 区间起始时刻的IMU读数
   * @param data_plus 区间结束时刻的IMU读数
   * @param F 区间内的状态转移矩阵
   * @param Qd 区间内的离散时间噪声协方差
   */
  void predict_and_compute(std::shared_ptr<State> state, const ov_core::ImuData &data_minus, const ov_core::ImuData &data_plus,
                           Eigen::MatrixXd &F, Eigen::MatrixXd &Qd);

  /**
   * @brief Discrete imu mean propagation.
   *
   * See @ref disc_prop for details on these equations.
   * \f{align*}{
   * \text{}^{I_{k+1}}_{G}\hat{\bar{q}}
   * &= \exp\bigg(\frac{1}{2}\boldsymbol{\Omega}\big({\boldsymbol{\omega}}_{m,k}-\hat{\mathbf{b}}_{g,k}\big)\Delta t\bigg)
   * \text{}^{I_{k}}_{G}\hat{\bar{q}} \\
   * ^G\hat{\mathbf{v}}_{k+1} &= \text{}^G\hat{\mathbf{v}}_{I_k} - {}^G\mathbf{g}\Delta t
   * +\text{}^{I_k}_G\hat{\mathbf{R}}^\top(\mathbf{a}_{m,k} - \hat{\mathbf{b}}_{\mathbf{a},k})\Delta t\\
   * ^G\hat{\mathbf{p}}_{I_{k+1}}
   * &= \text{}^G\hat{\mathbf{p}}_{I_k} + {}^G\hat{\mathbf{v}}_{I_k} \Delta t
   * - \frac{1}{2}{}^G\mathbf{g}\Delta t^2
   * + \frac{1}{2} \text{}^{I_k}_{G}\hat{\mathbf{R}}^\top(\mathbf{a}_{m,k} - \hat{\mathbf{b}}_{\mathbf{a},k})\Delta t^2
   * \f}
   *
   * @param state Pointer to state
   * @param dt Time we should integrate over
   * @param w_hat Angular velocity with bias removed
   * @param a_hat Linear acceleration with bias removed
   * @param new_q The resulting new orientation after integration
   * @param new_v The resulting new velocity after integration
   * @param new_p The resulting new position after integration
   */
  void predict_mean_discrete(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                             Eigen::Vector4d &new_q, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p);

  /**
   * @brief RK4 imu mean propagation.
   *
   * See this wikipedia page on [Runge-Kutta Methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods).
   * We are doing a RK4 method, [this wolfram page](http://mathworld.wolfram.com/Runge-KuttaMethod.html) has the forth order equation
   * defined below. We define function \f$ f(t,y) \f$ where y is a function of time t, see @ref imu_kinematic for the definition of the
   * continuous-time functions.
   *
   * \f{align*}{
   * {k_1} &= f({t_0}, y_0) \Delta t  \\
   * {k_2} &= f( {t_0}+{\Delta t \over 2}, y_0 + {1 \over 2}{k_1} ) \Delta t \\
   * {k_3} &= f( {t_0}+{\Delta t \over 2}, y_0 + {1 \over 2}{k_2} ) \Delta t \\
   * {k_4} &= f( {t_0} + {\Delta t}, y_0 + {k_3} ) \Delta t \\
   * y_{0+\Delta t} &= y_0 + \left( {{1 \over 6}{k_1} + {1 \over 3}{k_2} + {1 \over 3}{k_3} + {1 \over 6}{k_4}} \right)
   * \f}
   *
   * @param state Pointer to state
   * @param dt Time we should integrate over
   * @param w_hat1 Angular velocity with bias removed
   * @param a_hat1 Linear acceleration with bias removed
   * @param w_hat2 Next angular velocity with bias removed
   * @param a_hat2 Next linear acceleration with bias removed
   * @param new_q The resulting new orientation after integration
   * @param new_v The resulting new velocity after integration
   * @param new_p The resulting new position after integration
   */
  void predict_mean_rk4(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat1, const Eigen::Vector3d &a_hat1,
                        const Eigen::Vector3d &w_hat2, const Eigen::Vector3d &a_hat2, Eigen::Vector4d &new_q, Eigen::Vector3d &new_v,
                        Eigen::Vector3d &new_p);

  /**
   * @brief Analytically compute the integration components based on ACI^2
   *
   * See the @ref analytical_prop page and @ref analytical_integration_components for details.
   * For computing Xi_1, Xi_2, Xi_3 and Xi_4 we have:
   *
   * \f{align*}{
   * \boldsymbol{\Xi}_1 & = \mathbf{I}_3 \delta t + \frac{1 - \cos (\hat{\omega} \delta t)}{\hat{\omega}} \lfloor \hat{\mathbf{k}} \rfloor
   * + \left(\delta t  - \frac{\sin (\hat{\omega} \delta t)}{\hat{\omega}}\right) \lfloor \hat{\mathbf{k}} \rfloor^2 \\
   * \boldsymbol{\Xi}_2 & = \frac{1}{2} \delta t^2 \mathbf{I}_3 +
   * \frac{\hat{\omega} \delta t - \sin (\hat{\omega} \delta t)}{\hat{\omega}^2}\lfloor \hat{\mathbf{k}} \rfloor
   * + \left( \frac{1}{2} \delta t^2 - \frac{1  - \cos (\hat{\omega} \delta t)}{\hat{\omega}^2} \right) \lfloor \hat{\mathbf{k}} \rfloor ^2
   * \\ \boldsymbol{\Xi}_3  &= \frac{1}{2}\delta t^2  \lfloor \hat{\mathbf{a}} \rfloor
   * + \frac{\sin (\hat{\omega} \delta t_i) - \hat{\omega} \delta t }{\hat{\omega}^2} \lfloor\hat{\mathbf{a}} \rfloor \lfloor
   * \hat{\mathbf{k}} \rfloor
   * + \frac{\sin (\hat{\omega} \delta t) - \hat{\omega} \delta t \cos (\hat{\omega} \delta t)  }{\hat{\omega}^2}
   * \lfloor \hat{\mathbf{k}} \rfloor\lfloor\hat{\mathbf{a}} \rfloor
   * + \left( \frac{1}{2} \delta t^2 - \frac{1 - \cos (\hat{\omega} \delta t)}{\hat{\omega}^2} \right) 	\lfloor\hat{\mathbf{a}} \rfloor
   * \lfloor \hat{\mathbf{k}} \rfloor ^2
   * + \left(
   * \frac{1}{2} \delta t^2 + \frac{1 - \cos (\hat{\omega} \delta t) - \hat{\omega} \delta t \sin (\hat{\omega} \delta t) }{\hat{\omega}^2}
   *  \right)
   *  \lfloor \hat{\mathbf{k}} \rfloor ^2 \lfloor\hat{\mathbf{a}} \rfloor
   *  + \left(
   *  \frac{1}{2} \delta t^2 + \frac{1 - \cos (\hat{\omega} \delta t) - \hat{\omega} \delta t \sin (\hat{\omega} \delta t) }{\hat{\omega}^2}
   *  \right)  \hat{\mathbf{k}}^{\top} \hat{\mathbf{a}} \lfloor \hat{\mathbf{k}} \rfloor
   *  - \frac{ 3 \sin (\hat{\omega} \delta t) - 2 \hat{\omega} \delta t - \hat{\omega} \delta t \cos (\hat{\omega} \delta t)
   * }{\hat{\omega}^2} \hat{\mathbf{k}}^{\top} \hat{\mathbf{a}} \lfloor \hat{\mathbf{k}} \rfloor ^2  \\
   * \boldsymbol{\Xi}_4 & = \frac{1}{6}\delta
   * t^3 \lfloor\hat{\mathbf{a}} \rfloor
   * + \frac{2(1 - \cos (\hat{\omega} \delta t)) - (\hat{\omega}^2 \delta t^2)}{2 \hat{\omega}^3}
   *  \lfloor\hat{\mathbf{a}} \rfloor \lfloor \hat{\mathbf{k}} \rfloor
   *  + \left(
   *  \frac{2(1- \cos(\hat{\omega} \delta t)) - \hat{\omega} \delta t \sin (\hat{\omega} \delta t)}{\hat{\omega}^3}
   *  \right)
   *  \lfloor \hat{\mathbf{k}} \rfloor\lfloor\hat{\mathbf{a}} \rfloor
   *  + \left(
   *  \frac{\sin (\hat{\omega} \delta t) - \hat{\omega} \delta t}{\hat{\omega}^3} +
   *  \frac{\delta t^3}{6}
   *  \right)
   *  \lfloor\hat{\mathbf{a}} \rfloor \lfloor \hat{\mathbf{k}} \rfloor^2
   *  +
   *  \frac{\hat{\omega} \delta t - 2 \sin(\hat{\omega} \delta t) + \frac{1}{6}(\hat{\omega} \delta t)^3 + \hat{\omega} \delta t
   * \cos(\hat{\omega} \delta t)}{\hat{\omega}^3} \lfloor \hat{\mathbf{k}} \rfloor^2\lfloor\hat{\mathbf{a}} \rfloor
   *  +
   *  \frac{\hat{\omega} \delta t - 2 \sin(\hat{\omega} \delta t) + \frac{1}{6}(\hat{\omega} \delta t)^3 + \hat{\omega} \delta t
   * \cos(\hat{\omega} \delta t)}{\hat{\omega}^3} \hat{\mathbf{k}}^{\top} \hat{\mathbf{a}} \lfloor \hat{\mathbf{k}} \rfloor
   *  +
   *  \frac{4 \cos(\hat{\omega} \delta t) - 4 + (\hat{\omega} \delta t)^2 + \hat{\omega} \delta t \sin(\hat{\omega} \delta t) }
   *  {\hat{\omega}^3}
   *  \hat{\mathbf{k}}^{\top} \hat{\mathbf{a}} \lfloor \hat{\mathbf{k}} \rfloor^2
   * \f}
   *
   * @param state Pointer to state
   * @param dt Time we should integrate over
   * @param w_hat Angular velocity with bias removed
   * @param a_hat Linear acceleration with bias removed
   * @param Xi_sum All the needed integration components, including R_k, Xi_1, Xi_2, Jr, Xi_3, Xi_4 in order
   */
  void compute_Xi_sum(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                      Eigen::Matrix<double, 3, 18> &Xi_sum);

  /**
   * @brief Analytically predict IMU mean based on ACI^2
   *
   * See the @ref analytical_prop page for details.
   *
   * \f{align*}{
   * {}^{I_{k+1}}_G\hat{\mathbf{R}} & \simeq  \Delta \mathbf{R}_k {}^{I_k}_G\hat{\mathbf{R}}  \\
   * {}^G\hat{\mathbf{p}}_{I_{k+1}} & \simeq {}^{G}\hat{\mathbf{p}}_{I_k} + {}^G\hat{\mathbf{v}}_{I_k}\delta t_k  +
   * {}^{I_k}_G\hat{\mathbf{R}}^\top  \Delta \hat{\mathbf{p}}_k - \frac{1}{2}{}^G\mathbf{g}\delta t^2_k \\
   * {}^G\hat{\mathbf{v}}_{I_{k+1}} & \simeq  {}^{G}\hat{\mathbf{v}}_{I_k} + {}^{I_k}_G\hat{\mathbf{R}}^\top + \Delta \hat{\mathbf{v}}_k -
   * {}^G\mathbf{g}\delta t_k
   * \f}
   *
   * @param state Pointer to state
   * @param dt Time we should integrate over
   * @param w_hat Angular velocity with bias removed
   * @param a_hat Linear acceleration with bias removed
   * @param new_q The resulting new orientation after integration
   * @param new_v The resulting new velocity after integration
   * @param new_p The resulting new position after integration
   * @param Xi_sum All the needed integration components, including R_k, Xi_1, Xi_2, Jr, Xi_3, Xi_4
   */
  void predict_mean_analytic(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                             Eigen::Vector4d &new_q, Eigen::Vector3d &new_v, Eigen::Vector3d &new_p, Eigen::Matrix<double, 3, 18> &Xi_sum);

  /**
   * @brief Analytically compute state transition matrix F and noise Jacobian G based on ACI^2
   *
   * This function is for analytical integration of the linearized error-state.
   * This contains our state transition matrix and noise Jacobians.
   * If you have other state variables besides the IMU that evolve you would add them here.
   * See the @ref analytical_linearization page for details on how this was derived.
   *
   * @param state Pointer to state
   * @param dt Time we should integrate over
   * @param w_hat Angular velocity with bias removed
   * @param a_hat Linear acceleration with bias removed
   * @param w_uncorrected Angular velocity in acc frame with bias and gravity sensitivity removed
   * @param new_q The resulting new orientation after integration
   * @param new_v The resulting new velocity after integration
   * @param new_p The resulting new position after integration
   * @param Xi_sum All the needed integration components, including R_k, Xi_1, Xi_2, Jr, Xi_3, Xi_4
   * @param F State transition matrix
   * @param G Noise Jacobian
   */
  void compute_F_and_G_analytic(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                const Eigen::Vector3d &w_uncorrected, const Eigen::Vector3d &a_uncorrected, const Eigen::Vector4d &new_q,
                                const Eigen::Vector3d &new_v, const Eigen::Vector3d &new_p, const Eigen::Matrix<double, 3, 18> &Xi_sum,
                                Eigen::MatrixXd &F, Eigen::MatrixXd &G);

  /**
   * @brief compute state transition matrix F and noise Jacobian G
   *
   * This function is for analytical integration or when using a different state representation.
   * This contains our state transition matrix and noise Jacobians.
   * If you have other state variables besides the IMU that evolve you would add them here.
   * See the @ref error_prop page for details on how this was derived.
   *
   * @param state Pointer to state
   * @param dt Time we should integrate over
   * @param w_hat Angular velocity with bias removed
   * @param a_hat Linear acceleration with bias removed
   * @param w_uncorrected Angular velocity in acc frame with bias and gravity sensitivity removed
   * @param new_q The resulting new orientation after integration
   * @param new_v The resulting new velocity after integration
   * @param new_p The resulting new position after integration
   * @param F State transition matrix
   * @param G Noise Jacobian
   */
  void compute_F_and_G_discrete(std::shared_ptr<State> state, double dt, const Eigen::Vector3d &w_hat, const Eigen::Vector3d &a_hat,
                                const Eigen::Vector3d &w_uncorrected, const Eigen::Vector3d &a_uncorrected, const Eigen::Vector4d &new_q,
                                const Eigen::Vector3d &new_v, const Eigen::Vector3d &new_p, Eigen::MatrixXd &F, Eigen::MatrixXd &G);

  /// Container for the noise values
  NoiseManager _noises;

  /// Our history of IMU messages (time, angular, linear)
  std::vector<ov_core::ImuData> imu_data;
  std::mutex imu_data_mtx;

  /// Gravity vector
  Eigen::Vector3d _gravity;

  // Estimate for time offset at last propagation time
  double last_prop_time_offset = 0.0;
  bool have_last_prop_time_offset = false;

  // Cache of the last fast propagated state
  std::atomic<bool> cache_imu_valid;
  double cache_state_time;
  Eigen::MatrixXd cache_state_est;
  Eigen::MatrixXd cache_state_covariance;
  double cache_t_off;
};

} // namespace ov_msckf

#endif // OV_MSCKF_STATE_PROPAGATOR_H
