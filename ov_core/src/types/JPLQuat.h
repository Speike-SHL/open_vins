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

#ifndef OV_TYPE_TYPE_JPLQUAT_H
#define OV_TYPE_TYPE_JPLQUAT_H

#include "Type.h"
#include "utils/quat_ops.h"

namespace ov_type {

/**
 * @brief 实现 JPL 四元数的派生 Type 类
 *
 * 此四元数采用左乘误差状态，并遵循“JPL 约定”。
 * 请参考 quat_ops.h 文件中的实用函数。
 * 建议初学者参考以下资源了解四元数：
 * - http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf
 * - ftp://naif.jpl.nasa.gov/pub/naif/misc/Quaternion_White_Paper/Quaternions_White_Paper.pdf
 *
 *
 * 在与其他旋转格式互相转换时需特别注意边界情况。
 * 所有公式均基于以下技术报告 @cite Trawny2005TR :
 *
 * > Trawny, Nikolas, and Stergios I. Roumeliotis. "Indirect Kalman filter for 3D attitude estimation."
 * > University of Minnesota, Dept. of Comp. Sci. & Eng., Tech. Rep 2 (2005): 2005.
 * > http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf
 *
 * @section jplquat_define JPL 四元数定义
 *
 * 我们将四元数定义为以下线性组合：
 * @f[
 *  \bar{q} = q_4+q_1\mathbf{i}+q_2\mathbf{j}+q_3\mathbf{k}
 * @f]
 * 其中 i, j, k 定义如下：
 * @f[
 *  \mathbf{i}^2=-1~,~\mathbf{j}^2=-1~,~\mathbf{k}^2=-1
 * @f]
 * @f[
 *  -\mathbf{i}\mathbf{j}=\mathbf{j}\mathbf{i}=\mathbf{k}
 *  ~,~
 *  -\mathbf{j}\mathbf{k}=\mathbf{k}\mathbf{j}=\mathbf{i}
 *  ~,~
 *  -\mathbf{k}\mathbf{i}=\mathbf{i}\mathbf{k}=\mathbf{j}
 * @f]
 * 如 @cite Trawny2005TR 所述，这与 Hamilton 表示法不同，遵循“JPL 提议的标准约定”。
 * q_4 是四元数的“标量”部分，q_1、q_2、q_3 是“向量”部分。
 * 我们将 4x1 向量分为如下约定：
 * @f[
 *  \bar{q} = \begin{bmatrix}q_1\\q_2\\q_3\\q_4\end{bmatrix} = \begin{bmatrix}\mathbf{q}\\q_4\end{bmatrix}
 * @f]
 * 还需注意，四元数被约束在单位圆上：
 * @f[
 *  |\bar{q}| = \sqrt{\bar{q}^\top\bar{q}} = \sqrt{|\mathbf{q}|^2+q_4^2} = 1
 * @f]
 *
 *
 * @section jplquat_errorstate 误差状态定义
 *
 * 需要注意的是，可以证明左乘四元数误差等价于 SO(3) 误差。
 * 如果需要使用右乘误差，则需要实现不同的类型！
 * 详见 @cite Trawny2005TR 的公式 (71)。
 * 具体如下：
 * \f{align*}{
 * {}^{I}_G\bar{q} &\simeq \begin{bmatrix} \frac{1}{2} \delta \boldsymbol{\theta} \\ 1 \end{bmatrix} \otimes {}^{I}_G\hat{\bar{q}}
 * \f}
 * 等价于：
 * \f{align*}{
 * {}^{I}_G \mathbf{R} &\simeq \exp(-\delta \boldsymbol{\theta}) {}^{I}_G \hat{\mathbf{R}} \\
 * &\simeq (\mathbf{I} - \lfloor \delta \boldsymbol{\theta} \rfloor) {}^{I}_G \hat{\mathbf{R}} \\
 * \f}
 *
 */
class JPLQuat : public Type {

public:
  JPLQuat() : Type(3) {
    Eigen::Vector4d q0 = Eigen::Vector4d::Zero();
    q0(3) = 1.0;
    set_value_internal(q0);
    set_fej_internal(q0);
  }

  ~JPLQuat() {}

  /**
   * @brief 通过将当前四元数与由小轴角扰动构建的四元数左乘，实现更新操作。
   *
   * @f[
   * \bar{q}=norm\Big(\begin{bmatrix} \frac{1}{2} \delta \boldsymbol{\theta}_{dx} \\ 1 \end{bmatrix}\Big) \otimes \hat{\bar{q}}
   * @f]
   *
   * @param dx 扰动四元数的轴角表示
   */
  void update(const Eigen::VectorXd &dx) override {

    assert(dx.rows() == _size);

    // Build perturbing quaternion
    Eigen::Matrix<double, 4, 1> dq;
    dq << .5 * dx, 1.0;
    dq = ov_core::quatnorm(dq);

    // Update estimate and recompute R
    set_value(ov_core::quat_multiply(dq, _value));
  }

  /**
   * @brief 设置估计值并重新计算内部旋转矩阵
   * @param new_value 四元数估计的新值（JPL 四元数，格式为 x,y,z,w）
   */
  void set_value(const Eigen::MatrixXd &new_value) override { set_value_internal(new_value); }

  /**
   * @brief 设置第一估计值（fej）并重新计算第一估计的旋转矩阵
   * @param new_value 四元数估计的新值（JPL 四元数，格式为 x,y,z,w）
   */
  void set_fej(const Eigen::MatrixXd &new_value) override { set_fej_internal(new_value); }

  std::shared_ptr<Type> clone() override {
    auto Clone = std::shared_ptr<JPLQuat>(new JPLQuat());
    Clone->set_value(value());
    Clone->set_fej(fej());
    return Clone;
  }

  /// 旋转矩阵访问
  Eigen::Matrix<double, 3, 3> Rot() const { return _R; }

  /// 第一估计旋转矩阵访问
  Eigen::Matrix<double, 3, 3> Rot_fej() const { return _Rfej; }

protected:
  // Stores the rotation
  Eigen::Matrix<double, 3, 3> _R;

  // Stores the first-estimate rotation
  Eigen::Matrix<double, 3, 3> _Rfej;

  /**
   * @brief 设置估计值并重新计算内部旋转矩阵
   * @param new_value 四元数估计的新值
   */
  void set_value_internal(const Eigen::MatrixXd &new_value) {

    assert(new_value.rows() == 4);
    assert(new_value.cols() == 1);

    _value = new_value;

    // compute associated rotation
    _R = ov_core::quat_2_Rot(new_value);
  }

  /**
   * @brief 设置第一估计值（fej）并重新计算第一估计的旋转矩阵
   * @param new_value 四元数估计的新值
   */
  void set_fej_internal(const Eigen::MatrixXd &new_value) {

    assert(new_value.rows() == 4);
    assert(new_value.cols() == 1);

    _fej = new_value;

    // compute associated rotation
    _Rfej = ov_core::quat_2_Rot(new_value);
  }
};

} // namespace ov_type

#endif // OV_TYPE_TYPE_JPLQUAT_H
