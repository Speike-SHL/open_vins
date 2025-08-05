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

#ifndef OV_TYPE_TYPE_BASE_H
#define OV_TYPE_TYPE_BASE_H

#include <Eigen/Eigen>
#include <memory>

namespace ov_type {

/**
 * @brief 估计变量的基类
 *
 * 该类用于定义变量的表示和更新方式（例如向量或四元数）。
 * 每个变量由其误差状态大小（error state size）和在协方差矩阵中的位置决定。
 * 我们还要求所有子类型必须实现更新过程（update procedure）。
 */
class Type {

public:
  /**
   * @brief Type 的默认构造函数
   * @param size_ 变量的自由度（即误差状态的大小）
   */
  Type(int size_) { _size = size_; }

  virtual ~Type() {};

  /**
   * @brief 设置用于跟踪变量在滤波器协方差中的位置的id
   * 注意，最小ID为-1，表示该状态不在我们的协方差中。
   * 如果ID大于-1，则表示这是协方差矩阵中的索引位置。
   * @param new_id 与该变量对应的滤波器协方差中的条目
   */
  virtual void set_local_id(int new_id) { _id = new_id; }

  /**
   * @brief 访问变量的id（即其在协方差中的位置）
   */
  int id() { return _id; }

  /**
   * @brief 访问变量的大小（即其误差状态的大小）
   */
  int size() { return _size; }

  /**
   * @brief 由于误差状态的扰动而更新变量
   * @param dx 用于通过定义的“boxplus”操作更新变量的扰动量
   */
  virtual void update(const Eigen::VectorXd &dx) = 0;

  /**
   * @brief 访问变量的估计值
   */
  virtual const Eigen::MatrixXd &value() const { return _value; }

  /**
   * @brief 访问变量的首次估计值
   */
  virtual const Eigen::MatrixXd &fej() const { return _fej; }

  /**
   * @brief 覆盖状态估计值
   * @param new_value 用于覆盖状态估计值的新值
   */
  virtual void set_value(const Eigen::MatrixXd &new_value) {
    assert(_value.rows() == new_value.rows());
    assert(_value.cols() == new_value.cols());
    _value = new_value;
  }

  /**
   * @brief 覆盖首次估计值
   * @param new_value 用于覆盖状态首次估计值的新值
   */
  virtual void set_fej(const Eigen::MatrixXd &new_value) {
    assert(_fej.rows() == new_value.rows());
    assert(_fej.cols() == new_value.cols());
    _fej = new_value;
  }

  /**
   * @brief 创建该变量的克隆
   */
  virtual std::shared_ptr<Type> clone() = 0;

  /**
   * @brief 判断传入的变量是否为子变量
   *
   * 如果传入的变量是当前变量的子变量或当前变量本身，则返回该变量。
   * 否则返回nullptr，表示未找到该变量。
   *
   * @param check 用于与我们的子变量进行比较的Type指针
   * @note 因为PoseJPL类中包含了JPLQuat和Vec子变量，IMU类中包含了更多
   */
  virtual std::shared_ptr<Type> check_if_subvariable(const std::shared_ptr<Type> check) { return nullptr; }

protected:
  /// 状态量的首次估计值
  Eigen::MatrixXd _fej;

  /// 状态量的当前估计值
  Eigen::MatrixXd _value;

  /// Location of error state in covariance
  int _id = -1;

  /// 误差状态的维度
  int _size = -1;
};

} // namespace ov_type

#endif // OV_TYPE_TYPE_BASE_H
