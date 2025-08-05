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

#ifndef OV_CORE_FEATURE_H
#define OV_CORE_FEATURE_H

#include <Eigen/Eigen>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace ov_core {


/**
 * @brief 用于收集测量数据的稀疏特征类
 * 此要素类可用于存储给定要素的所有追踪信息。
 * 每个特征都被赋予一个唯一的ID，并应附带一组功能追踪记录。
 * 有关如何将信息加载至此处以及如何删除特征的详细信息，请参阅FeatureDatabase类。
 */
class Feature {

public:
  /// 特征点的全局唯一ID
  size_t featid;

  /// 这个特征是否应该被删除的标记c
  bool to_delete;

  /**
   * 原始像素坐标观测容器
   * 结构：相机ID -> 该相机下的观测坐标序列
   * 示例：uvs[0] = 相机0的{ (x1,y1), (x2,y2), ... }
   */
  std::unordered_map<size_t, std::vector<Eigen::VectorXf>> uvs;

  /**
   * 归一化平面坐标观测容器
   * 结构：相机ID -> 去畸变后的归一化平面坐标序列
   * 坐标系：z=1的平面坐标 (x/z, y/z, 1)
   */
  std::unordered_map<size_t, std::vector<Eigen::VectorXf>> uvs_norm;

  /**
   * 观测时间戳容器
   * 结构：相机ID -> 观测发生的时间戳序列
   * 关键：时间戳需与uvs/uvs_norm中的坐标严格对应
   */
  std::unordered_map<size_t, std::vector<double>> timestamps;

  /**
   * 锚定相机ID（该特征点的参考坐标系）
   * 默认：初始观测帧所在的相机
   * 特殊值：-1表示尚未初始化锚定帧
   */
  int anchor_cam_id = -1;

  /**
   * 锚定帧的克隆体时间戳
   */
  double anchor_clone_timestamp;

  /// 该特征点在锚定坐标系下 三角化后的位置
  Eigen::Vector3d p_FinA;

  /// 该特征点在全局坐标系下 三角化后的位置
  Eigen::Vector3d p_FinG;

  /**
   * @brief 移除不在指定时间戳上的观测数据。
   *
   * 给定一组有效时间戳，此函数会移除所有未在这些时间发生的测量数据。
   * 通常用于确保我们拥有的测量数据与克隆时间点对齐。
   *
   * @param valid_times 测量数据必须匹配的时间戳向量
   */
  void clean_old_measurements(const std::vector<double> &valid_times);

  /**
   * @brief 删除无效时间点上的观测
   *
   * Given a series of invalid timestamps, this will remove all measurements that have occurred at these times.
   *
   * @param invalid_times Vector of timestamps that our measurements should not
   */
  void clean_invalid_measurements(const std::vector<double> &invalid_times);

  /**
   * @brief 删除早于指定时间戳的所有观测
   *
   * Given a valid timestamp, this will remove all measurements that have occured earlier then this.
   *
   * @param timestamp Timestamps that our measurements must occur after
   */
  void clean_older_measurements(double timestamp);
};

} // namespace ov_core

#endif /* OV_CORE_FEATURE_H */