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

#ifndef OV_CORE_FEATURE_DATABASE_H
#define OV_CORE_FEATURE_DATABASE_H

#include <Eigen/Eigen>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace ov_core {

class Feature;

/**
 * @brief 包含我们当前正在跟踪的特征的数据库。
 *
 * 每个视觉跟踪器都有这个数据库，它包含我们正在跟踪的所有特征。
 * 跟踪器在从跟踪中获得新测量时会将信息插入到这个数据库中。
 * 用户然后查询这个数据库以获取可用于更新的特征，并在处理后移除它们。
 *
 *
 * @m_class{m-note m-warning}
 *
 * @par 关于多线程支持的说明
 * 对异步多线程访问有一定的支持。
 * 由于每个特征都是一个指针，直接返回和使用它们不是线程安全的。
 * 因此，为了线程安全，使用每个函数的"remove"标志，它将从这个特征数据库中移除它。
 * 这防止跟踪器添加新测量和编辑特征信息。
 * 例如，如果您正在异步跟踪相机并选择更新状态，那么移除您将在更新中使用的所有特征。
 * 特征跟踪器将在您更新时继续添加特征，这些测量可以在下一个更新步骤中使用！
 *
 */
class FeatureDatabase {

public:
  /**
   * @brief 默认构造函数
   */
  FeatureDatabase() {}

  /**
   * @brief 获取指定的特征
   * @param id 我们想要获取的特征ID
   * @param remove 如果您想从数据库中移除特征，设为true（您需要处理内存释放）
   * @return 特征对象，如果不在数据库中则返回null。
   */
  std::shared_ptr<Feature> get_feature(size_t id, bool remove = false);

  /**
   * @brief 获取指定特征的克隆（指针是线程安全的）
   * @param id 我们想要获取的特征ID
   * @param feat 包含数据的特征
   * @return 如果找到特征则返回true
   */
  bool get_feature_clone(size_t id, Feature &feat);

  /**
   * @brief 更新特征对象
   * @param id 我们将更新的特征ID
   * @param timestamp 此测量发生的时间
   * @param cam_id 此测量来自哪个相机
   * @param u 原始u坐标
   * @param v 原始v坐标
   * @param u_n 去畸变/归一化的u坐标
   * @param v_n 去畸变/归一化的v坐标
   *
   * 这将根据传递的ID更新给定的特征。
   * 如果是我们之前没有见过的ID，它将创建一个新特征。
   */
  void update_feature(size_t id, double timestamp, size_t cam_id, float u, float v, float u_n, float v_n);

  /**
   * @brief 获取没有比指定时间更新的测量的特征。
   *
   * 此函数将返回所有没有在大于指定时间的时刻进行测量的特征。
   * 例如，这可以用来获取没有成功跟踪到最新帧的特征。
   * 返回的所有特征都不会有在大于指定时间的时刻发生的任何测量。
   */
  std::vector<std::shared_ptr<Feature>> features_not_containing_newer(double timestamp, bool remove = false, bool skip_deleted = false);

  /**
   * @brief 获取有比指定时间更早的测量的特征。
   *
   * 这将收集所有在指定时间戳之前发生测量的特征。
   * 例如，我们想要移除所有比滑动窗口中最后一个克隆/状态更早的特征。
   */
  std::vector<std::shared_ptr<Feature>> features_containing_older(double timestamp, bool remove = false, bool skip_deleted = false);

  /**
   * @brief 获取在指定时间有测量的特征。
   *
   * 此函数将返回所有包含指定时间的特征。
   * 这将用于获取在特定克隆/状态时发生的所有特征。
   */
  std::vector<std::shared_ptr<Feature>> features_containing(double timestamp, bool remove = false, bool skip_deleted = false);

  /**
   * @brief 此函数将删除所有已用完的特征。
   *
   * 如果特征无法使用，它仍将保留，因为它不会设置删除标志
   */
  void cleanup();

  /**
   * @brief 此函数将删除所有早于指定时间戳的特征测量
   */
  void cleanup_measurements(double timestamp);

  /**
   * @brief 此函数将删除所有在指定时间戳的特征测量
   */
  void cleanup_measurements_exact(double timestamp);

  /**
   * @brief 返回特征数据库的大小
   */
  size_t size() {
    std::lock_guard<std::mutex> lck(mtx);
    return features_idlookup.size();
  }

  /**
   * @brief 返回内部数据（通常不应使用）
   */
  std::unordered_map<size_t, std::shared_ptr<Feature>> get_internal_data() {
    std::lock_guard<std::mutex> lck(mtx);
    return features_idlookup;
  }

  /**
   * @brief 获取数据库中最早的时间
   */
  double get_oldest_timestamp();

  /**
   * @brief 将使用此数据库的最新特征信息更新传递的数据库。
   */
  void append_new_measurements(const std::shared_ptr<FeatureDatabase> &database);

protected:
  /// 我们映射的互斥锁
  std::mutex mtx;

  /// 允许我们基于ID查询的查找数组
  /// unordered_map<特征ID, 特征指针>
  /// 特征指针内: 
  //     1. 特征ID 
  //     2. unordered_map<相机ID, 观测坐标序列> 
  //     3. unordered_map<相机ID, 归一化平面坐标序列> 
  //     4. unordered_map<相机ID, 观测时间戳序列> 
  std::unordered_map<size_t, std::shared_ptr<Feature>> features_idlookup;
};

} // namespace ov_core

#endif /* OV_CORE_FEATURE_DATABASE_H */