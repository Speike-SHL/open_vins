/*
 * OpenVINS: 视觉惯性研究开放平台
 * 版权所有 (C) 2018-2023 Patrick Geneva
 * 版权所有 (C) 2018-2023 Guoquan Huang
 * 版权所有 (C) 2018-2023 OpenVINS 贡献者
 * 版权所有 (C) 2018-2019 Kevin Eckenhoff
 *
 * 本程序为自由软件：您可以在自由软件基金会发布的GNU通用公共许可证的条款下
 * 重新分发和/或修改它，无论是许可证第3版，还是（根据您的选择）任何更高版本。
 *
 * 分发本程序是希望它有用，但不提供任何保证；甚至不提供适销性或特定用途
 * 适用性的默示保证。有关更多详细信息，请参见GNU通用公共许可证。
 *
 * 您应该已经收到了GNU通用公共许可证的副本。如果没有，请参见
 * <https://www.gnu.org/licenses/>。
 */

#ifndef OV_CORE_TRACK_BASE_H
#define OV_CORE_TRACK_BASE_H

#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "utils/colors.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace ov_core {

class Feature;
class CamBase;
class FeatureDatabase;

/**
 * @brief 视觉特征跟踪基类
 *
 * 这是我们所有视觉跟踪器的基类。
 * 这里的目标是提供一个通用接口，使所有底层跟踪器都可以简单地隐藏所有复杂性。
 * 我们有一个叫做"特征数据库"的东西，它包含了所有的跟踪信息。
 * 用户可以向这个数据库查询特征，然后可以在MSCKF或基于批处理的设置中使用这些特征。
 * 特征轨迹存储原始（畸变）和去畸变/归一化的值。
 * 目前我们只支持两种相机模型，请参见：undistort_point_brown() 和 undistort_point_fisheye()。
 *
 * @m_class{m-note m-warning}
 *
 * @par 关于多线程支持的说明
 * 对于独立相机的异步多线程特征跟踪有一些支持。
 * 实现过程中的关键假设是用户不会尝试并行跟踪同一个相机，而是调用不同的相机。
 * 例如，如果我有两个相机，我可以按顺序调用feed函数，或者将每个相机分别放入独立的
 * 线程中并等待它们返回。@ref currid 是原子的，允许多个线程无问题地访问它，
 * 并确保所有特征都有唯一的id值。我们还有用于访问校准和先前图像和轨迹的互斥锁
 * （在可视化期间使用）。需要注意的是，如果线程调用可视化，它可能会挂起，或者
 * feed线程可能会挂起，这是由于获取特定相机id/feed的互斥锁。
 *
 * 这个基类还处理了可视化的大部分繁重工作，但子类可以重写这个功能并执行
 * 自己的逻辑（例如TrackAruco有自己的可视化逻辑）。
 * 这种可视化需要访问先前的图像和它们的轨迹，因此在多线程情况下必须同步。
 * 这不应该影响性能，但高频率的可视化调用可能会对性能产生负面影响。
 */
class TrackBase {

public:
  /**
   * @brief 期望的图像预处理方法。
   */
  enum HistogramMethod { NONE, HISTOGRAM, CLAHE };

  /**
   * @brief 带配置变量的公共构造函数
   * @param cameras 包含所有相机内参的相机校准对象
   * @param numfeats 我们想要跟踪的特征数量（例如从帧到帧跟踪200个点）
   * @param numaruco aruco标签的最大id，确保我们的非aruco特征从这个值之上开始
   * @param stereo 是否应该进行立体特征跟踪或双目跟踪
   * @param histmethod 应该进行什么类型的直方图预处理（直方图均衡化？）
   */
  TrackBase(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
            HistogramMethod histmethod);

  virtual ~TrackBase() {}

  /**
   * @brief 处理新图像
   * @param message 包含时间戳、图像和相机id
   */
  virtual void feed_new_camera(const CameraData &message) = 0;

  /**
   * @brief 显示在最后一张图像中提取的特征
   * @param img_out 我们将在其上叠加特征的图像
   * @param r1,g1,b1 第一种绘制颜色
   * @param r2,g2,b2 第二种绘制颜色
   * @param overlay 替换屏幕左上角正常"cam0"的文本叠加
   */
  virtual void display_active(cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::string overlay = "");

  /**
   * @brief 为每个特征显示"轨迹"（即其历史）
   * @param img_out 我们将在其上叠加特征的图像
   * @param r1,g1,b1 第一种绘制颜色
   * @param r2,g2,b2 第二种绘制颜色
   * @param highlighted 我们希望突出显示的唯一id（例如slam特征）
   * @param overlay 替换屏幕左上角正常"cam0"的文本叠加
   */
  virtual void display_history(cv::Mat &img_out, int r1, int g1, int b1, int r2, int g2, int b2, std::vector<size_t> highlighted = {},
                               std::string overlay = "");

  /**
   * @brief 获取包含所有跟踪信息的特征数据库
   * @return 可以查询特征的FeatureDatabase指针
   */
  std::shared_ptr<FeatureDatabase> get_feature_database() { return database; }

  /**
   * @brief 将一个正在跟踪的特征的ID更改为另一个。
   *
   * 如果您检测到与旧帧的闭环，此函数会很有帮助。
   * 然后可以更改活动特征的id以匹配旧特征id！
   *
   * @param id_old 我们想要更改的旧id
   * @param id_new 我们想要将旧id更改为的新id
   */
  void change_feat_id(size_t id_old, size_t id_new);

  /// 获取最后一帧中活动特征的getter方法（每个相机的观测）
  std::unordered_map<size_t, std::vector<cv::KeyPoint>> get_last_obs() {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    return pts_last;
  }

  /// 获取最后一帧中活动特征的getter方法（每个相机id）
  std::unordered_map<size_t, std::vector<size_t>> get_last_ids() {
    std::lock_guard<std::mutex> lckv(mtx_last_vars);
    return ids_last;
  }

  /// 获取活动特征数量的getter方法
  int get_num_features() { return num_features; }

  /// 设置活动特征数量的setter方法
  void set_num_features(int _num_features) { num_features = _num_features; }

protected:
  /// 包含所有校准的相机对象
  std::unordered_map<size_t, std::shared_ptr<CamBase>> camera_calib;

  /// 包含我们所有当前特征的数据库
  std::shared_ptr<FeatureDatabase> database;

  /// 我们是否为鱼眼模型
  std::map<size_t, bool> camera_fisheye;

  /// 我们应该尝试从帧到帧跟踪的特征数量
  int num_features;

  /// 对于多相机，我们是否应该使用双目跟踪或立体跟踪
  bool use_stereo;

  /// 我们应该使用什么直方图均衡化方法来预处理图像？
  HistogramMethod histogram_method;

  /// 我们最后一组图像存储的互斥锁（img_last、pts_last和ids_last）
  std::vector<std::mutex> mtx_feeds;

  /// 用于编辑*_last变量的互斥锁
  std::mutex mtx_last_vars;

  /// 最后一组图像（使用map使所有跟踪器以相同顺序渲染）
  std::map<size_t, cv::Mat> img_last;

  /// 最后一组图像面罩（使用map使所有跟踪器以相同顺序渲染）
  std::map<size_t, cv::Mat> img_mask_last;

  /// 最后一组跟踪点
  std::unordered_map<size_t, std::vector<cv::KeyPoint>> pts_last;

  /// 数据库中每个当前特征的ID集合
  std::unordered_map<size_t, std::vector<size_t>> ids_last;

  /// 此跟踪器的主要ID（原子操作以允许多线程）
  std::atomic<size_t> currid;

  // 时间变量（大多数子类使用这些...）
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_BASE_H */
