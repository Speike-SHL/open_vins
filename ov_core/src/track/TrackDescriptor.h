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

#ifndef OV_CORE_TRACK_DESC_H
#define OV_CORE_TRACK_DESC_H

#include "TrackBase.h"

namespace ov_core {

/**
 * @brief 基于描述子的视觉跟踪
 *
 * 这里我们使用描述子匹配来从一帧跟踪特征到下一帧。
 * 我们在时间上和立体对之间进行跟踪，以获得立体约束。
 * 目前我们使用ORB描述子，因为我们发现在计算描述子时它是最快的。
 * 然后根据比率测试和RANSAC拒绝轨迹。
 */
class TrackDescriptor : public TrackBase {

public:
  /**
   * @brief 带配置变量的公共构造函数
   * @param cameras 包含所有相机内参的相机校准对象
   * @param numfeats 我们想要跟踪的特征数量（例如从帧到帧跟踪200个点）
   * @param numaruco aruco标签的最大id，确保我们的非aruco特征从这个值之上开始
   * @param stereo 是否应该进行立体特征跟踪或双目跟踪
   * @param histmethod 应该进行什么类型的直方图预处理（直方图均衡化？）
   * @param fast_threshold FAST检测阈值
   * @param gridx x方向/u方向的网格大小
   * @param gridy y方向/v方向的网格大小
   * @param minpxdist 特征之间需要至少这个像素数的距离
   * @param knnratio 所需的匹配比率（较小的值在匹配期间强制前两个描述子更不同）
   */
  explicit TrackDescriptor(std::unordered_map<size_t, std::shared_ptr<CamBase>> cameras, int numfeats, int numaruco, bool stereo,
                           HistogramMethod histmethod, int fast_threshold, int gridx, int gridy, int minpxdist, double knnratio)
      : TrackBase(cameras, numfeats, numaruco, stereo, histmethod), threshold(fast_threshold), grid_x(gridx), grid_y(gridy),
        min_px_dist(minpxdist), knn_ratio(knnratio) {}

  /**
   * @brief 处理新图像
   * @param message 包含时间戳、图像和相机id
   */
  void feed_new_camera(const CameraData &message) override;

protected:
  /**
   * @brief 处理新的单目图像
   * @param message 包含时间戳、图像和相机id
   * @param msg_id 消息数据向量中的相机索引
   */
  void feed_monocular(const CameraData &message, size_t msg_id);

  /**
   * @brief 处理新的立体图像对
   * @param message 包含时间戳、图像和相机id
   * @param msg_id_left 消息数据向量中的第一个图像索引
   * @param msg_id_right 消息数据向量中的第二个图像索引
   */
  void feed_stereo(const CameraData &message, size_t msg_id_left, size_t msg_id_right);

  /**
   * @brief 在当前图像中检测新特征
   * @param img0 我们将在其上检测特征的图像
   * @param mask0 包含我们不希望在其中提取特征的ROI的面罩
   * @param pts0 提取的关键点向量
   * @param desc0 提取的描述子向量
   * @param ids0 所有新ID的向量
   *
   * 给定一组图像及其当前提取的特征，这将尝试添加新特征。
   * 我们在这里返回所有提取的描述子，因为我们不需要从左到右进行立体跟踪。
   * 当我们在时间上将特征与前一帧的特征匹配时，我们ID的向量将在后面被重写。
   * 查看robust_match()获取匹配。
   */
  void perform_detection_monocular(const cv::Mat &img0, const cv::Mat &mask0, std::vector<cv::KeyPoint> &pts0, cv::Mat &desc0,
                                   std::vector<size_t> &ids0);

  /**
   * @brief 在当前立体对中检测新特征
   * @param img0 我们将在其上检测特征的左图像
   * @param img1 我们将在其上检测特征的右图像
   * @param mask0 包含我们不希望在其中提取特征的ROI的面罩
   * @param mask1 包含我们不希望在其中提取特征的ROI的面罩
   * @param pts0 左侧新关键点向量
   * @param pts1 右侧新关键点向量
   * @param desc0 左侧提取的描述子向量
   * @param desc1 右侧提取的描述子向量
   * @param cam_id0 第一个相机的id
   * @param cam_id1 第二个相机的id
   * @param ids0 左侧所有新ID的向量
   * @param ids1 右侧所有新ID的向量
   *
   * 这做了与perform_detection_monocular()函数相同的逻辑，但我们还强制执行立体约束。
   * 我们还从左到右进行立体匹配，并仅返回在左右两侧都找到的好匹配。
   * 当我们在时间上将特征与前一帧的特征匹配时，我们ID的向量将在后面被重写。
   * 查看robust_match()获取匹配。
   */
  void perform_detection_stereo(const cv::Mat &img0, const cv::Mat &img1, const cv::Mat &mask0, const cv::Mat &mask1,
                                std::vector<cv::KeyPoint> &pts0, std::vector<cv::KeyPoint> &pts1, cv::Mat &desc0, cv::Mat &desc1,
                                size_t cam_id0, size_t cam_id1, std::vector<size_t> &ids0, std::vector<size_t> &ids1);

  /**
   * @brief 在两个关键点+描述子集合之间查找匹配。
   * @param pts0 第一个关键点向量
   * @param pts1 第二个关键点向量
   * @param desc0 第一个描述子向量
   * @param desc1 第二个描述子向量
   * @param id0 第一个相机的id
   * @param id1 第二个相机的id
   * @param matches 我们找到的匹配向量
   *
   * 这将在两组点之间执行"健壮匹配"（较慢但效果很好）。
   * 首先我们进行从1到2和从2到1的简单KNN匹配，然后进行比率检查和对称检查。
   * 原始代码来自opencv示例中的"RobustMatcher"，在匹配中似乎给出了非常好的结果。
   * https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/RobustMatcher.cpp
   */
  void robust_match(const std::vector<cv::KeyPoint> &pts0, const std::vector<cv::KeyPoint> &pts1, const cv::Mat &desc0,
                    const cv::Mat &desc1, size_t id0, size_t id1, std::vector<cv::DMatch> &matches);

  // robust_match函数的辅助函数
  // 原始代码来自opencv示例中的"RobustMatcher"
  // https://github.com/opencv/opencv/blob/master/samples/cpp/tutorial_code/calib3d/real_time_pose_estimation/src/RobustMatcher.cpp
  void robust_ratio_test(std::vector<std::vector<cv::DMatch>> &matches);
  void robust_symmetry_test(std::vector<std::vector<cv::DMatch>> &matches1, std::vector<std::vector<cv::DMatch>> &matches2,
                            std::vector<cv::DMatch> &good_matches);

  // 时间变量
  boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

  // 我们的orb提取器
  cv::Ptr<cv::ORB> orb0 = cv::ORB::create();
  cv::Ptr<cv::ORB> orb1 = cv::ORB::create();

  // 我们的描述子匹配器
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  // 我们FAST网格检测器的参数
  int threshold;
  int grid_x;
  int grid_y;

  // 作为不同提取特征的最小像素距离（"足够远"）
  int min_px_dist;

  // 两个kNN匹配之间的比率，如果该比率大于此阈值
  // 则两个特征太接近，应被认为是模糊/坏匹配
  double knn_ratio;

  // 描述子矩阵
  std::unordered_map<size_t, cv::Mat> desc_last;
};

} // namespace ov_core

#endif /* OV_CORE_TRACK_DESC_H */
