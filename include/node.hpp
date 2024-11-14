#ifndef BEVFUSION_NODE_HPP_
#define BEVFUSION_NODE_HPP_

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include "rclcpp/rclcpp.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <autoware_auto_perception_msgs/msg/detected_object_kinematics.hpp>
#include <autoware_auto_perception_msgs/msg/detected_objects.hpp>
#include <autoware_auto_perception_msgs/msg/object_classification.hpp>
#include <autoware_auto_perception_msgs/msg/shape.hpp>
// #include "perception_utils/perception_utils.hpp"
// #include "tier4_autoware_utils/geometry/geometry.hpp"
// #include "tier4_autoware_utils/math/constants.hpp"
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/twist_with_covariance.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include "model.hpp"
#include "common/visualize.hpp"

#include <tf2/utils.h>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

class BEVFusionNode : public rclcpp::Node {
public:
  explicit BEVFusionNode(const rclcpp::NodeOptions & node_options);
  ~BEVFusionNode(void);

private:

  std::shared_ptr<Model> model_;

  message_filters::Subscriber<sensor_msgs::msg::Image> left_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> lidar_sub_;
  // for lidar only sense
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_subscription_;

  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                          sensor_msgs::msg::PointCloud2> SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;
  std::shared_ptr<Sync> sync_;

  rclcpp::Publisher<autoware_auto_perception_msgs::msg::DetectedObjects>::SharedPtr objects_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr bboxes_markerArray_pub_;


  void callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg);
  void callback_lidar(const sensor_msgs::msg::PointCloud2::ConstSharedPtr lidar_msg);

  uint8_t getSemanticType(const std::string &class_name);
  bool isCarLikeVehicle(const uint8_t label);
  void box3DToDetectedObject(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes,
                             autoware_auto_perception_msgs::msg::DetectedObjects& obj_msg);

  void box3DToBboxesMarkerArray(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, 
		                            visualization_msgs::msg::MarkerArray& bboxes_markerArray, 
				                        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg);
};

#endif  // BEVFUSION_NODE_HPP_
