#include <memory>
#include <string>
#include <vector>
#include <string>
#include <algorithm>
#include <dirent.h>
#include <time.h>
#include <unistd.h>
#include "node.hpp"


BEVFusionNode::BEVFusionNode(const rclcpp::NodeOptions & node_options)
: Node("bevfusion", node_options){

  model_ = std::make_shared<Model>("install/bevfusion/share/bevfusion/configs/chengdu30000_pillar.yaml");
  
  if (model_->use_camera_) {
    left_sub_.subscribe(this, "/left");
    lidar_sub_.subscribe(this, "/lidar");
    sync_ = std::make_shared<Sync>(SyncPolicy(10), left_sub_, lidar_sub_);

    using std::placeholders::_1;
    using std::placeholders::_2;
    sync_->registerCallback(std::bind(&BEVFusionNode::callback, this, _1, _2));
  } else {
    using std::placeholders::_1;
    lidar_subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/lidar", rclcpp::QoS{1}, std::bind(&BEVFusionNode::callback_lidar, this, _1));
  }

  objects_pub_ = this->create_publisher<autoware_auto_perception_msgs::msg::DetectedObjects>("/detection_objects", rclcpp::QoS{1});
  bboxes_markerArray_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/detection_marker_array", rclcpp::QoS{1});
}

void BEVFusionNode::callback(const sensor_msgs::msg::Image::ConstSharedPtr& left_msg,
                             const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg) {
  std::vector<cv_bridge::CvImagePtr> in_image_ptr_list(model_->num_camera_);
  try {
    auto desired_encoding = sensor_msgs::image_encodings::BGR8;
    in_image_ptr_list[0] = cv_bridge::toCvCopy(left_msg, desired_encoding);
    
  } catch (cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  std::vector<unsigned char*> images;
  for (size_t i = 0; i < model_->num_camera_; i++){
    images.push_back(in_image_ptr_list[i]->image.data);
  }

  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::fromROSMsg<pcl::PointXYZI>(*lidar_msg, cloud);

  float* points_bufer = new float[cloud.points.size() * model_->num_feature_];
  for (size_t i = 0; i < cloud.points.size(); ++i) {
    pcl::PointXYZI p = cloud.points[i];
    points_bufer[i * model_->num_feature_] = p.x;
    points_bufer[i * model_->num_feature_ + 1] = p.y;
    points_bufer[i * model_->num_feature_ + 2] = p.z;
    points_bufer[i * model_->num_feature_ + 3] = p.intensity;
  }

  std::vector<int64_t> shape;
  shape.push_back(cloud.points.size());
  shape.push_back(model_->num_feature_);
  nv::Tensor lidar_points = nv::Tensor::from_data(points_bufer, shape, nv::DataType::Float32, false);
  nv::Tensor lidar_points_device = lidar_points.to_device();
  nv::Tensor lidar_points_half_host = lidar_points_device.to_half().to_host();

  std::vector<bevfusion::head::transbbox::BoundingBox> bboxes;
  model_->doInfer((const unsigned char**)images.data(), lidar_points_half_host, bboxes);

  // print detction results
  std::cout << "predictions size: " << bboxes.size() << std::endl;
  for (int i = 0; i < bboxes.size(); i++){
    std::cout << "[x: " << bboxes[i].position.x;
    std::cout << ", y: " << bboxes[i].position.y;
    std::cout << ", z: " << bboxes[i].position.z + bboxes[i].size.h / 2;
    std::cout << ", w: " << bboxes[i].size.w;
    std::cout << ", l: " << bboxes[i].size.l;
    std::cout << ", h: " << bboxes[i].size.h;
    std::cout << ", rot:" << -bboxes[i].z_rotation - M_PI / 2;
    std::cout << ", score: " << bboxes[i].score;
    std::cout << ", id: " << bboxes[i].id;
    std:: cout << "]" << std::endl;
  }

  // publish autoware object list msgs for tracking
  autoware_auto_perception_msgs::msg::DetectedObjects obj_msg;
  box3DToDetectedObject(bboxes, obj_msg);
  obj_msg.header.stamp.sec = lidar_msg->header.stamp.sec;
  obj_msg.header.stamp.nanosec = lidar_msg->header.stamp.nanosec;
  objects_pub_->publish(obj_msg);

  // publish visualization msgs
  visualization_msgs::msg::MarkerArray bboxes_markerArray;
  bboxes_markerArray.markers.clear();
  box3DToBboxesMarkerArray(bboxes, bboxes_markerArray, lidar_msg);
  bboxes_markerArray_pub_->publish(bboxes_markerArray);

  delete[] points_bufer;
}

void BEVFusionNode::callback_lidar(const sensor_msgs::msg::PointCloud2::ConstSharedPtr lidar_msg) {

  pcl::PointCloud<pcl::PointXYZI> cloud;
  pcl::fromROSMsg<pcl::PointXYZI>(*lidar_msg, cloud);

  float* points_bufer = new float[cloud.points.size() * model_->num_feature_];
  for (size_t i = 0; i < cloud.points.size(); ++i) {
    pcl::PointXYZI p = cloud.points[i];
    points_bufer[i * model_->num_feature_] = p.x;
    points_bufer[i * model_->num_feature_ + 1] = p.y;
    points_bufer[i * model_->num_feature_ + 2] = p.z;
    points_bufer[i * model_->num_feature_ + 3] = p.intensity;
  }

  std::vector<int64_t> shape;
  shape.push_back(cloud.points.size());
  shape.push_back(model_->num_feature_);
  nv::Tensor lidar_points = nv::Tensor::from_data(points_bufer, shape, nv::DataType::Float32, false);
  nv::Tensor lidar_points_device = lidar_points.to_device();
  nv::Tensor lidar_points_half_host = lidar_points_device.to_half().to_host();

  std::vector<bevfusion::head::transbbox::BoundingBox> bboxes;
  model_->doInfer(nullptr, lidar_points_half_host, bboxes);

  // print detction results
  std::cout << "predictions size: " << bboxes.size() << std::endl;
  for (int i = 0; i < bboxes.size(); i++){
    std::cout << "[x: " << bboxes[i].position.x;
    std::cout << ", y: " << bboxes[i].position.y;
    std::cout << ", z: " << bboxes[i].position.z + bboxes[i].size.h / 2;
    std::cout << ", w: " << bboxes[i].size.w;
    std::cout << ", l: " << bboxes[i].size.l;
    std::cout << ", h: " << bboxes[i].size.h;
    std::cout << ", rot:" << -bboxes[i].z_rotation - M_PI / 2;
    std::cout << ", score: " << bboxes[i].score;
    std::cout << ", id: " << bboxes[i].id;
    std:: cout << "]" << std::endl;
  }

  // publish autoware object list msgs for tracking
  autoware_auto_perception_msgs::msg::DetectedObjects obj_msg;
  box3DToDetectedObject(bboxes, obj_msg);
  obj_msg.header.stamp.sec = lidar_msg->header.stamp.sec;
  obj_msg.header.stamp.nanosec = lidar_msg->header.stamp.nanosec;
  objects_pub_->publish(obj_msg);

  // publish visualization msgs
  visualization_msgs::msg::MarkerArray bboxes_markerArray;
  bboxes_markerArray.markers.clear();
  box3DToBboxesMarkerArray(bboxes, bboxes_markerArray, lidar_msg);
  bboxes_markerArray_pub_->publish(bboxes_markerArray);

  delete[] points_bufer;
}


uint8_t BEVFusionNode::getSemanticType(const std::string &class_name) {
  if (class_name == "Car") {
    return autoware_auto_perception_msgs::msg::ObjectClassification::CAR;
  } else if (class_name == "Truck" || class_name == "Medium_Truck" || class_name == "Big_Truck") {
    return autoware_auto_perception_msgs::msg::ObjectClassification::TRUCK;
  } else if (class_name == "Cyclist") {
    return autoware_auto_perception_msgs::msg::ObjectClassification::MOTORCYCLE;
  } else if (class_name == "Pedestrian") {
    return autoware_auto_perception_msgs::msg::ObjectClassification::PEDESTRIAN;
  } else {
    return autoware_auto_perception_msgs::msg::ObjectClassification::UNKNOWN;
  }
}

bool BEVFusionNode::isCarLikeVehicle(const uint8_t label)
{
  return label == autoware_auto_perception_msgs::msg::ObjectClassification::BUS || 
         label == autoware_auto_perception_msgs::msg::ObjectClassification::CAR ||
         label == autoware_auto_perception_msgs::msg::ObjectClassification::TRAILER || 
         label == autoware_auto_perception_msgs::msg::ObjectClassification::TRUCK;
}


void BEVFusionNode::box3DToDetectedObject(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes,
                                          autoware_auto_perception_msgs::msg::DetectedObjects& obj_msg) {
  
  for (int i = 0; i < bboxes.size(); i++){
    autoware_auto_perception_msgs::msg::DetectedObject obj;
    obj.existence_probability = bboxes[i].score;

    // classification
    autoware_auto_perception_msgs::msg::ObjectClassification classification;
    classification.probability = 1.0f;
    if (bboxes[i].id >= 0 && static_cast<size_t>(bboxes[i].id) < model_->class_names_.size()) {
      classification.label = getSemanticType(model_->class_names_[bboxes[i].id]);
    } else {
      classification.label = autoware_auto_perception_msgs::msg::ObjectClassification::UNKNOWN;
      RCLCPP_WARN_STREAM(rclcpp::get_logger("BEVFusion"), "Unexpected label: UNKNOWN is set.");
    }

    if (isCarLikeVehicle(classification.label)) {
      obj.kinematics.orientation_availability =
          autoware_auto_perception_msgs::msg::DetectedObjectKinematics::SIGN_UNKNOWN;
    }
    // if (perception_utils::isCarLikeVehicle(classification.label)) {
    //   obj.kinematics.orientation_availability =
    //       autoware_auto_perception_msgs::msg::DetectedObjectKinematics::SIGN_UNKNOWN;
    // }
    obj.classification.emplace_back(classification);

    geometry_msgs::msg::Point position;
    position.x = bboxes[i].position.x;
    position.y = bboxes[i].position.y;
    position.z = bboxes[i].position.z + bboxes[i].size.h / 2;
    obj.kinematics.pose_with_covariance.pose.position = position;
    tf2::Quaternion orientation;
    orientation.setRPY(0, 0, -bboxes[i].z_rotation - M_PI / 2);
    obj.kinematics.pose_with_covariance.pose.orientation = tf2::toMsg(orientation);
    geometry_msgs::msg::Vector3 dimension;
    dimension.x = bboxes[i].size.w;
    dimension.y = bboxes[i].size.l;
    dimension.z = bboxes[i].size.h;
    obj.shape.dimensions = dimension;
    obj.shape.type = autoware_auto_perception_msgs::msg::Shape::BOUNDING_BOX;
    obj_msg.objects.emplace_back(obj);

    // obj.kinematics.pose_with_covariance.pose.position =
    //     tier4_autoware_utils::createPoint(bboxes[i].position.x, bboxes[i].position.y, bboxes[i].position.z);
    // obj.kinematics.pose_with_covariance.pose.orientation =
    //     tier4_autoware_utils::createQuaternionFromYaw(-bboxes[i].z_rotation - M_PI / 2);
    // obj.shape.type = autoware_auto_perception_msgs::msg::Shape::BOUNDING_BOX;
    // obj.shape.dimensions =
    //     tier4_autoware_utils::createTranslation(bboxes[i].size.w, bboxes[i].size.l, bboxes[i].size.h);
  }
}

void BEVFusionNode::box3DToBboxesMarkerArray(const std::vector<bevfusion::head::transbbox::BoundingBox>& bboxes, 
	                                           visualization_msgs::msg::MarkerArray& bboxes_markerArray, 
					                                   const sensor_msgs::msg::PointCloud2::ConstSharedPtr& lidar_msg){
  for (size_t i = 0; i < bboxes.size(); i++) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = lidar_msg->header.frame_id;
    marker.header.stamp = lidar_msg->header.stamp;
    marker.type = marker.CUBE;
    marker.id = i + 1;
    marker.lifetime = rclcpp::Duration(0, 100000000);
    // 位置
    marker.pose.position.x = bboxes[i].position.x;
    marker.pose.position.y = bboxes[i].position.y;
    marker.pose.position.z = bboxes[i].position.z + bboxes[i].size.h / 2;
    // 大小
    marker.scale.x = bboxes[i].size.w;
    marker.scale.y = bboxes[i].size.l;
    marker.scale.z = bboxes[i].size.h;
    // 旋转
    tf2::Quaternion quat;
    quat.setRPY(0.0, 0.0, -bboxes[i].z_rotation - M_PI / 2);
    marker.pose.orientation.x = quat.x();
    marker.pose.orientation.y = quat.y();
    marker.pose.orientation.z = quat.z();
    marker.pose.orientation.w = quat.w();
    // 颜色
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1;  // 设置透明度
    bboxes_markerArray.markers.push_back(marker);
  }
}

BEVFusionNode::~BEVFusionNode(){
  model_.reset();
}



#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(BEVFusionNode)
