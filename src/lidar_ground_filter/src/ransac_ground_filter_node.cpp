#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>

using std::placeholders::_1;

class RansacGroundFilterNode : public rclcpp::Node
{
public:
  RansacGroundFilterNode() : Node("ransac_ground_filter_node")
  {
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_link");
    max_iterations_ = this->declare_parameter<int>("max_iterations", 1000);
    distance_threshold_ = this->declare_parameter<double>("distance_threshold", 0.1);
    plane_slope_threshold_ = this->declare_parameter<double>("plane_slope_threshold", 10.0);
    voxel_size_ = this->declare_parameter<double>("voxel_size", 0.1);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_,this);

    // Pub/Sub
    sub_points_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "input/points", rclcpp::SensorDataQoS().keep_last(1), 
      std::bind(&RansacGroundFilterNode::onPointCloud, this, _1));

    pub_no_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("perception/scan/no_ground", 1);
    pub_ground_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("perception/scan/ground", 1);
    
    RCLCPP_INFO(this->get_logger(), "Node initialized. Base Frame: %s", base_frame_.c_str());
  }

private:
  void onPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    sensor_msgs::msg::PointCloud2 cloud_transformed_msg;
    bool transformed = false;
    try {
      if (base_frame_ != msg->header.frame_id) {
        if (!tf_buffer_->canTransform(base_frame_, msg->header.frame_id, msg->header.stamp, rclcpp::Duration::from_seconds(0.2))) {
          RCLCPP_WARN(this->get_logger(), "Impossible de transformer de %s vers %s", msg->header.frame_id.c_str(), base_frame_.c_str());
          return;
        }
        tf_buffer_->transform(*msg, cloud_transformed_msg, base_frame_);
        transformed = true;
      } else {
        cloud_transformed_msg = *msg;
        transformed = true;
      }
    } catch (tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(), "TF Error: %s", ex.what());
      return;
    }

    if (!transformed) return;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(cloud_transformed_msg, *cloud);

    if (cloud->empty()) return;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setLeafSize(voxel_size_, voxel_size_, voxel_size_);
    sor.filter(*cloud_filtered);

    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(max_iterations_);
    seg.setDistanceThreshold(distance_threshold_);
    seg.setInputCloud(cloud_filtered);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.empty()) {
      pub_no_ground_->publish(cloud_transformed_msg);
      return;
    }

    Eigen::Vector3d plane_normal(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
    Eigen::Vector3d z_axis(0.0, 0.0, 1.0);
    
    double angle_rad = std::acos(std::abs(plane_normal.dot(z_axis)) / (plane_normal.norm() * z_axis.norm()));
    double angle_deg = angle_rad * 180.0 / M_PI;

    if (angle_deg > plane_slope_threshold_) {
      RCLCPP_DEBUG(this->get_logger(), "Plane slope too high: %.2f deg", angle_deg);
      pub_no_ground_->publish(cloud_transformed_msg);
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_ground(new pcl::PointCloud<pcl::PointXYZ>);
    
    pcl::ExtractIndices<pcl::PointXYZ> extract;
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers);
    
    extract.setNegative(false);
    extract.filter(*cloud_ground);

    extract.setNegative(true);
    extract.filter(*cloud_no_ground);

    sensor_msgs::msg::PointCloud2 output_ground_msg;
    sensor_msgs::msg::PointCloud2 output_no_ground_msg;

    pcl::toROSMsg(*cloud_ground, output_ground_msg);
    pcl::toROSMsg(*cloud_no_ground, output_no_ground_msg);

    output_ground_msg.header = cloud_transformed_msg.header;
    output_no_ground_msg.header = cloud_transformed_msg.header;

    pub_ground_->publish(output_ground_msg);
    pub_no_ground_->publish(output_no_ground_msg);
  }

  std::string base_frame_;
  int max_iterations_;
  double distance_threshold_;
  double plane_slope_threshold_;
  double voxel_size_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_points_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_no_ground_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RansacGroundFilterNode>());
  rclcpp::shutdown();
  return 0;
}
