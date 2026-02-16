#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>

using std::placeholders::_1;

class EuclideanClusterNode : public rclcpp::Node
{
public:
  EuclideanClusterNode() : Node("euclidean_cluster_node")
  {
    cluster_tolerance_ = this->declare_parameter<double>("cluster_tolerance", 0.2);
    min_cluster_size_ = this->declare_parameter<int>("min_cluster_size", 20);
    max_cluster_size_ = this->declare_parameter<int>("max_cluster_size", 5000);

    sub_points_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "input/no_ground", 10, std::bind(&EuclideanClusterNode::onPointCloud, this, _1));

    pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("perception/scan/cluster3D", 10);
    
    RCLCPP_INFO(this->get_logger(), "Euclidean Cluster Node Initialized");
  }

private:
  void onPointCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty()) return;

    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);

    std::vector<pcl::PointIndices> cluster_indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(cluster_tolerance_); 
    ec.setMinClusterSize(min_cluster_size_);
    ec.setMaxClusterSize(max_cluster_size_);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(cluster_indices);

    visualization_msgs::msg::MarkerArray marker_array;
    int cluster_id = 0;

    for (const auto& indices : cluster_indices)
    {
      pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
      for (const auto& index : indices.indices) {
        cluster->points.push_back(cloud->points[index]);
      }
      
      pcl::PointXYZ min_pt, max_pt;
      pcl::getMinMax3D(*cluster, min_pt, max_pt);

      visualization_msgs::msg::Marker marker;
      marker.header = msg->header;
      marker.ns = "detected_objects";
      marker.id = cluster_id++;
      marker.type = visualization_msgs::msg::Marker::CUBE;
      marker.action = visualization_msgs::msg::Marker::ADD;

      marker.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
      marker.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
      marker.pose.position.z = (min_pt.z + max_pt.z) / 2.0;
      marker.pose.orientation.w = 1.0;

      marker.scale.x = max_pt.x - min_pt.x;
      marker.scale.y = max_pt.y - min_pt.y;
      marker.scale.z = max_pt.z - min_pt.z;

      marker.color.r = 1.0f;
      marker.color.g = 1.0f;
      marker.color.b = 0.0f;
      marker.color.a = 0.5f;

      marker.lifetime = rclcpp::Duration::from_seconds(0.5);
      marker_array.markers.push_back(marker);
    }
    
    pub_markers_->publish(marker_array);
  }

  double cluster_tolerance_;
  int min_cluster_size_;
  int max_cluster_size_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_points_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EuclideanClusterNode>());
  rclcpp::shutdown();
  return 0;
}