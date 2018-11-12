#pragma once

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/PointCloud2.h>
#include <FullSystem/FullSystem.h>

class DsoNode
{
public:
  DsoNode();
  void spin();

private:
  void callback(const sensor_msgs::ImageConstPtr& msg);

  image_transport::ImageTransport it;
  image_transport::Subscriber sub;
  ros::NodeHandle n;
  ros::Publisher pub;
  ros::Rate rate;
  sensor_msgs::PointCloud2 pub_val;
  int frame_id;
  dso::FullSystem* full_system;
};
