#include <IOWrapper/OutputWrapper/SampleOutputWrapper.h>
#include <cv_bridge/cv_bridge.h>
#include "DsoNode.h"

DsoNode::DsoNode()
  : it(n)
  , pub(n.advertise<sensor_msgs::PointCloud2>("cloud", 1))
  , sub(it.subscribe("video", 1, &DsoNode::DsoNode::callback, this))
  , rate(ros::Rate(10))
  , frame_id(0)
  , full_system(0)
{}

void DsoNode::callback(const sensor_msgs::ImageConstPtr& msg)
{
  auto cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);

  if(dso::setting_fullResetRequested)
    {
      auto wraps = full_system->outputWrapper;
      delete full_system;
      for(auto const& ow : wraps) ow->reset();
      full_system = new dso::FullSystem();
      full_system->linearizeOperation = false;
      full_system->outputWrapper = wraps;
    }

  dso::MinimalImageB min_img(cv_ptr->image.cols, cv_ptr->image.rows, (unsigned char*) cv_ptr->image.data);
  frame_id++;
}

void DsoNode::spin()
{
  while (ros::ok())
    {
      pub.publish(pub_val);
      ros::spinOnce();
      rate.sleep();
    }
}

int main()
{
}
