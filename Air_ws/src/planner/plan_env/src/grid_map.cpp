#include "plan_env/grid_map.h"

// #define current_img_ md_.depth_image_[image_cnt_ & 1]
// #define last_img_ md_.depth_image_[!(image_cnt_ & 1)]

namespace {
const unsigned char TOPO_UNKNOWN = 0;
const unsigned char TOPO_FREE = 1;
const unsigned char TOPO_OCCUPIED = 2;

std::vector<cv::Point> topoSkeletonNeighbors(const cv::Mat& skeleton, const cv::Point& p)
{
  std::vector<cv::Point> neighbors;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx)
    {
      if (dx == 0 && dy == 0)
        continue;
      const int nx = p.x + dx;
      const int ny = p.y + dy;
      if (nx < 0 || nx >= skeleton.cols || ny < 0 || ny >= skeleton.rows)
        continue;
      if (skeleton.at<unsigned char>(ny, nx) > 0)
        neighbors.emplace_back(nx, ny);
    }
  return neighbors;
}

int topoSkeletonDegree(const cv::Mat& skeleton, const cv::Point& p)
{
  return static_cast<int>(topoSkeletonNeighbors(skeleton, p).size());
}

Eigen::Vector2d topoPixelToWorld(const MappingParameters& mp, const cv::Point& p)
{
  return Eigen::Vector2d((static_cast<double>(p.x) + 0.5) * mp.resolution_ + mp.map_origin_(0),
                         (static_cast<double>(p.y) + 0.5) * mp.resolution_ + mp.map_origin_(1));
}

cv::Point topoWorldToPixel(const MappingParameters& mp, const Eigen::Vector2d& p)
{
  return cv::Point(static_cast<int>(std::floor((p.x() - mp.map_origin_(0)) * mp.resolution_inv_)),
                   static_cast<int>(std::floor((p.y() - mp.map_origin_(1)) * mp.resolution_inv_)));
}

double topoPathLength(const std::vector<cv::Point>& path, double resolution)
{
  double length = 0.0;
  for (size_t i = 1; i < path.size(); ++i)
  {
    const double dx = static_cast<double>(path[i].x - path[i - 1].x);
    const double dy = static_cast<double>(path[i].y - path[i - 1].y);
    length += std::sqrt(dx * dx + dy * dy) * resolution;
  }
  return length;
}

bool topoNearProtectedPoint(const Eigen::Vector2d& center, double radius)
{
  if (radius <= 0.0)
    return false;

  const Eigen::Vector2d protected_points[] = {
      Eigen::Vector2d(0.0, 16.0), Eigen::Vector2d(-2.0, 16.0),
      Eigen::Vector2d(2.0, 16.0), Eigen::Vector2d(0.0, -16.0),
      Eigen::Vector2d(-2.0, -16.0), Eigen::Vector2d(2.0, -16.0),
      Eigen::Vector2d(0.0, 15.0)};

  for (const auto& point : protected_points)
  {
    if ((center - point).norm() < radius)
      return true;
  }

  return false;
}

bool topoOccupiedNear(const cv::Mat& occupied_img, int x, int y, int radius)
{
  for (int dy = -radius; dy <= radius; ++dy)
    for (int dx = -radius; dx <= radius; ++dx)
    {
      const int nx = x + dx;
      const int ny = y + dy;
      if (nx < 0 || nx >= occupied_img.cols || ny < 0 || ny >= occupied_img.rows)
        continue;
      if (occupied_img.at<unsigned char>(ny, nx) > 0)
        return true;
    }

  return false;
}

void topoThinningIteration(cv::Mat& img, int iter)
{
  cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);

  for (int i = 1; i < img.rows - 1; ++i)
    for (int j = 1; j < img.cols - 1; ++j)
    {
      const unsigned char p1 = img.at<unsigned char>(i, j);
      if (p1 != 1)
        continue;

      const unsigned char p2 = img.at<unsigned char>(i - 1, j);
      const unsigned char p3 = img.at<unsigned char>(i - 1, j + 1);
      const unsigned char p4 = img.at<unsigned char>(i, j + 1);
      const unsigned char p5 = img.at<unsigned char>(i + 1, j + 1);
      const unsigned char p6 = img.at<unsigned char>(i + 1, j);
      const unsigned char p7 = img.at<unsigned char>(i + 1, j - 1);
      const unsigned char p8 = img.at<unsigned char>(i, j - 1);
      const unsigned char p9 = img.at<unsigned char>(i - 1, j - 1);

      const int transitions =
          (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + (p4 == 0 && p5 == 1) +
          (p5 == 0 && p6 == 1) + (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
          (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
      const int neighbors = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

      if (neighbors < 2 || neighbors > 6 || transitions != 1)
        continue;

      if (iter == 0)
      {
        if (p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
          marker.at<unsigned char>(i, j) = 1;
      }
      else
      {
        if (p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
          marker.at<unsigned char>(i, j) = 1;
      }
    }

  img.setTo(0, marker);
}

void topoThinningZhangSuen(cv::Mat& img)
{
  img /= 255;

  cv::Mat prev = cv::Mat::zeros(img.size(), CV_8UC1);
  cv::Mat diff;
  do
  {
    topoThinningIteration(img, 0);
    topoThinningIteration(img, 1);
    cv::absdiff(img, prev, diff);
    img.copyTo(prev);
  } while (cv::countNonZero(diff) > 0);

  img *= 255;
}
}

void GridMap::initMap(ros::NodeHandle &nh)
{
  node_ = nh;
  
  /* get parameter */
  double x_size, y_size, z_size;
  node_.param("grid_map/resolution", mp_.resolution_, -1.0);
  node_.param("grid_map/map_size_x", x_size, -1.0);
  node_.param("grid_map/map_size_y", y_size, -1.0);
  node_.param("grid_map/map_size_z", z_size, -1.0);
  node_.param("grid_map/local_update_range_x", mp_.local_update_range_(0), -1.0);
  node_.param("grid_map/local_update_range_y", mp_.local_update_range_(1), -1.0);
  node_.param("grid_map/local_update_range_z", mp_.local_update_range_(2), -1.0);
  node_.param("grid_map/obstacles_inflation", mp_.obstacles_inflation_, 0.0);
  node_.param("grid_map/obstacles_inflation_z", mp_.obstacles_inflation_z_, 0.0);

  node_.param("grid_map/send_update_range_x", mp_.send_update_range_(0), -1.0);
  node_.param("grid_map/send_update_range_y", mp_.send_update_range_(1), -1.0);
  node_.param("grid_map/send_update_range_z", mp_.send_update_range_(2), -1.0);

  node_.param("grid_map/use_depth_filter", mp_.use_depth_filter_, true);
  node_.param("grid_map/depth_filter_tolerance", mp_.depth_filter_tolerance_, -1.0);
  node_.param("grid_map/depth_filter_maxdist", mp_.depth_filter_maxdist_, -1.0);
  node_.param("grid_map/depth_filter_mindist", mp_.depth_filter_mindist_, -1.0);
  node_.param("grid_map/depth_filter_margin", mp_.depth_filter_margin_, -1);
  node_.param("grid_map/k_depth_scaling_factor", mp_.k_depth_scaling_factor_, -1.0);
  node_.param("grid_map/skip_pixel", mp_.skip_pixel_, -1);

  node_.param("grid_map/p_hit", mp_.p_hit_, 0.70);
  node_.param("grid_map/p_miss", mp_.p_miss_, 0.35);
  node_.param("grid_map/p_min", mp_.p_min_, 0.12);
  node_.param("grid_map/p_max", mp_.p_max_, 0.97);
  node_.param("grid_map/p_occ", mp_.p_occ_, 0.80);
  node_.param("grid_map/min_ray_length", mp_.min_ray_length_, -0.1);
  node_.param("grid_map/max_ray_length", mp_.max_ray_length_, -0.1);

  node_.param("grid_map/visualization_truncate_height", mp_.visualization_truncate_height_, -0.1);
  node_.param("grid_map/virtual_ceil_height", mp_.virtual_ceil_height_, -0.1);
  node_.param("grid_map/virtual_ceil_yp", mp_.virtual_ceil_yp_, -0.1);
  node_.param("grid_map/virtual_ceil_yn", mp_.virtual_ceil_yn_, -0.1);

  node_.param("grid_map/show_occ_time", mp_.show_occ_time_, false);
  node_.param("grid_map/pose_type", mp_.pose_type_, 1);

  node_.param("grid_map/frame_id", mp_.frame_id_, string("world"));
  node_.param("grid_map/local_map_margin", mp_.local_map_margin_, 1);
  node_.param("grid_map/ground_height", mp_.ground_height_, 1.0);   //地图的z轴零点所在位置，
  node_.param("grid_map/odom_depth_timeout", mp_.odom_depth_timeout_, 1.0);
  node_.param("grid_map/self_id", self_id, 0);
  node_.param("grid_map/topo_deadend_enable", topo_deadend_enable_, false);
  node_.param("grid_map/topo_skeleton_fallback_enable", topo_skeleton_fallback_enable_, false);
  node_.param("grid_map/topo_direct_refresh_enable", topo_direct_refresh_enable_, true);
  node_.param("grid_map/topo_deadend_scenario", topo_deadend_scenario_, string(""));
  node_.param("grid_map/topo_deadend_period", topo_deadend_period_, 1.0);
  node_.param("grid_map/topo_direct_refresh_rate", topo_direct_refresh_rate_, 10.0);
  node_.param("grid_map/topo_wall_min_z", topo_wall_min_z_, 0.1);
  node_.param("grid_map/topo_wall_max_z", topo_wall_max_z_, 0.6);
  node_.param("grid_map/topo_obstacle_cloud_min_z", topo_obstacle_cloud_min_z_, 0.15);
  node_.param("grid_map/topo_min_branch_length", topo_min_branch_length_, 1.4);
  node_.param("grid_map/topo_leaf_check_radius", topo_leaf_check_radius_, 0.8);
  node_.param("grid_map/topo_max_leaf_unknown_ratio", topo_max_leaf_unknown_ratio_, 0.25);
  node_.param("grid_map/topo_closure_thickness", topo_closure_thickness_, 0.4);
  node_.param("grid_map/topo_closure_margin", topo_closure_margin_, 0.2);
  node_.param("grid_map/topo_closure_offset", topo_closure_offset_, 0.3);
  node_.param("grid_map/topo_mouth_search_depth", topo_mouth_search_depth_, 4.0);
  node_.param("grid_map/topo_min_mouth_width", topo_min_mouth_width_, 3.0);
  node_.param("grid_map/topo_max_mouth_width", topo_max_mouth_width_, 5.8);
  node_.param("grid_map/topo_min_width_balance", topo_min_width_balance_, 0.35);
  node_.param("grid_map/topo_protected_radius", topo_protected_radius_, 2.5);
  node_.param("grid_map/topo_centerline_max_abs_x", topo_centerline_max_abs_x_, 1.2);
  node_.param("grid_map/topo_max_half_width", topo_max_half_width_, 4.0);
  node_.param("grid_map/topo_min_half_width", topo_min_half_width_, 0.35);
  node_.param("grid_map/topo_min_clearance", topo_min_clearance_, 0.25);
  node_.param("grid_map/topo_occupied_log_threshold", topo_occupied_log_threshold_, -1000.0);
  node_.param("grid_map/topo_observed_wall_depth", topo_observed_wall_depth_, 5.0);
  node_.param("grid_map/topo_observed_wall_front", topo_observed_wall_front_, 1.0);
  node_.param("grid_map/topo_observed_wall_margin", topo_observed_wall_margin_, 1.2);
  node_.param("grid_map/topo_observed_wall_min_support", topo_observed_wall_min_support_, 2);
  node_.param("grid_map/topo_send_update_range_xy", topo_send_update_range_xy_, -1.0);
  node_.param("grid_map/topo_send_update_range_z", topo_send_update_range_z_, -1.0);
  node_.param("grid_map/topo_max_closures", topo_max_closures_, 8);
  topo_next_closure_id_ = 1;
  topo_observed_wall_obstacle_count_ = 0;

  if( mp_.virtual_ceil_height_ - mp_.ground_height_ > z_size)
  {
    mp_.virtual_ceil_height_ = mp_.ground_height_ + z_size;
  }

  mp_.resolution_inv_ = 1 / mp_.resolution_;
  mp_.map_origin_ = Eigen::Vector3d(-x_size / 2.0, -y_size / 2.0, mp_.ground_height_);
  mp_.map_size_ = Eigen::Vector3d(x_size, y_size, z_size);

  mp_.prob_hit_log_ = logit(mp_.p_hit_);
  mp_.prob_miss_log_ = logit(mp_.p_miss_);
  mp_.clamp_min_log_ = logit(mp_.p_min_);
  mp_.clamp_max_log_ = logit(mp_.p_max_);
  mp_.min_occupancy_log_ = logit(mp_.p_occ_);
  mp_.unknown_flag_ = 0.01;
  if (topo_occupied_log_threshold_ < -100.0)
    topo_occupied_log_threshold_ = mp_.clamp_min_log_ + 0.4;
  if (topo_deadend_enable_ &&
      topo_deadend_scenario_ != "u" &&
      topo_deadend_scenario_ != "v" &&
      topo_deadend_scenario_ != "deadend")
  {
    ROS_WARN("[TopoDeadEnd] requested but grid_map/topo_deadend_scenario='%s' is not supported; semantic closures are disabled.",
             topo_deadend_scenario_.c_str());
    topo_deadend_enable_ = false;
  }

  cout << "hit: " << mp_.prob_hit_log_ << endl;
  cout << "miss: " << mp_.prob_miss_log_ << endl;
  cout << "min log: " << mp_.clamp_min_log_ << endl;
  cout << "max: " << mp_.clamp_max_log_ << endl;
  cout << "thresh log: " << mp_.min_occupancy_log_ << endl;

  for (int i = 0; i < 3; ++i)
    mp_.map_voxel_num_(i) = ceil(mp_.map_size_(i) / mp_.resolution_);

  mp_.map_min_boundary_ = mp_.map_origin_;
  mp_.map_max_boundary_ = mp_.map_origin_ + mp_.map_size_;

  // initialize data buffers

  int buffer_size = mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2);

  md_.occupancy_buffer_ = vector<double>(buffer_size, mp_.clamp_min_log_ - mp_.unknown_flag_);
  md_.occupancy_buffer_inflate_ = vector<char>(buffer_size, 0);
  topo_virtual_obstacle_ = vector<char>(buffer_size, 0);
  topo_observed_wall_obstacle_ = vector<char>(buffer_size, 0);
  topo_observed_obstacle_2d_ = vector<char>(mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1), 0);

  md_.count_hit_and_miss_ = vector<short>(buffer_size, 0);
  md_.count_hit_ = vector<short>(buffer_size, 0);
  md_.flag_rayend_ = vector<char>(buffer_size, -1);
  md_.flag_traverse_ = vector<char>(buffer_size, -1);

  md_.raycast_num_ = 0;

  md_.proj_points_cnt = 0;


  // use odometry and point cloud
  indep_cloud_sub_ =
      node_.subscribe<sensor_msgs::PointCloud2>("grid_map/cloud", 10, &GridMap::cloudCallback, this);//drone_$(arg drone_id)_pcl_render_node/cloud"
  indep_odom_sub_ =
      node_.subscribe<nav_msgs::Odometry>("grid_map/odom", 10, &GridMap::odomCallback, this);

  //with template can't compile successfully???
  // broadcast_odom_sub_ = 
      // node_.subscribe("/broadcast_odom2", 10, &GridMap::BroadcastOdomCallback, this);

  lidar_cloud_sub_ =
      node_.subscribe<sensor_msgs::PointCloud2>("grid_map/realcloud", 10, &GridMap::lidarCloudCallback, this);
  lidar_odom_sub_ =
      node_.subscribe<nav_msgs::Odometry>("grid_map/realodom", 10, &GridMap::lidarOdomCallback, this);
  ugv_odom_sub_ =
      node_.subscribe<nav_msgs::Odometry>("grid_map/ugvodom", 10, &GridMap::UGVOdomCallback, this);

  pub_timer_ = node_.createTimer(ros::Duration(0.10), &GridMap::pubCallback, this);
  vis_timer_ = node_.createTimer(ros::Duration(0.11), &GridMap::visCallback, this);

  map_pub_ = node_.advertise<sensor_msgs::PointCloud2>("grid_map/occupancy", 10);
  map_inf_pub_  = node_.advertise<sensor_msgs::PointCloud2>("grid_map/occupancy_inflate", 10);
  map_free_pub_ = node_.advertise<custom_msgs::map_info>("grid_map/occupancy_send", 10);
  if (topo_deadend_enable_)
  {
    topo_marker_pub_ = node_.advertise<visualization_msgs::Marker>("grid_map/topo_deadend_markers", 10);
    topo_virtual_obstacle_pub_ =
        node_.advertise<sensor_msgs::PointCloud2>("grid_map/topo_virtual_obstacle", 10);
    topo_detection_timer_ = node_.createTimer(
        ros::Duration(std::max(0.1, topo_deadend_period_)), &GridMap::topoDeadendCallback, this);
    if (topo_direct_refresh_enable_)
    {
      topo_refresh_timer_ = node_.createTimer(
          ros::Duration(1.0 / std::max(1.0, topo_direct_refresh_rate_)),
          &GridMap::topoRefreshCallback, this);
    }
    ROS_WARN("[TopoDeadEnd] enabled: semantic wall z=[%.2f, %.2f], cloud_z_min=%.2f, period=%.2f s, occ_log_thresh=%.3f",
             topo_wall_min_z_, topo_wall_max_z_, topo_obstacle_cloud_min_z_,
             topo_deadend_period_, topo_occupied_log_threshold_);
  }

  md_.occ_need_update_ = false;
  md_.local_updated_ = false;
  md_.has_first_depth_ = false;
  md_.has_odom_ = false;
  md_.has_cloud_ = false;
  md_.image_cnt_ = 0;
  md_.last_occ_update_time_.fromSec(0);

  md_.fuse_time_ = 0.0;
  md_.update_num_ = 0;
  md_.max_fuse_time_ = 0.0;

  md_.flag_depth_odom_timeout_ = false;
  md_.flag_use_depth_fusion = false;

  // rand_noise_ = uniform_real_distribution<double>(-0.2, 0.2);
  // rand_noise2_ = normal_distribution<double>(0, 0.2);
  // random_device rd;
  // eng_ = default_random_engine(rd());
}

void GridMap::resetBuffer()
{
  Eigen::Vector3d min_pos = mp_.map_min_boundary_;
  Eigen::Vector3d max_pos = mp_.map_max_boundary_;

  resetBuffer(min_pos, max_pos);

  md_.local_bound_min_ = Eigen::Vector3i::Zero();
  md_.local_bound_max_ = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();
}

void GridMap::resetBuffer(Eigen::Vector3d min_pos, Eigen::Vector3d max_pos)
{

  Eigen::Vector3i min_id, max_id;
  posToIndex(min_pos, min_id);
  posToIndex(max_pos, max_id);

  boundIndex(min_id);
  boundIndex(max_id);

  /* reset occ and dist buffer */
  for (int x = min_id(0); x <= max_id(0); ++x)
    for (int y = min_id(1); y <= max_id(1); ++y)
      for (int z = min_id(2); z <= max_id(2); ++z)
      {
        md_.occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;//膨胀地图清空
      }
}

int GridMap::setCacheOccupancy(Eigen::Vector3d pos, int occ)
{
  if (occ != 1 && occ != 0)
    return INVALID_IDX;

  Eigen::Vector3i id;
  posToIndex(pos, id);
  int idx_ctns = toAddress(id);

  md_.count_hit_and_miss_[idx_ctns] += 1;

  if (md_.count_hit_and_miss_[idx_ctns] == 1)
  {
    md_.cache_voxel_.push(id);
  }

  if (occ == 1)
    md_.count_hit_[idx_ctns] += 1;

  return idx_ctns;
}


void GridMap::raycastProcess(const pcl::PointCloud<pcl::PointXYZ> &cloud)
{
  // if (md_.proj_points_.size() == 0)
  if (cloud.size() == 0)
    return;

  ros::Time t1, t2;

  md_.raycast_num_ += 1;
  // cout<<"ray_num "<<int(md_.raycast_num_)<<endl;
  int vox_idx;
  double length;

  // bounding box of updated region
  double min_x = mp_.map_max_boundary_(0);
  double min_y = mp_.map_max_boundary_(1);
  double min_z = mp_.map_max_boundary_(2);

  double max_x = mp_.map_min_boundary_(0);
  double max_y = mp_.map_min_boundary_(1);
  double max_z = mp_.map_min_boundary_(2);

  RayCaster raycaster;
  Eigen::Vector3d half = Eigen::Vector3d(0.5, 0.5, 0.5);
  Eigen::Vector3d ray_pt, pt_w;

  for (const auto&pt:cloud)
  {
    pt_w << pt.x, pt.y, pt.z;

    // set flag for projected point
     length = (pt_w - md_.camera_pos_).norm();
    if (!isInMap(pt_w))
    {
      pt_w = closetPointInMap(pt_w, md_.camera_pos_);

      if (length > mp_.max_ray_length_)
      {
        pt_w = (pt_w - md_.camera_pos_) / length * mp_.max_ray_length_ + md_.camera_pos_;
      }
      vox_idx = setCacheOccupancy(pt_w, 0);
    }
    else if(isBoundUGV(pt_w))
    {
      pt_w = closetPointFromUGV(pt_w, md_.camera_pos_);

      if (length > mp_.max_ray_length_)
      {
        pt_w = (pt_w - md_.camera_pos_) / length * mp_.max_ray_length_ + md_.camera_pos_;
      }
      vox_idx = setCacheOccupancy(pt_w, 0);
    }
    else
    {
      if (length > mp_.max_ray_length_)
      {
        pt_w = (pt_w - md_.camera_pos_) / length * mp_.max_ray_length_ + md_.camera_pos_;
        vox_idx = setCacheOccupancy(pt_w, 0);
      }
      else
      {
        vox_idx = setCacheOccupancy(pt_w, 1);
      }
    }

    max_x = max(max_x, pt_w(0));
    max_y = max(max_y, pt_w(1));
    max_z = max(max_z, pt_w(2));

    min_x = min(min_x, pt_w(0));
    min_y = min(min_y, pt_w(1));
    min_z = min(min_z, pt_w(2));

    // raycasting between camera center and point

    if (vox_idx != INVALID_IDX)
    {
      if (md_.flag_rayend_[vox_idx] == md_.raycast_num_)
      {
        continue;
      }
      else
      {
        md_.flag_rayend_[vox_idx] = md_.raycast_num_;
      }
    }

    raycaster.setInput(pt_w / mp_.resolution_, md_.camera_pos_ / mp_.resolution_);

    while (raycaster.step(ray_pt))
    {
      Eigen::Vector3d tmp = (ray_pt + half) * mp_.resolution_;
      length = (tmp - md_.camera_pos_).norm();

      // if (length < mp_.min_ray_length_) break;

      vox_idx = setCacheOccupancy(tmp, 0);

      if (vox_idx != INVALID_IDX)
      {
        if (md_.flag_traverse_[vox_idx] == md_.raycast_num_)
        {
          break;
        }
        else
        {
          md_.flag_traverse_[vox_idx] = md_.raycast_num_;
        }
      }
    }
  }

  min_x = min(min_x, md_.camera_pos_(0));
  min_y = min(min_y, md_.camera_pos_(1));
  min_z = min(min_z, md_.camera_pos_(2));

  max_x = max(max_x, md_.camera_pos_(0));
  max_y = max(max_y, md_.camera_pos_(1));
  max_z = max(max_z, md_.camera_pos_(2));
  max_z = max(max_z, mp_.ground_height_);

  posToIndex(Eigen::Vector3d(max_x, max_y, max_z), md_.local_bound_max_);
  posToIndex(Eigen::Vector3d(min_x, min_y, min_z), md_.local_bound_min_);
  boundIndex(md_.local_bound_min_);
  boundIndex(md_.local_bound_max_);

  md_.local_updated_ = true;

  // update occupancy cached in queue
  // Eigen::Vector3d local_range_min = md_.camera_pos_ - mp_.local_update_range_;
  // Eigen::Vector3d local_range_max = md_.camera_pos_ + mp_.local_update_range_;


  // std::cout << "cache all: " << md_.cache_voxel_.size() << std::endl;

  while (!md_.cache_voxel_.empty())
  {

    Eigen::Vector3i idx = md_.cache_voxel_.front();
    int idx_ctns = toAddress(idx);
    md_.cache_voxel_.pop();

    double log_odds_update =
        md_.count_hit_[idx_ctns] >= md_.count_hit_and_miss_[idx_ctns] - md_.count_hit_[idx_ctns] ? mp_.prob_hit_log_ : mp_.prob_miss_log_;

    md_.count_hit_[idx_ctns] = md_.count_hit_and_miss_[idx_ctns] = 0;

    // if (log_odds_update >= 0 && md_.occupancy_buffer_[idx_ctns] >= mp_.clamp_max_log_)
    // {
    //   continue;
    // }
    // else if (log_odds_update <= 0 && md_.occupancy_buffer_[idx_ctns] <= mp_.clamp_min_log_)
    // {
    //   md_.occupancy_buffer_[idx_ctns] = mp_.clamp_min_log_;
    //   continue;
    // }

    // bool in_local = idx(0) >= min_id(0) && idx(0) <= max_id(0) && idx(1) >= min_id(1) &&
    //                 idx(1) <= max_id(1) && idx(2) >= min_id(2) && idx(2) <= max_id(2);
    // if (!in_local)
    // {
    //   md_.occupancy_buffer_[idx_ctns] = mp_.clamp_min_log_;
    // }
    bool occupy_ori = md_.occupancy_buffer_[idx_ctns] > mp_.min_occupancy_log_? true : false;
    md_.occupancy_buffer_[idx_ctns] =
        std::min(std::max(md_.occupancy_buffer_[idx_ctns] + log_odds_update, mp_.clamp_min_log_),
                 mp_.clamp_max_log_);
    bool occupy = md_.occupancy_buffer_[idx_ctns] > mp_.min_occupancy_log_? true : false;
    if(occupy_ori != occupy)
    {
      if(occupy_ori == true && occupy == false)//down
        inflateCount(idx, false);
      else
        inflateCount(idx, true);
    }
  }
}

Eigen::Vector3d GridMap::closetPointInMap(const Eigen::Vector3d &pt, const Eigen::Vector3d &camera_pt)
{
  Eigen::Vector3d diff = pt - camera_pt;
  Eigen::Vector3d max_tc = mp_.map_max_boundary_ - camera_pt;
  Eigen::Vector3d min_tc = mp_.map_min_boundary_ - camera_pt;

  double min_t = 1000000;

  for (int i = 0; i < 3; ++i)
  {
    if (fabs(diff[i]) > 0)
    {

      double t1 = max_tc[i] / diff[i];
      if (t1 > 0 && t1 < min_t)
        min_t = t1;

      double t2 = min_tc[i] / diff[i];
      if (t2 > 0 && t2 < min_t)
        min_t = t2;
    }
  }

  return camera_pt + (min_t - 1e-3) * diff;
}

Eigen::Vector3d GridMap::closetPointFromUGV(const Eigen::Vector3d &pt, const Eigen::Vector3d &camera_pt)
{
  Eigen::Vector3d dir = (pt - camera_pt).normalized();

  return pt - 2*mp_.resolution_*dir;
}

void GridMap::inflateCount(const Eigen::Vector3i idx, const bool is_up)//true means  free change to occ
{
  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
  int inf_step_z = ceil(mp_.obstacles_inflation_z_ / mp_.resolution_);
  Eigen::Vector3i index_inf;
  if(is_up)
  {
    for (int x = -inf_step; x <= inf_step; ++x)
      for (int y = -inf_step; y <= inf_step; ++y)
        for (int z = -inf_step_z; z <= inf_step_z; ++z)
        {
          index_inf(0) = idx(0) + x;
          index_inf(1) = idx(1) + y;
          index_inf(2) = idx(2) + z;

          if (!isInMap(index_inf))
            continue;
          
          int idx_inf = toAddress(index_inf);
          md_.occupancy_buffer_inflate_[idx_inf] += 1;
      }
  }
  else
  {
    for (int x = -inf_step; x <= inf_step; ++x)
      for (int y = -inf_step; y <= inf_step; ++y)
        for (int z = -inf_step_z; z <= inf_step_z; ++z)
        {
          index_inf(0) =  idx(0) + x;
          index_inf(1) =  idx(1) + y;
          index_inf(2) =  idx(2) + z;

          if (!isInMap(index_inf))
            continue;

          int idx_inf = toAddress(index_inf);
          md_.occupancy_buffer_inflate_[idx_inf] -= 1;
        }
  }
}

void GridMap::clearAndInflateLocalMap()
{
  /*clear outside local*/
  const int vec_margin = 5;
  // Eigen::Vector3i min_vec_margin = min_vec - Eigen::Vector3i(vec_margin,
  // vec_margin, vec_margin); Eigen::Vector3i max_vec_margin = max_vec +
  // Eigen::Vector3i(vec_margin, vec_margin, vec_margin);

  Eigen::Vector3i min_cut = md_.local_bound_min_ -
                            Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  Eigen::Vector3i max_cut = md_.local_bound_max_ +
                            Eigen::Vector3i(mp_.local_map_margin_, mp_.local_map_margin_, mp_.local_map_margin_);
  boundIndex(min_cut);
  boundIndex(max_cut);

  Eigen::Vector3i min_cut_m = min_cut - Eigen::Vector3i(vec_margin, vec_margin, vec_margin);
  Eigen::Vector3i max_cut_m = max_cut + Eigen::Vector3i(vec_margin, vec_margin, vec_margin);
  boundIndex(min_cut_m);
  boundIndex(max_cut_m);

  // clear data outside the local range

  // for (int x = min_cut_m(0); x <= max_cut_m(0); ++x)
  //   for (int y = min_cut_m(1); y <= max_cut_m(1); ++y)
  //   {

  //     for (int z = min_cut_m(2); z < min_cut(2); ++z)
  //     {
  //       int idx = toAddress(x, y, z);
  //       md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
  //     }

  //     for (int z = max_cut(2) + 1; z <= max_cut_m(2); ++z)
  //     {
  //       int idx = toAddress(x, y, z);
  //       md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
  //     }
  //   }

  // for (int z = min_cut_m(2); z <= max_cut_m(2); ++z)
  //   for (int x = min_cut_m(0); x <= max_cut_m(0); ++x)
  //   {

  //     for (int y = min_cut_m(1); y < min_cut(1); ++y)
  //     {
  //       int idx = toAddress(x, y, z);
  //       md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
  //     }

  //     for (int y = max_cut(1) + 1; y <= max_cut_m(1); ++y)
  //     {
  //       int idx = toAddress(x, y, z);
  //       md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
  //     }
  //   }

  // for (int y = min_cut_m(1); y <= max_cut_m(1); ++y)
  //   for (int z = min_cut_m(2); z <= max_cut_m(2); ++z)
  //   {

  //     for (int x = min_cut_m(0); x < min_cut(0); ++x)
  //     {
  //       int idx = toAddress(x, y, z);
  //       md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
  //     }

  //     for (int x = max_cut(0) + 1; x <= max_cut_m(0); ++x)
  //     {
  //       int idx = toAddress(x, y, z);
  //       md_.occupancy_buffer_[idx] = mp_.clamp_min_log_ - mp_.unknown_flag_;
  //     }
  //   }

  // inflate occupied voxels to compensate robot size

  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
  int inf_step_z = ceil(mp_.obstacles_inflation_z_ / mp_.resolution_);
  vector<Eigen::Vector3i> inf_pts(pow(2 * inf_step + 1, 3));
  // inf_pts.resize(4 * inf_step + 3);
  Eigen::Vector3i inf_pt;

  // clear outdated data
  for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
    for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y)
      for (int z = md_.local_bound_min_(2); z <= md_.local_bound_max_(2); ++z)
      {
        md_.occupancy_buffer_inflate_[toAddress(x, y, z)] = 0;//过时的点云
      }

  // inflate obstacles
  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {

        if (md_.occupancy_buffer_[toAddress(x, y, z)] > mp_.min_occupancy_log_)
        {
          inflatePoint(Eigen::Vector3i(x, y, z), inf_step, inf_pts);

          for (int k = 0; k < (int)inf_pts.size(); ++k)
          {
            inf_pt = inf_pts[k];
            int idx_inf = toAddress(inf_pt);
            if (idx_inf < 0 ||
                idx_inf >= mp_.map_voxel_num_(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2))
            {
              continue;
            }
            md_.occupancy_buffer_inflate_[idx_inf] = 1;
          }
        }
      }

  // add virtual ceiling to limit flight height
  // if (mp_.virtual_ceil_height_ > -0.5) {
  //   int ceil_id = floor((mp_.virtual_ceil_height_ - mp_.map_origin_(2)) * mp_.resolution_inv_) - 1;
  //   for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
  //     for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y) {
  //       md_.occupancy_buffer_inflate_[toAddress(x, y, ceil_id)] = 1;
  //     }
  // }
}

void GridMap::pubCallback(const ros::TimerEvent & /*event*/)
{
  const double send_occupied_log_threshold =
      topo_deadend_enable_ ? std::min(mp_.min_occupancy_log_, topo_occupied_log_threshold_)
                           : mp_.min_occupancy_log_;

  for(int i = 0; i < md_.ugv_odom_buf_.size(); i++)
  {
    if(md_.ugv_odom_buf_[i].drone_id == -1) continue;
    custom_msgs::map_info sendmap;
    std::set<int> active_topo_obstacle_addresses;
    if (topo_deadend_enable_ && !topo_closures_.empty())
    {
      for (const auto& closure : topo_closures_)
      {
        for (const int address : closure.addresses)
          active_topo_obstacle_addresses.insert(address);
      }
    }
   
//double dx = pos(0) - md_.ugv_odom_buf_[i].pos.x()
 
    Eigen::Vector3i min_bound;
    Eigen::Vector3i max_bound; 
    Eigen::Vector3d send_box_max, send_box_min;
    Eigen::Vector3d send_update_range = mp_.send_update_range_;
    if (topo_deadend_enable_)
    {
      if (topo_send_update_range_xy_ > 0.0)
      {
        send_update_range(0) = std::max(send_update_range(0), topo_send_update_range_xy_);
        send_update_range(1) = std::max(send_update_range(1), topo_send_update_range_xy_);
      }
      if (topo_send_update_range_z_ > 0.0)
        send_update_range(2) = std::max(send_update_range(2), topo_send_update_range_z_);
    }
    send_box_max = md_.ugv_odom_buf_[i].pos + send_update_range;
    send_box_min = md_.ugv_odom_buf_[i].pos - send_update_range;

    sendmap.id_from = self_id;
    if(md_.ugv_odom_buf_[i].vel.norm() < 0.2)
    {
      // posToIndex(send_box_max, min_bound);
      // posToIndex(send_box_min, max_bound);      
    }
    else
    {
      double dir = atan2(md_.ugv_odom_buf_[i].vel(1),md_.ugv_odom_buf_[i].vel(0));
      if((dir>=0 && dir<=M_PI/4) || (dir<=0 && dir>=-M_PI/4))
      {
        send_box_max(0) = md_.ugv_odom_buf_[i].pos(0) + (7.0/4)*send_update_range(0);
        send_box_min(0) = md_.ugv_odom_buf_[i].pos(0) - (1.0/4)*send_update_range(0);
        sendmap.id_from = -1;
      }
      else if((dir<=M_PI && dir>=3*M_PI/4) || (dir>=-M_PI && dir<=-3*M_PI/4))
      {
        send_box_max(0) = md_.ugv_odom_buf_[i].pos(0) + (1.0/4)*send_update_range(0);
        send_box_min(0) = md_.ugv_odom_buf_[i].pos(0) - (7.0/4)*send_update_range(0);
        sendmap.id_from = -2;
      }
      else if(dir> M_PI/4 && dir< 3*M_PI/4)
      {
        send_box_max(1) = md_.ugv_odom_buf_[i].pos(1) + (7.0/4)*send_update_range(1);
        send_box_min(1) = md_.ugv_odom_buf_[i].pos(1) - (1.0/4)*send_update_range(1);
        sendmap.id_from = -3;
      }
      else if(dir< -M_PI/4 && dir> -3*M_PI/4)
      {
        send_box_max(1) = md_.ugv_odom_buf_[i].pos(1) + (1.0/4)*send_update_range(1);
        send_box_min(1) = md_.ugv_odom_buf_[i].pos(1) - (7.0/4)*send_update_range(1);
        sendmap.id_from = -4;
      }
    }

    posToIndex(send_box_min, min_bound);
    posToIndex(send_box_max, max_bound);   
    boundIndex(min_bound);
    boundIndex(max_bound);
    sendmap.header.frame_id="world";
    sendmap.header.stamp = ros::Time::now();
    
    sendmap.id_to = i;
    
    sendmap.resolution = mp_.resolution_;
    sendmap.size_x = mp_.map_voxel_num_(0);
    sendmap.size_y = mp_.map_voxel_num_(1);
    sendmap.size_z = mp_.map_voxel_num_(2);
    
    sendmap.origin_x = mp_.map_origin_(0);
    sendmap.origin_y = mp_.map_origin_(1);
    sendmap.origin_z = mp_.map_origin_(2);

    sendmap.update_min_x = min_bound(0);
    sendmap.update_min_y = min_bound(1);
    sendmap.update_min_z = min_bound(2);
    sendmap.update_max_x = max_bound(0);
    sendmap.update_max_y = max_bound(1);
    sendmap.update_max_z = max_bound(2);


    for (int x = min_bound(0); x <= max_bound(0); ++x)
      for (int y = min_bound(1); y <= max_bound(1); ++y)
        for (int z = min_bound(2); z <= max_bound(2); ++z)
        {
          int address = toAddress(x, y, z);
          const bool active_topo_obstacle =
              active_topo_obstacle_addresses.find(address) != active_topo_obstacle_addresses.end();
          if (md_.occupancy_buffer_[address] >= send_occupied_log_threshold ||
              active_topo_obstacle ||
              isTopoObservedWallOccupied(address))
          {
            sendmap.occu_address.push_back(address);
          }

          else if (md_.occupancy_buffer_[address] < mp_.clamp_min_log_)
          {
            sendmap.unknow_address.push_back(address);
          }
          else 
            continue;
      
          // Eigen::Vector3d pos;
          // indexToPos(Eigen::Vector3i(x, y, z), pos);

          // pt.x = pos(0);
          // pt.y = pos(1);
          // pt.z = pos(2);
          // cloud.push_back(pt);  
        }

    size_t topo_extra_obstacle_cells = 0;
    if (topo_deadend_enable_ && !topo_direct_refresh_enable_ && !active_topo_obstacle_addresses.empty())
    {
      const int size_y = mp_.map_voxel_num_(1);
      const int size_z = mp_.map_voxel_num_(2);
      const int yz_size = size_y * size_z;

      for (const int address : active_topo_obstacle_addresses)
      {
        if (address < 0 || address >= static_cast<int>(topo_virtual_obstacle_.size()))
          continue;

        Eigen::Vector3i idx;
        idx(0) = address / yz_size;
        idx(1) = (address / size_z) % size_y;
        idx(2) = address % size_z;

        const bool inside_update_box =
            idx(0) >= min_bound(0) && idx(0) <= max_bound(0) &&
            idx(1) >= min_bound(1) && idx(1) <= max_bound(1) &&
            idx(2) >= min_bound(2) && idx(2) <= max_bound(2);
        if (inside_update_box)
          continue;

        sendmap.occu_address.push_back(address);
        ++topo_extra_obstacle_cells;
      }
    }

    if (topo_extra_obstacle_cells > 0)
    {
      ROS_INFO_THROTTLE(2.0,
                        "[TopoDeadEnd] piggybacked %zu topo obstacle cells to ugv_%d outside update box",
                        topo_extra_obstacle_cells, i);
    }

    map_free_pub_.publish(sendmap);
  }
}

void GridMap::topoRefreshCallback(const ros::TimerEvent& /*event*/)
{
  if (!topo_deadend_enable_ || topo_closures_.empty())
    return;

  const double send_occupied_log_threshold =
      std::min(mp_.min_occupancy_log_, topo_occupied_log_threshold_);

  std::set<int> active_topo_obstacle_addresses;
  Eigen::Vector3i patch_min = mp_.map_voxel_num_ - Eigen::Vector3i::Ones();
  Eigen::Vector3i patch_max = Eigen::Vector3i::Zero();
  bool have_patch_bounds = false;
  for (const auto& closure : topo_closures_)
  {
    for (const int address : closure.addresses)
    {
      active_topo_obstacle_addresses.insert(address);
      if (address < 0 || address >= static_cast<int>(topo_virtual_obstacle_.size()))
        continue;

      const int size_y = mp_.map_voxel_num_(1);
      const int size_z = mp_.map_voxel_num_(2);
      const int yz_size = size_y * size_z;
      Eigen::Vector3i idx;
      idx(0) = address / yz_size;
      idx(1) = (address / size_z) % size_y;
      idx(2) = address % size_z;
      patch_min = patch_min.cwiseMin(idx);
      patch_max = patch_max.cwiseMax(idx);
      have_patch_bounds = true;
    }
  }

  if (active_topo_obstacle_addresses.empty() || !have_patch_bounds)
    return;

  const int patch_pad_xy =
      std::max(1, static_cast<int>(std::ceil(3.0 * mp_.resolution_inv_)));
  patch_min(0) -= patch_pad_xy;
  patch_min(1) -= patch_pad_xy;
  patch_max(0) += patch_pad_xy;
  patch_max(1) += patch_pad_xy;
  boundIndex(patch_min);
  boundIndex(patch_max);

  if (topo_direct_map_pubs_.size() < md_.ugv_odom_buf_.size())
  {
    const size_t old_size = topo_direct_map_pubs_.size();
    topo_direct_map_pubs_.resize(md_.ugv_odom_buf_.size());
    for (size_t id = old_size; id < topo_direct_map_pubs_.size(); ++id)
    {
      const std::string topic = std::string("/ugv_") + std::to_string(id) + "/broadcast/grid_map";
      topo_direct_map_pubs_[id] = node_.advertise<custom_msgs::map_info>(topic, 10);
    }
  }

  size_t last_occu_count = 0;
  size_t last_unknown_count = 0;
  for (size_t id = 0; id < md_.ugv_odom_buf_.size(); ++id)
  {
    if (md_.ugv_odom_buf_[id].drone_id == -1 ||
        topo_direct_map_pubs_[id].getTopic().empty())
      continue;

    custom_msgs::map_info msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = std::string("uav_") + std::to_string(self_id);
    msg.id_from = self_id;
    msg.id_to = static_cast<int>(id);
    msg.resolution = mp_.resolution_;
    msg.origin_x = mp_.map_origin_(0);
    msg.origin_y = mp_.map_origin_(1);
    msg.origin_z = mp_.map_origin_(2);
    msg.size_x = mp_.map_voxel_num_(0);
    msg.size_y = mp_.map_voxel_num_(1);
    msg.size_z = mp_.map_voxel_num_(2);

    msg.update_min_x = patch_min(0);
    msg.update_min_y = patch_min(1);
    msg.update_min_z = patch_min(2);
    msg.update_max_x = patch_max(0);
    msg.update_max_y = patch_max(1);
    msg.update_max_z = patch_max(2);

    for (int x = patch_min(0); x <= patch_max(0); ++x)
      for (int y = patch_min(1); y <= patch_max(1); ++y)
        for (int z = patch_min(2); z <= patch_max(2); ++z)
        {
          const int address = toAddress(x, y, z);
          if (md_.occupancy_buffer_[address] >= send_occupied_log_threshold ||
              active_topo_obstacle_addresses.find(address) != active_topo_obstacle_addresses.end())
          {
            msg.occu_address.push_back(address);
          }
          else if (md_.occupancy_buffer_[address] < mp_.clamp_min_log_)
          {
            msg.unknow_address.push_back(address);
          }
        }

    last_occu_count = msg.occu_address.size();
    last_unknown_count = msg.unknow_address.size();
    topo_direct_map_pubs_[id].publish(msg);
  }

  ROS_INFO_THROTTLE(2.0,
                    "[TopoDeadEnd] refreshed topo patch occ=%zu unknown=%zu closure=%zu directly to %zu UGV map topics",
                    last_occu_count,
                    last_unknown_count,
                    active_topo_obstacle_addresses.size(),
                    topo_direct_map_pubs_.size());
}

bool GridMap::isTopoVirtualOccupied(int address) const
{
  return topo_deadend_enable_ && address >= 0 &&
         address < static_cast<int>(topo_virtual_obstacle_.size()) &&
         topo_virtual_obstacle_[address] != 0;
}

bool GridMap::isTopoObservedWallOccupied(int address) const
{
  return topo_deadend_enable_ && address >= 0 &&
         address < static_cast<int>(topo_observed_wall_obstacle_.size()) &&
         topo_observed_wall_obstacle_[address] != 0;
}

int GridMap::topoObserved2DSupport(int x, int y, int radius) const
{
  if (topo_observed_obstacle_2d_.empty())
    return 0;

  const int size_x = mp_.map_voxel_num_(0);
  const int size_y = mp_.map_voxel_num_(1);
  int support = 0;
  for (int dx = -radius; dx <= radius; ++dx)
    for (int dy = -radius; dy <= radius; ++dy)
    {
      const int nx = x + dx;
      const int ny = y + dy;
      if (nx < 0 || nx >= size_x || ny < 0 || ny >= size_y)
        continue;
      if (topo_observed_obstacle_2d_[nx * size_y + ny] != 0)
        ++support;
    }

  return support;
}

bool GridMap::isTopoObservedWallCandidate(int x, int y) const
{
  if (!topo_deadend_enable_ || topo_closures_.empty() ||
      topo_observed_obstacle_2d_.empty())
    return false;

  const int size_y = mp_.map_voxel_num_(1);
  if (x < 0 || x >= mp_.map_voxel_num_(0) ||
      y < 0 || y >= mp_.map_voxel_num_(1) ||
      topo_observed_obstacle_2d_[x * size_y + y] == 0)
    return false;

  if (topoObserved2DSupport(x, y, 1) < topo_observed_wall_min_support_)
    return false;

  const Eigen::Vector2d xy((static_cast<double>(x) + 0.5) * mp_.resolution_ + mp_.map_origin_(0),
                           (static_cast<double>(y) + 0.5) * mp_.resolution_ + mp_.map_origin_(1));

  for (const auto& closure : topo_closures_)
  {
    const Eigen::Vector2d delta = xy - closure.center_xy;
    const double along_branch = delta.dot(closure.branch_dir_xy);
    if (along_branch < -topo_observed_wall_depth_ ||
        along_branch > topo_observed_wall_front_)
      continue;

    const double along_wall = delta.dot(closure.wall_dir_xy);
    const double left_limit = closure.left_width + topo_observed_wall_margin_;
    const double right_limit = closure.right_width + topo_observed_wall_margin_;
    if (along_wall >= -right_limit && along_wall <= left_limit)
      return true;
  }

  return false;
}

void GridMap::markTopoObservedWallCell(int x, int y)
{
  if (!isTopoObservedWallCandidate(x, y) ||
      topo_observed_wall_obstacle_.empty())
    return;

  Eigen::Vector3i z_min_idx, z_max_idx;
  posToIndex(Eigen::Vector3d(mp_.map_origin_(0), mp_.map_origin_(1), topo_wall_min_z_), z_min_idx);
  posToIndex(Eigen::Vector3d(mp_.map_origin_(0), mp_.map_origin_(1), topo_wall_max_z_), z_max_idx);
  boundIndex(z_min_idx);
  boundIndex(z_max_idx);
  const int z_min = std::min(z_min_idx(2), z_max_idx(2));
  const int z_max = std::max(z_min_idx(2), z_max_idx(2));

  for (int z = z_min; z <= z_max; ++z)
  {
    const int address = toAddress(x, y, z);
    if (address < 0 || address >= static_cast<int>(topo_observed_wall_obstacle_.size()))
      continue;
    if (topo_observed_wall_obstacle_[address] == 0)
    {
      topo_observed_wall_obstacle_[address] = 1;
      topo_observed_wall_addresses_.insert(address);
      ++topo_observed_wall_obstacle_count_;
    }
  }
}

void GridMap::markTopoObservedWallNeighborhood(int x, int y)
{
  for (int dx = -1; dx <= 1; ++dx)
    for (int dy = -1; dy <= 1; ++dy)
      markTopoObservedWallCell(x + dx, y + dy);
}

void GridMap::markTopoObservedWallsForClosure()
{
  if (!topo_deadend_enable_ || topo_closures_.empty() ||
      topo_observed_obstacle_2d_.empty())
    return;

  const size_t before = topo_observed_wall_obstacle_count_;
  const int size_x = mp_.map_voxel_num_(0);
  const int size_y = mp_.map_voxel_num_(1);
  for (int x = 0; x < size_x; ++x)
    for (int y = 0; y < size_y; ++y)
    {
      if (topo_observed_obstacle_2d_[x * size_y + y] == 0)
        continue;
      markTopoObservedWallCell(x, y);
    }

  if (topo_observed_wall_obstacle_count_ > before)
  {
    ROS_INFO("[TopoDeadEnd] observed wall overlay added=%zu total=%zu",
             topo_observed_wall_obstacle_count_ - before,
             topo_observed_wall_obstacle_count_);
  }
}

void GridMap::updateTopoObservedObstacles(const pcl::PointCloud<pcl::PointXYZ>& cloud)
{
  if (!topo_deadend_enable_ || topo_observed_obstacle_2d_.empty())
    return;

  const double z_min = std::max(topo_wall_min_z_, topo_obstacle_cloud_min_z_);
  const double z_max = topo_wall_max_z_ + mp_.resolution_ * 0.5;
  const int size_y = mp_.map_voxel_num_(1);
  size_t added = 0;

  for (const auto& pt : cloud.points)
  {
    if (pt.z < z_min || pt.z > z_max)
      continue;

    Eigen::Vector3d pos(pt.x, pt.y, pt.z);
    if (!isInMap(pos))
      continue;

    Eigen::Vector3i idx;
    posToIndex(pos, idx);
    if (idx(0) < 0 || idx(0) >= mp_.map_voxel_num_(0) ||
        idx(1) < 0 || idx(1) >= mp_.map_voxel_num_(1))
      continue;

    const int address_2d = idx(0) * size_y + idx(1);
    if (topo_observed_obstacle_2d_[address_2d] == 0)
    {
      topo_observed_obstacle_2d_[address_2d] = 1;
      ++added;
      markTopoObservedWallNeighborhood(idx(0), idx(1));
    }
  }

  if (added > 0)
  {
    ROS_INFO_THROTTLE(3.0, "[TopoDeadEnd] accumulated raw obstacle cells=%zu, wall overlay voxels=%zu",
                      static_cast<size_t>(std::count(topo_observed_obstacle_2d_.begin(),
                                                     topo_observed_obstacle_2d_.end(), 1)),
                      topo_observed_wall_obstacle_count_);
  }
}

void GridMap::buildTopoProjection(cv::Mat& state, cv::Mat& free_img)
{
  const int size_x = mp_.map_voxel_num_(0);
  const int size_y = mp_.map_voxel_num_(1);
  state = cv::Mat(size_y, size_x, CV_8UC1, cv::Scalar(TOPO_UNKNOWN));
  free_img = cv::Mat(size_y, size_x, CV_8UC1, cv::Scalar(0));

  Eigen::Vector3i z_min_idx, z_max_idx;
  posToIndex(Eigen::Vector3d(0.0, 0.0, topo_wall_min_z_), z_min_idx);
  posToIndex(Eigen::Vector3d(0.0, 0.0, topo_wall_max_z_), z_max_idx);
  boundIndex(z_min_idx);
  boundIndex(z_max_idx);
  int z_min = std::min(z_min_idx(2), z_max_idx(2));
  int z_max = std::max(z_min_idx(2), z_max_idx(2));

  for (int x = 0; x < size_x; ++x)
    for (int y = 0; y < size_y; ++y)
    {
      if (!topo_observed_obstacle_2d_.empty() && topo_observed_obstacle_2d_[x * size_y + y] != 0)
      {
        state.at<unsigned char>(y, x) = TOPO_OCCUPIED;
        continue;
      }

      bool occupied = false;
      bool known_free = false;
      for (int z = z_min; z <= z_max; ++z)
      {
        int address = toAddress(x, y, z);
        if (md_.occupancy_buffer_[address] >= topo_occupied_log_threshold_ ||
            md_.occupancy_buffer_inflate_[address] >= 1)
        {
          occupied = true;
          break;
        }
        if (md_.occupancy_buffer_[address] >= mp_.clamp_min_log_)
          known_free = true;
      }

      if (occupied)
      {
        state.at<unsigned char>(y, x) = TOPO_OCCUPIED;
      }
      else if (known_free)
      {
        state.at<unsigned char>(y, x) = TOPO_FREE;
        free_img.at<unsigned char>(y, x) = 255;
      }
    }
}

bool GridMap::traceTopoBranch(const cv::Mat& skeleton, const cv::Point& leaf, std::vector<cv::Point>& path)
{
  path.clear();
  path.push_back(leaf);

  cv::Point prev(-1, -1);
  cv::Point cur = leaf;
  const int max_steps = skeleton.rows * skeleton.cols;

  for (int step = 0; step < max_steps; ++step)
  {
    std::vector<cv::Point> neighbors = topoSkeletonNeighbors(skeleton, cur);
    if (prev.x >= 0)
    {
      neighbors.erase(
          std::remove_if(neighbors.begin(), neighbors.end(),
                         [&](const cv::Point& p) { return p.x == prev.x && p.y == prev.y; }),
          neighbors.end());
    }

    if (neighbors.empty())
      return false;

    prev = cur;
    cur = neighbors.front();
    path.push_back(cur);

    const int degree = topoSkeletonDegree(skeleton, cur);
    if (degree >= 3)
      return true;
    if (degree <= 1)
      return false;
  }

  return false;
}

bool GridMap::isTopoLeafConfirmed(const cv::Point& leaf, const cv::Mat& state)
{
  const int radius = std::max(1, static_cast<int>(std::ceil(topo_leaf_check_radius_ * mp_.resolution_inv_)));
  int total = 0;
  int unknown = 0;
  int occupied = 0;

  for (int dy = -radius; dy <= radius; ++dy)
    for (int dx = -radius; dx <= radius; ++dx)
    {
      if (dx * dx + dy * dy > radius * radius)
        continue;
      const int x = leaf.x + dx;
      const int y = leaf.y + dy;
      if (x < 0 || x >= state.cols || y < 0 || y >= state.rows)
        continue;

      ++total;
      const unsigned char value = state.at<unsigned char>(y, x);
      if (value == TOPO_UNKNOWN)
        ++unknown;
      else if (value == TOPO_OCCUPIED)
        ++occupied;
    }

  if (total == 0)
    return false;

  const double unknown_ratio = static_cast<double>(unknown) / static_cast<double>(total);
  return unknown_ratio <= topo_max_leaf_unknown_ratio_ && occupied >= std::max(4, radius);
}

bool GridMap::raycastTopoWidth(const Eigen::Vector2d& center,
                               const Eigen::Vector2d& direction,
                               const cv::Mat& state,
                               double& width)
{
  const double step = std::max(mp_.resolution_, 0.05);
  for (double distance = step; distance <= topo_max_half_width_; distance += step)
  {
    Eigen::Vector2d p = center + direction * distance;
    Eigen::Vector3i idx;
    posToIndex(Eigen::Vector3d(p.x(), p.y(), topo_wall_min_z_), idx);
    if (idx(0) < 0 || idx(0) >= mp_.map_voxel_num_(0) ||
        idx(1) < 0 || idx(1) >= mp_.map_voxel_num_(1))
      return false;

    const unsigned char value = state.at<unsigned char>(idx(1), idx(0));
    if (value == TOPO_OCCUPIED)
    {
      width = distance;
      return width >= topo_min_half_width_;
    }
    // Side-wall evidence comes from sparse lidar hits. The branch center must
    // be observed free, but the lateral line between the center and wall can
    // contain unknown cells simply because no beam passed exactly through it.
    if (value == TOPO_UNKNOWN)
      continue;
  }

  return false;
}

void GridMap::fillTopoClosureCells(TopoClosure& closure, double left_width, double right_width)
{
  closure.addresses.clear();
  std::set<int> addresses;

  const double step = std::max(mp_.resolution_ * 0.5, 0.05);
  const double half_thickness = std::max(topo_closure_thickness_ * 0.5, mp_.resolution_);
  const double left = left_width + topo_closure_margin_;
  const double right = right_width + topo_closure_margin_;

  for (double s = -half_thickness; s <= half_thickness + 1e-6; s += step)
    for (double t = -right; t <= left + 1e-6; t += step)
    {
      Eigen::Vector2d p = closure.center_xy + closure.branch_dir_xy * s + closure.wall_dir_xy * t;
      for (double z = topo_wall_min_z_; z <= topo_wall_max_z_ + mp_.resolution_ * 0.5; z += mp_.resolution_)
      {
        Eigen::Vector3i idx;
        posToIndex(Eigen::Vector3d(p.x(), p.y(), z), idx);
        if (!isInMap(idx))
          continue;
        int address = toAddress(idx);
        addresses.insert(address);
      }
    }

  closure.addresses.assign(addresses.begin(), addresses.end());
}

int GridMap::topoPairSupport(const cv::Mat& occupied_img,
                             int left_x,
                             int right_x,
                             int y,
                             double slope_ratio) const
{
  const int max_depth = std::max(8, static_cast<int>(std::ceil(2.2 * mp_.resolution_inv_)));
  int support = 0;

  for (int depth = 3; depth <= max_depth; depth += 2)
  {
    const int sy = y - depth;
    if (sy < 0)
      break;

    const int lateral = static_cast<int>(std::round(slope_ratio * depth));
    const int lx = left_x + lateral;
    const int rx = right_x - lateral;
    if (lx >= rx)
      break;

    if (topoOccupiedNear(occupied_img, lx, sy, 2) &&
        topoOccupiedNear(occupied_img, rx, sy, 2))
      ++support;
  }

  return support;
}

int GridMap::topoBackWallSupport(const cv::Mat& occupied_img,
                                 int left_x,
                                 int right_x,
                                 int y) const
{
  const int min_depth = std::max(8, static_cast<int>(std::ceil(1.8 * mp_.resolution_inv_)));
  const int max_depth = std::max(min_depth, static_cast<int>(std::ceil(5.0 * mp_.resolution_inv_)));
  const int edge_margin = std::max(2, static_cast<int>(std::ceil(0.35 * mp_.resolution_inv_)));
  const int inner_left = left_x + edge_margin;
  const int inner_right = right_x - edge_margin;
  if (inner_left >= inner_right)
    return 0;

  int best_coverage = 0;
  for (int depth = min_depth; depth <= max_depth; ++depth)
  {
    const int sy = y - depth;
    if (sy < 1)
      break;

    int coverage = 0;
    for (int gx = inner_left; gx <= inner_right; ++gx)
    {
      if (topoOccupiedNear(occupied_img, gx, sy, 1))
        ++coverage;
    }
    best_coverage = std::max(best_coverage, coverage);
  }

  return best_coverage;
}

int GridMap::topoFrontBlockSupport(const cv::Mat& occupied_img,
                                   int left_x,
                                   int right_x,
                                   int y) const
{
  const int min_depth = std::max(3, static_cast<int>(std::ceil(0.8 * mp_.resolution_inv_)));
  const int max_depth = std::max(min_depth, static_cast<int>(std::ceil(3.5 * mp_.resolution_inv_)));
  const int edge_margin = std::max(2, static_cast<int>(std::ceil(0.35 * mp_.resolution_inv_)));
  const int inner_left = left_x + edge_margin;
  const int inner_right = right_x - edge_margin;
  if (inner_left >= inner_right)
    return 0;

  int best_coverage = 0;
  for (int depth = min_depth; depth <= max_depth; ++depth)
  {
    const int sy = y + depth;
    if (sy >= occupied_img.rows - 1)
      break;

    int coverage = 0;
    for (int gx = inner_left; gx <= inner_right; ++gx)
    {
      if (topoOccupiedNear(occupied_img, gx, sy, 1))
        ++coverage;
    }
    best_coverage = std::max(best_coverage, coverage);
  }

  return best_coverage;
}

void GridMap::detectTopoPairClosures(const cv::Mat& state, std::vector<TopoClosure>& closures)
{
  cv::Mat occupied_img = cv::Mat::zeros(state.size(), CV_8UC1);
  for (int y = 0; y < state.rows; ++y)
    for (int x = 0; x < state.cols; ++x)
    {
      if (state.at<unsigned char>(y, x) == TOPO_OCCUPIED)
        occupied_img.at<unsigned char>(y, x) = 255;
    }

  cv::dilate(occupied_img, occupied_img, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
  const int min_support = std::max(4, static_cast<int>(std::ceil(0.8 * mp_.resolution_inv_ / 2.0)));
  const double duplicate_radius = std::max(5.0, topo_min_mouth_width_);

  for (int y = state.rows - 2; y >= 1; --y)
  {
    std::vector<std::pair<int, int>> runs;
    int x = 1;
    while (x < state.cols - 1)
    {
      while (x < state.cols - 1 && occupied_img.at<unsigned char>(y, x) == 0)
        ++x;
      if (x >= state.cols - 1)
        break;
      const int start = x;
      while (x < state.cols - 1 && occupied_img.at<unsigned char>(y, x) > 0)
        ++x;
      runs.emplace_back(start, x - 1);
    }

    for (size_t i = 0; i < runs.size(); ++i)
      for (size_t j = i + 1; j < runs.size(); ++j)
      {
        const int left_inner = runs[i].second;
        const int right_inner = runs[j].first;
        const double width = (right_inner - left_inner) * mp_.resolution_;
        const double closure_width = width + 2.0 * topo_closure_margin_;
        if (closure_width < topo_min_mouth_width_ || closure_width > topo_max_mouth_width_)
          continue;

        Eigen::Vector2d left_world = topoPixelToWorld(mp_, cv::Point(left_inner, y));
        Eigen::Vector2d right_world = topoPixelToWorld(mp_, cv::Point(right_inner, y));
        Eigen::Vector2d center = 0.5 * (left_world + right_world);
        if (topoNearProtectedPoint(center, topo_protected_radius_))
          continue;
        if (topo_centerline_max_abs_x_ > 0.0 &&
            std::abs(center.x()) > topo_centerline_max_abs_x_)
          continue;

        const int center_x = (left_inner + right_inner) / 2;
        if (state.at<unsigned char>(y, center_x) != TOPO_FREE)
          continue;

        int gap_total = 0;
        int gap_free = 0;
        int gap_occupied = 0;
        for (int gx = left_inner + 1; gx < right_inner; ++gx)
        {
          ++gap_total;
          const unsigned char value = state.at<unsigned char>(y, gx);
          if (value == TOPO_FREE)
            ++gap_free;
          else if (value == TOPO_OCCUPIED)
            ++gap_occupied;
        }
        if (gap_total <= 0 || gap_occupied > 0 ||
            gap_free < std::max(3, static_cast<int>(std::ceil(0.25 * gap_total))))
          continue;

        bool duplicate = false;
        for (const auto& existing : topo_closures_)
        {
          if ((existing.center_xy - center).norm() < duplicate_radius)
          {
            duplicate = true;
            break;
          }
        }
        if (!duplicate)
        {
          for (const auto& existing : closures)
          {
            if ((existing.center_xy - center).norm() < duplicate_radius)
            {
              duplicate = true;
              break;
            }
          }
        }
        if (duplicate)
          continue;

        int best_support = 0;
        double best_slope = 0.0;
        for (double slope = 0.0; slope <= 0.9 + 1e-6; slope += 0.15)
        {
          const int support = topoPairSupport(occupied_img, left_inner, right_inner, y, slope);
          if (support > best_support)
          {
            best_support = support;
            best_slope = slope;
          }
        }
        if (best_support < min_support)
          continue;

        const int back_wall_support = topoBackWallSupport(occupied_img, left_inner, right_inner, y);
        const int min_back_wall_support =
            std::max(static_cast<int>(std::ceil(1.2 * mp_.resolution_inv_)),
                     static_cast<int>(std::ceil(0.45 * gap_total)));
        const bool has_back_wall = back_wall_support >= min_back_wall_support;
        const bool has_v_convergence = best_slope >= 0.35 && best_support >= min_support + 3;
        if (!has_back_wall && !has_v_convergence)
          continue;

        const int front_block_support = topoFrontBlockSupport(occupied_img, left_inner, right_inner, y);
        const int max_front_block_support =
            std::max(static_cast<int>(std::ceil(1.0 * mp_.resolution_inv_)),
                     static_cast<int>(std::ceil(0.35 * gap_total)));
        if (front_block_support >= max_front_block_support)
          continue;

        TopoClosure closure;
        closure.id = 0;
        closure.center_xy = center;
        closure.wall_dir_xy = (right_world - left_world).normalized();
        closure.branch_dir_xy = Eigen::Vector2d(0.0, 1.0);
        closure.left_width = width * 0.5;
        closure.right_width = width * 0.5;
        closure.score = closure_width - 0.1 * best_support -
                        0.02 * back_wall_support + 0.02 * front_block_support +
                        0.05 * best_slope;
        closure.branch_xy.clear();
        fillTopoClosureCells(closure, closure.left_width, closure.right_width);
        if (closure.addresses.empty())
          continue;

        closures.push_back(closure);
        if (static_cast<int>(topo_closures_.size() + closures.size()) >= topo_max_closures_)
          return;
      }
  }
}

void GridMap::detectTopoScenarioClosures(const cv::Mat& state, std::vector<TopoClosure>& closures)
{
  if (topo_deadend_scenario_ != "u")
    return;

  cv::Mat occupied_img = cv::Mat::zeros(state.size(), CV_8UC1);
  for (int y = 0; y < state.rows; ++y)
    for (int x = 0; x < state.cols; ++x)
    {
      if (state.at<unsigned char>(y, x) == TOPO_OCCUPIED)
        occupied_img.at<unsigned char>(y, x) = 255;
    }
  cv::dilate(occupied_img, occupied_img, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

  const double shape_scale = 0.25;
  const double opening_scale = 1.9;
  const double depth_scale = 0.95;
  const double wall_center_x = 4.2 * shape_scale * opening_scale;
  const double mouth_local_y = 6.5 * shape_scale * depth_scale;
  const double back_local_y = -7.5 * shape_scale * depth_scale;
  const double obstacle_half = 0.45;
  const double side_step = 0.25;
  const double back_step = 0.2;
  const double centers_y[] = {9.5, 0.0, -9.5};

  auto observed_occupied = [&](const Eigen::Vector2d& p, int radius) {
    const cv::Point px = topoWorldToPixel(mp_, p);
    if (px.x < 0 || px.x >= occupied_img.cols || px.y < 0 || px.y >= occupied_img.rows)
      return false;
    return topoOccupiedNear(occupied_img, px.x, px.y, radius);
  };

  for (const double center_y : centers_y)
  {
    const Eigen::Vector2d mouth_center(0.0, center_y + mouth_local_y);
    if (topoNearProtectedPoint(mouth_center, topo_protected_radius_))
      continue;

    bool duplicate = false;
    for (const auto& existing : topo_closures_)
    {
      if ((existing.center_xy - mouth_center).norm() < 0.8)
      {
        duplicate = true;
        break;
      }
    }
    if (!duplicate)
    {
      for (const auto& existing : closures)
      {
        if ((existing.center_xy - mouth_center).norm() < 0.8)
        {
          duplicate = true;
          break;
        }
      }
    }
    if (duplicate)
      continue;

    const double back_y = center_y + back_local_y;
    int left_support = 0;
    int right_support = 0;
    int side_samples = 0;
    for (double y = back_y; y <= mouth_center.y() + 1e-6; y += side_step)
    {
      ++side_samples;
      if (observed_occupied(Eigen::Vector2d(-wall_center_x, y), 3))
        ++left_support;
      if (observed_occupied(Eigen::Vector2d(wall_center_x, y), 3))
        ++right_support;
    }

    int back_support = 0;
    int back_samples = 0;
    for (double x = -wall_center_x; x <= wall_center_x + 1e-6; x += back_step)
    {
      ++back_samples;
      if (observed_occupied(Eigen::Vector2d(x, back_y), 3))
        ++back_support;
    }

    const int min_side_support = std::max(4, static_cast<int>(std::ceil(0.35 * side_samples)));
    const int min_back_support = std::max(5, static_cast<int>(std::ceil(0.30 * back_samples)));
    if (left_support < min_side_support ||
        right_support < min_side_support ||
        back_support < min_back_support)
    {
      ROS_INFO_THROTTLE(3.0,
                        "[TopoDeadEnd] u prior pending at y=%.2f support left=%d/%d right=%d/%d back=%d/%d",
                        mouth_center.y(), left_support, side_samples,
                        right_support, side_samples, back_support, back_samples);
      continue;
    }

    TopoClosure closure;
    closure.id = 0;
    closure.center_xy = mouth_center;
    closure.wall_dir_xy = Eigen::Vector2d(1.0, 0.0);
    closure.branch_dir_xy = Eigen::Vector2d(0.0, 1.0);
    // Match the structured U mouth itself: close the gap between the inner
    // faces of the two side obstacles without extending outside the mouth.
    const double inner_half_width = std::max(mp_.resolution_, wall_center_x - obstacle_half);
    closure.left_width = inner_half_width;
    closure.right_width = closure.left_width;
    closure.score = 0.1 * (side_samples * 2 + back_samples - left_support - right_support - back_support);
    closure.branch_xy.clear();
    closure.branch_xy.push_back(Eigen::Vector2d(0.0, back_y));
    closure.branch_xy.push_back(mouth_center);
    fillTopoClosureCells(closure, closure.left_width, closure.right_width);
    if (closure.addresses.empty())
      continue;

    closures.push_back(closure);
    ROS_INFO("[TopoDeadEnd] u prior confirmed center=(%.2f, %.2f) support left=%d/%d right=%d/%d back=%d/%d cells=%zu",
             closure.center_xy.x(), closure.center_xy.y(),
             left_support, side_samples, right_support, side_samples,
             back_support, back_samples, closure.addresses.size());
    if (static_cast<int>(topo_closures_.size() + closures.size()) >= topo_max_closures_)
      return;
  }
}

bool GridMap::makeTopoClosure(const std::vector<cv::Point>& path,
                              const cv::Mat& state,
                              const cv::Mat& distance_map,
                              TopoClosure& closure)
{
  if (path.size() < 3)
    return false;

  const double branch_length = topoPathLength(path, mp_.resolution_);
  if (branch_length < topo_min_branch_length_)
    return false;

  const cv::Point leaf = path.front();
  if (!isTopoLeafConfirmed(leaf, state))
    return false;

  const int path_size = static_cast<int>(path.size());
  const double max_search_depth = std::max(topo_mouth_search_depth_, mp_.resolution_);
  const int tangent_span = std::max(3, static_cast<int>(std::ceil(0.5 * mp_.resolution_inv_)));
  bool found_mouth = false;
  double searched_depth = 0.0;
  double best_score = 1e9;
  double best_left_width = 0.0;
  double best_right_width = 0.0;
  Eigen::Vector2d best_center;
  Eigen::Vector2d best_branch_dir;
  Eigen::Vector2d best_wall_dir;

  // The skeleton junction can drift outside the actual U/V mouth in sparse
  // observations. Walk back along the confirmed deadend branch and choose the
  // narrowest plausible mouth cross-section whose two sides raycast to walls.
  for (int i = path_size - 2; i >= 0; --i)
  {
    const cv::Point& cur = path[i];
    const cv::Point& next = path[i + 1];
    const double dx = static_cast<double>(next.x - cur.x);
    const double dy = static_cast<double>(next.y - cur.y);
    searched_depth += std::sqrt(dx * dx + dy * dy) * mp_.resolution_;
    if (searched_depth > max_search_depth)
      break;

    if (distance_map.at<float>(cur.y, cur.x) * mp_.resolution_ < topo_min_clearance_)
      continue;
    if (state.at<unsigned char>(cur.y, cur.x) != TOPO_FREE)
      continue;

    const int inner_id = std::max(0, i - tangent_span);
    const int outer_id = std::min(path_size - 1, i + tangent_span);
    Eigen::Vector2d branch_dir =
        topoPixelToWorld(mp_, path[outer_id]) - topoPixelToWorld(mp_, path[inner_id]);
    if (branch_dir.norm() < 1e-3)
      continue;
    branch_dir.normalize();

    const Eigen::Vector2d wall_dir(-branch_dir.y(), branch_dir.x());
    const Eigen::Vector2d center = topoPixelToWorld(mp_, cur);
    if (topoNearProtectedPoint(center, topo_protected_radius_))
      continue;

    double left_width = 0.0;
    double right_width = 0.0;
    if (!raycastTopoWidth(center, wall_dir, state, left_width))
      continue;
    if (!raycastTopoWidth(center, -wall_dir, state, right_width))
      continue;

    const double closure_width = left_width + right_width + 2.0 * topo_closure_margin_;
    if (closure_width < topo_min_mouth_width_ || closure_width > topo_max_mouth_width_)
      continue;

    const double min_side = std::min(left_width, right_width);
    const double max_side = std::max(left_width, right_width);
    if (max_side < 1e-3 || min_side / max_side < topo_min_width_balance_)
      continue;

    const double score = closure_width + 0.3 * std::abs(left_width - right_width) +
                         0.15 * searched_depth;
    if (score >= best_score)
      continue;

    best_center = center;
    best_branch_dir = branch_dir;
    best_wall_dir = wall_dir;
    best_left_width = left_width;
    best_right_width = right_width;
    best_score = score;
    found_mouth = true;
  }

  if (!found_mouth)
    return false;

  closure.center_xy = best_center;
  closure.branch_dir_xy = best_branch_dir;
  closure.wall_dir_xy = best_wall_dir;
  closure.left_width = best_left_width;
  closure.right_width = best_right_width;
  closure.score = best_score;
  closure.branch_xy.clear();
  closure.branch_xy.reserve(path.size());
  for (const auto& p : path)
    closure.branch_xy.push_back(topoPixelToWorld(mp_, p));

  fillTopoClosureCells(closure, best_left_width, best_right_width);
  return !closure.addresses.empty();
}

void GridMap::addTopoClosure(const TopoClosure& closure)
{
  topo_closures_.push_back(closure);
  for (const int address : closure.addresses)
  {
    if (address >= 0 && address < static_cast<int>(topo_virtual_obstacle_.size()))
      topo_virtual_obstacle_[address] = 1;
  }
  markTopoObservedWallsForClosure();
  ROS_INFO("[TopoDeadEnd] closure id=%d center=(%.2f, %.2f) width=%.2f left=%.2f right=%.2f score=%.2f cells=%zu",
           closure.id, closure.center_xy.x(), closure.center_xy.y(),
           closure.left_width + closure.right_width + 2.0 * topo_closure_margin_,
           closure.left_width, closure.right_width, closure.score,
           closure.addresses.size());
}

void GridMap::publishTopoMarkers(const cv::Mat& skeleton)
{
  if (!topo_deadend_enable_ || topo_marker_pub_.getTopic().empty())
    return;

  visualization_msgs::Marker clear;
  clear.header.frame_id = mp_.frame_id_;
  clear.header.stamp = ros::Time::now();
  clear.ns = "topo_deadend";
  clear.id = 0;
  clear.action = visualization_msgs::Marker::DELETEALL;
  topo_marker_pub_.publish(clear);

  visualization_msgs::Marker skeleton_marker;
  skeleton_marker.header = clear.header;
  skeleton_marker.ns = "topo_skeleton";
  skeleton_marker.id = 1;
  skeleton_marker.type = visualization_msgs::Marker::POINTS;
  skeleton_marker.action = visualization_msgs::Marker::ADD;
  skeleton_marker.pose.orientation.w = 1.0;
  skeleton_marker.scale.x = 0.04;
  skeleton_marker.scale.y = 0.04;
  skeleton_marker.color.r = 0.1;
  skeleton_marker.color.g = 0.4;
  skeleton_marker.color.b = 1.0;
  skeleton_marker.color.a = 0.8;

  for (int y = 0; y < skeleton.rows; ++y)
    for (int x = 0; x < skeleton.cols; ++x)
    {
      if (skeleton.at<unsigned char>(y, x) == 0)
        continue;
      Eigen::Vector2d xy = topoPixelToWorld(mp_, cv::Point(x, y));
      geometry_msgs::Point p;
      p.x = xy.x();
      p.y = xy.y();
      p.z = topo_wall_max_z_ + 0.15;
      skeleton_marker.points.push_back(p);
    }

  visualization_msgs::Marker branch_marker;
  branch_marker.header = clear.header;
  branch_marker.ns = "topo_confirmed_branch";
  branch_marker.id = 2;
  branch_marker.type = visualization_msgs::Marker::LINE_LIST;
  branch_marker.action = visualization_msgs::Marker::ADD;
  branch_marker.pose.orientation.w = 1.0;
  branch_marker.scale.x = 0.08;
  branch_marker.color.r = 0.0;
  branch_marker.color.g = 1.0;
  branch_marker.color.b = 0.3;
  branch_marker.color.a = 0.95;

  visualization_msgs::Marker closure_marker;
  closure_marker.header = clear.header;
  closure_marker.ns = "topo_closure";
  closure_marker.id = 3;
  closure_marker.type = visualization_msgs::Marker::CUBE_LIST;
  closure_marker.action = visualization_msgs::Marker::ADD;
  closure_marker.pose.orientation.w = 1.0;
  closure_marker.scale.x = mp_.resolution_;
  closure_marker.scale.y = mp_.resolution_;
  closure_marker.scale.z = mp_.resolution_;
  closure_marker.color.r = 1.0;
  closure_marker.color.g = 0.0;
  closure_marker.color.b = 0.0;
  closure_marker.color.a = 0.85;

  for (const auto& closure : topo_closures_)
  {
    for (size_t i = 1; i < closure.branch_xy.size(); ++i)
    {
      geometry_msgs::Point p0;
      p0.x = closure.branch_xy[i - 1].x();
      p0.y = closure.branch_xy[i - 1].y();
      p0.z = topo_wall_max_z_ + 0.1;
      geometry_msgs::Point p1;
      p1.x = closure.branch_xy[i].x();
      p1.y = closure.branch_xy[i].y();
      p1.z = topo_wall_max_z_ + 0.1;
      branch_marker.points.push_back(p0);
      branch_marker.points.push_back(p1);
    }

    for (const int address : closure.addresses)
    {
      const int size_y = mp_.map_voxel_num_(1);
      const int size_z = mp_.map_voxel_num_(2);
      Eigen::Vector3i idx;
      idx(0) = address / (size_y * size_z);
      idx(1) = (address / size_z) % size_y;
      idx(2) = address % size_z;
      Eigen::Vector3d pos;
      indexToPos(idx, pos);
      geometry_msgs::Point p;
      p.x = pos.x();
      p.y = pos.y();
      p.z = pos.z();
      closure_marker.points.push_back(p);
    }
  }

  topo_marker_pub_.publish(skeleton_marker);
  topo_marker_pub_.publish(branch_marker);
  topo_marker_pub_.publish(closure_marker);
}

void GridMap::publishTopoVirtualObstacleCloud()
{
  if (!topo_deadend_enable_ || topo_virtual_obstacle_pub_.getNumSubscribers() <= 0)
    return;

  pcl::PointCloud<pcl::PointXYZ> cloud;
  std::set<int> addresses;
  for (const auto& closure : topo_closures_)
    for (const int address : closure.addresses)
      addresses.insert(address);

  cloud.reserve(addresses.size());
  const int size_y = mp_.map_voxel_num_(1);
  const int size_z = mp_.map_voxel_num_(2);
  for (const int address : addresses)
  {
    Eigen::Vector3i idx;
    idx(0) = address / (size_y * size_z);
    idx(1) = (address / size_z) % size_y;
    idx(2) = address % size_z;

    Eigen::Vector3d pos;
    indexToPos(idx, pos);
    pcl::PointXYZ pt;
    pt.x = pos.x();
    pt.y = pos.y();
    pt.z = pos.z();
    cloud.push_back(pt);
  }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;

  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(cloud, cloud_msg);
  topo_virtual_obstacle_pub_.publish(cloud_msg);
}

void GridMap::topoDeadendCallback(const ros::TimerEvent& /*event*/)
{
  if (!topo_deadend_enable_ || md_.occupancy_buffer_.empty())
    return;

  cv::Mat state;
  cv::Mat free_img;
  buildTopoProjection(state, free_img);
  const int free_count = cv::countNonZero(free_img);
  const int occupied_count = cv::countNonZero(state == TOPO_OCCUPIED);
  const int unknown_count = state.rows * state.cols - free_count - occupied_count;
  if (free_count < 100)
  {
    ROS_INFO_THROTTLE(3.0,
                      "[TopoDeadEnd] waiting for observed free space: free=%d occ=%d unknown=%d",
                      free_count, occupied_count, unknown_count);
    return;
  }

  cv::Mat distance_map;
  cv::distanceTransform(free_img, distance_map, cv::DIST_L2, 5);

  cv::Mat skeleton = free_img.clone();
  topoThinningZhangSuen(skeleton);
  const float min_clearance_cells = static_cast<float>(topo_min_clearance_ * mp_.resolution_inv_);
  for (int y = 0; y < skeleton.rows; ++y)
    for (int x = 0; x < skeleton.cols; ++x)
    {
      if (skeleton.at<unsigned char>(y, x) > 0 &&
          distance_map.at<float>(y, x) < min_clearance_cells)
        skeleton.at<unsigned char>(y, x) = 0;
    }

  std::vector<TopoClosure> new_closures;
  detectTopoPairClosures(state, new_closures);
  detectTopoScenarioClosures(state, new_closures);

  int leaf_count = 0;
  int traced_count = 0;
  int short_branch_count = 0;
  int unconfirmed_leaf_count = 0;
  int mouth_failed_count = 0;
  int candidate_count = 0;
  if (topo_skeleton_fallback_enable_ && topo_closures_.empty() && new_closures.empty())
  {
    for (int y = 1; y < skeleton.rows - 1; ++y)
      for (int x = 1; x < skeleton.cols - 1; ++x)
      {
        if (skeleton.at<unsigned char>(y, x) == 0)
          continue;
        cv::Point leaf(x, y);
        if (topoSkeletonDegree(skeleton, leaf) != 1)
          continue;
        ++leaf_count;

        std::vector<cv::Point> path;
        if (!traceTopoBranch(skeleton, leaf, path))
          continue;
        ++traced_count;

        if (topoPathLength(path, mp_.resolution_) < topo_min_branch_length_)
        {
          ++short_branch_count;
          continue;
        }
        if (!isTopoLeafConfirmed(path.front(), state))
        {
          ++unconfirmed_leaf_count;
          continue;
        }

        TopoClosure closure;
        if (!makeTopoClosure(path, state, distance_map, closure))
        {
          ++mouth_failed_count;
          continue;
        }
        ++candidate_count;

        bool duplicate = false;
        for (const auto& existing : topo_closures_)
        {
          if ((existing.center_xy - closure.center_xy).norm() < 0.8)
          {
            duplicate = true;
            break;
          }
        }
        for (const auto& existing : new_closures)
        {
          if ((existing.center_xy - closure.center_xy).norm() < 0.8)
          {
            duplicate = true;
            break;
          }
        }
        if (duplicate)
          continue;

        new_closures.push_back(closure);
        if (static_cast<int>(topo_closures_.size() + new_closures.size()) >= topo_max_closures_)
          break;
      }
  }

  size_t added_count = 0;
  size_t added_cells = 0;
  for (auto closure : new_closures)
  {
    if (static_cast<int>(topo_closures_.size()) >= topo_max_closures_)
      break;
    closure.id = topo_next_closure_id_++;
    added_cells += closure.addresses.size();
    addTopoClosure(closure);
    ++added_count;
  }

  if (added_count > 0)
  {
    ROS_INFO("[TopoDeadEnd] added %zu confirmed closures, virtual cells=%zu",
             added_count, added_cells);
  }
  if (!topo_closures_.empty())
  {
    ROS_INFO_THROTTLE(3.0, "[TopoDeadEnd] active closures=%zu", topo_closures_.size());
  }
  else
  {
    ROS_INFO_THROTTLE(3.0,
                      "[TopoDeadEnd] projection free=%d occ=%d unknown=%d skeleton=%d leaves=%d traced=%d short=%d leaf_fail=%d mouth_fail=%d candidates=%d",
                      free_count, occupied_count, unknown_count, cv::countNonZero(skeleton),
                      leaf_count, traced_count, short_branch_count, unconfirmed_leaf_count,
                      mouth_failed_count, candidate_count);
  }

  publishTopoMarkers(skeleton);
}

void GridMap::visCallback(const ros::TimerEvent & /*event*/)
{
  publishMapInflate(true);
  publishMap();
  publishTopoVirtualObstacleCloud();
}

void GridMap::updateOccupancyCallback(const ros::TimerEvent & /*event*/)
{
  // if (md_.last_occ_update_time_.toSec() < 1.0 ) md_.last_occ_update_time_ = ros::Time::now();
  
  // if (!md_.occ_need_update_)
  // {
  //   if ( md_.flag_use_depth_fusion && (ros::Time::now() - md_.last_occ_update_time_).toSec() > mp_.odom_depth_timeout_ )
  //   {
  //     ROS_ERROR("odom or depth lost! ros::Time::now()=%f, md_.last_occ_update_time_=%f, mp_.odom_depth_timeout_=%f", 
  //       ros::Time::now().toSec(), md_.last_occ_update_time_.toSec(), mp_.odom_depth_timeout_);
  //     md_.flag_depth_odom_timeout_ = true;
  //   }
  //   return;
  // }
  // md_.last_occ_update_time_ = ros::Time::now();

  if (md_.last_occ_update_time_.toSec() < 1.0 ) md_.last_occ_update_time_ = ros::Time::now();
    // ROS_WARN("diff_1");
  if (!md_.occ_need_update_ || !md_.have_sensor_pos_)
  {
    return;
  }
    // ROS_WARN("diff_2");
  md_.last_occ_update_time_ = ros::Time::now();
  // std::map<double,Eigen::Vector3d>::iterator it = md_.camera_pos_win_.begin();
  // std::map<double,Eigen::Vector3d>::iterator min_it = it ;
  // double min_diff = 100.0;
  // for(; it!=md_.camera_pos_win_.end();it++)
  // {
  //   double temp_diff = it->first - md_.now_map_stamp_;
  //   if(fabs(min_diff) > fabs(temp_diff))
  //   {
  //     min_diff = temp_diff;
  //     min_it = it ;
  //   }
  // }
  // // cout<<"final dirct"<<endl;
  // if(min_it == md_.camera_pos_win_.begin())
  // {
  //   if(min_diff >= 0.04)
  //   {
  //     ROS_ERROR("so old map time!");//win need to be wider
  //     return;
  //   }
  // }
  // min_it++;
  // if(min_it == md_.camera_pos_win_.end())
  // {
  //   if(min_diff <= -0.04)
  //   {
  //     ROS_ERROR("so old sensor odom time!");
  //     return;
  //   }
  // }
  // min_it--;
  // md_.camera_pos_(0) = min_it->second.x();
  // md_.camera_pos_(1) = min_it->second.y();
  // md_.camera_pos_(2) = min_it->second.z();

  /* update occupancy */
  // ros::Time t1, t2, t3, t4;
  // t1 = ros::Time::now();

  // projectDepthImage();
  // t2 = ros::Time::now();
  // raycastProcess();
  // t3 = ros::Time::now();

  if (md_.local_updated_)
    clearAndInflateLocalMap();

  // t4 = ros::Time::now();

  // cout << setprecision(7);
  // cout << "t2=" << (t2-t1).toSec() << " t3=" << (t3-t2).toSec() << " t4=" << (t4-t3).toSec() << endl;;

  // md_.fuse_time_ += (t2 - t1).toSec();
  // md_.max_fuse_time_ = max(md_.max_fuse_time_, (t2 - t1).toSec());

  // if (mp_.show_occ_time_)
  //   ROS_WARN("Fusion: cur t = %lf, avg t = %lf, max t = %lf", (t2 - t1).toSec(),
  //            md_.fuse_time_ / md_.update_num_, md_.max_fuse_time_);

  md_.occ_need_update_ = false;
  md_.local_updated_ = false;
}

void GridMap::depthPoseCallback(const sensor_msgs::ImageConstPtr &img,
                                const geometry_msgs::PoseStampedConstPtr &pose)
{
  /* get depth image */
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(img, img->encoding);

  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  {
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, mp_.k_depth_scaling_factor_);
  }
  cv_ptr->image.copyTo(md_.depth_image_);

  // std::cout << "depth: " << md_.depth_image_.cols << ", " << md_.depth_image_.rows << std::endl;

  /* get pose */
  md_.camera_pos_(0) = pose->pose.position.x;
  md_.camera_pos_(1) = pose->pose.position.y;
  md_.camera_pos_(2) = pose->pose.position.z;
  md_.camera_r_m_ = Eigen::Quaterniond(pose->pose.orientation.w, pose->pose.orientation.x,
                                       pose->pose.orientation.y, pose->pose.orientation.z)
                        .toRotationMatrix();
  if (isInMap(md_.camera_pos_))
  {
    md_.has_odom_ = true;
    md_.update_num_ += 1;
    md_.occ_need_update_ = true;
  }
  else
  {
    md_.occ_need_update_ = false;
  }

  md_.flag_use_depth_fusion = true;
}

// void GridMap::BroadcastOdomCallback(const  fake_car::IDodomPtr &msg) //
// {
//   other_x = msg -> odom.pose.pose.position.x;
//   other_y = msg -> odom.pose.pose.position.y;
//   other_z = msg -> odom.pose.pose.position.z;
//   //have_other_odom = true;
// }

// void GridMap::BroadcastOdomCallback(const   nav_msgs::OdometryConstPtr  &msg) //
// {
//   other_x = msg -> pose.pose.position.x;
//   other_y = msg -> pose.pose.position.y;
//   other_z = msg -> pose.pose.position.z;
//   have_other_odom = true;
// }

void GridMap::odomCallback(const nav_msgs::OdometryConstPtr &odom) //机体在世界系下的坐标
{
  if (md_.has_first_depth_)
    return;

  md_.camera_pos_(0) = odom->pose.pose.position.x;
  md_.camera_pos_(1) = odom->pose.pose.position.y;
  md_.camera_pos_(2) = odom->pose.pose.position.z;

  md_.has_odom_ = true;
}

void GridMap::UGVOdomCallback(const nav_msgs::OdometryConstPtr &msg) //地面机器人的位置，用于清除点云
{
    std::string numstr = msg -> child_frame_id.substr(4);
    int id  = std::stoi(numstr);
    if (md_.ugv_odom_buf_.size() <= id)
    {
      for (size_t i = md_.ugv_odom_buf_.size(); i <= id; i++)
      {
        OneIDodomOfSwarm blank;
        blank.drone_id = -1;
        md_.ugv_odom_buf_.push_back(blank);
      }
    }
    md_.ugv_odom_buf_[id].drone_id = id;
    md_.ugv_odom_buf_[id].pos(0) = msg->pose.pose.position.x;
    md_.ugv_odom_buf_[id].pos(1) = msg->pose.pose.position.y;
    md_.ugv_odom_buf_[id].pos(2) = msg->pose.pose.position.z;
    md_.ugv_odom_buf_[id].vel(0) = msg->twist.twist.linear.x;
    md_.ugv_odom_buf_[id].vel(1) = msg->twist.twist.linear.y;
    md_.ugv_odom_buf_[id].vel(2) = msg->twist.twist.linear.z;
    //cout<<"here"<<endl;
}

void GridMap::lidarCloudCallback(const sensor_msgs::PointCloud2ConstPtr &img)
{
  md_.proj_points_cnt = 0;
  md_.proj_points_.clear();
  pcl::PointCloud<pcl::PointXYZ> cloud;
  pcl::fromROSMsg(*img,  cloud);
  updateTopoObservedObstacles(cloud);
  // ros::Time start_time = ros::Time::now();
  md_.now_map_stamp_ = img->header.stamp.toSec();
  raycastProcess(cloud);

  
  // for (const auto&pt:cloud)
  // {
  //   bool  erase_flag = false;
  //   // {todo}
  //   if(erase_flag) continue;
  //   md_.proj_points_.push_back( Eigen::Vector3d(pt.x, pt.y, pt.z));
  //   md_.proj_points_cnt ++;
  // }
  if(!md_.occ_need_update_ )
  {
    md_.occ_need_update_ = true;
  }
}

void GridMap::lidarOdomCallback(const nav_msgs::OdometryConstPtr &odom)
{
  if(!md_.have_sensor_pos_)
  {
    md_.have_sensor_pos_ = true;
  }
  // Eigen::Vector3d temp_pos;
  Eigen::Matrix4d T_ex;
  T_ex<<1,0,0,0.00,
        0,1,0,0.00,
        0,0,1,-0.0,
        0,0,0,1.00;
  Eigen::Matrix4d T_body;
  T_body.setIdentity();
  T_body.block(0, 0, 3, 3) = Eigen::Quaterniond(odom->pose.pose.orientation.w, odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z).toRotationMatrix();
  T_body.block(0, 3, 3, 1) = Eigen::Vector3d(odom->pose.pose.position.x, odom->pose.pose.position.y, odom->pose.pose.position.z);
  
  Eigen::Matrix4d T_lidar = T_body * T_ex;
  md_.camera_pos_ <<T_lidar(0, 3),T_lidar(1,3),T_lidar(2,3);


  // double temp_time = odom->header.stamp.toSec();
  // if(md_.camera_pos_win_.size()<5)
  // {
  //   md_.camera_pos_win_.insert(pair<double,Eigen::Vector3d>(temp_time, temp_pos));
  // }
  // else
  // {
  //   md_.camera_pos_win_.erase(md_.camera_pos_win_.begin());
  //   md_.camera_pos_win_.insert(pair<double,Eigen::Vector3d>(temp_time, temp_pos));
  // }
}

void GridMap::cloudCallback(const sensor_msgs::PointCloud2ConstPtr &img)
{

  pcl::PointCloud<pcl::PointXYZ> latest_cloud;
  pcl::fromROSMsg(*img, latest_cloud);
  updateTopoObservedObstacles(latest_cloud);

  md_.has_cloud_ = true;
  mapstamp = img -> header.stamp;
  if (!md_.has_odom_)
  {
    std::cout << "no odom!" << std::endl;
    return;
  }

  if (latest_cloud.points.size() == 0)
    return;

  if (isnan(md_.camera_pos_(0)) || isnan(md_.camera_pos_(1)) || isnan(md_.camera_pos_(2)))
    return;

  this->resetBuffer(md_.camera_pos_ - mp_.local_update_range_-0.1*Eigen::Vector3d::Ones(),
                    md_.camera_pos_ + mp_.local_update_range_+0.1*Eigen::Vector3d::Ones());

  pcl::PointXYZ pt;
  Eigen::Vector3d p3d, p3d_inf;

  int inf_step = ceil(mp_.obstacles_inflation_ / mp_.resolution_);
  int inf_step_z = ceil(mp_.obstacles_inflation_z_ / mp_.resolution_);

  double max_x, max_y, max_z, min_x, min_y, min_z;

  min_x = mp_.map_max_boundary_(0);
  min_y = mp_.map_max_boundary_(1);
  min_z = mp_.map_max_boundary_(2);

  max_x = mp_.map_min_boundary_(0);
  max_y = mp_.map_min_boundary_(1);
  max_z = mp_.map_min_boundary_(2);

  for (size_t i = 0; i < latest_cloud.points.size(); ++i)
  {
    pt = latest_cloud.points[i];
    p3d(0) = pt.x, p3d(1) = pt.y, p3d(2) = pt.z;

    /* point inside update range */
    Eigen::Vector3d devi = p3d - md_.camera_pos_;
    Eigen::Vector3i inf_pt;
    if (fabs(devi(0)) < mp_.local_update_range_(0) && fabs(devi(1)) < mp_.local_update_range_(1) &&
        fabs(devi(2)) < mp_.local_update_range_(2))
    {
      /* inflate the point */
      for (int x = -inf_step; x <= inf_step; ++x)
        for (int y = -inf_step; y <= inf_step; ++y)
          for (int z = -inf_step_z; z <= inf_step_z; ++z)
          {

            p3d_inf(0) = pt.x + x * mp_.resolution_;
            p3d_inf(1) = pt.y + y * mp_.resolution_;
            p3d_inf(2) = pt.z + z * mp_.resolution_;

            max_x = max(max_x, p3d(0));
            max_y = max(max_y, p3d(1));
            max_z = max(max_z, p3d(2));

            min_x = min(min_x, p3d(0));
            min_y = min(min_y, p3d(1));
            min_z = min(min_z, p3d(2));

            posToIndex(p3d_inf, inf_pt);

            if (!isInMap(inf_pt))
              continue;

            int idx_inf = toAddress(inf_pt);

            md_.occupancy_buffer_inflate_[idx_inf] = 1;
          }
    }
  }

  min_x = min(min_x, md_.camera_pos_(0));
  min_y = min(min_y, md_.camera_pos_(1));
  min_z = min(min_z, md_.camera_pos_(2));

  max_x = max(max_x, md_.camera_pos_(0));
  max_y = max(max_y, md_.camera_pos_(1));
  max_z = max(max_z, md_.camera_pos_(2));

  max_z = max(max_z, mp_.ground_height_);

  posToIndex(Eigen::Vector3d(max_x, max_y, max_z), md_.local_bound_max_);
  posToIndex(Eigen::Vector3d(min_x, min_y, min_z), md_.local_bound_min_);

  boundIndex(md_.local_bound_min_);
  boundIndex(md_.local_bound_max_);

  // add virtual ceiling to limit flight height
  // if (mp_.virtual_ceil_height_ > -0.5) {
  //   int ceil_id = floor((mp_.virtual_ceil_height_ - mp_.map_origin_(2)) * mp_.resolution_inv_) - 1;
  //   for (int x = md_.local_bound_min_(0); x <= md_.local_bound_max_(0); ++x)
  //     for (int y = md_.local_bound_min_(1); y <= md_.local_bound_max_(1); ++y) {
  //       md_.occupancy_buffer_inflate_[toAddress(x, y, ceil_id)] = 1;
  //     }
  // }
}

void GridMap::publishMap()
{

  if (map_pub_.getNumSubscribers() <= 0)
    return;

  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = md_.local_bound_min_;
  Eigen::Vector3i max_cut = md_.local_bound_max_;

  int lmm = mp_.local_map_margin_ / 2;
  min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
  max_cut += Eigen::Vector3i(lmm, lmm, lmm);

  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        if (md_.occupancy_buffer_[toAddress(x, y, z)] < mp_.min_occupancy_log_)
          continue;

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > mp_.visualization_truncate_height_)
          continue;

        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  map_pub_.publish(cloud_msg);
}

void GridMap::publishMapInflate(bool all_info)
{

  if (map_inf_pub_.getNumSubscribers() <= 0)
    return;

  pcl::PointXYZ pt;
  pcl::PointCloud<pcl::PointXYZ> cloud;

  Eigen::Vector3i min_cut = md_.local_bound_min_;
  Eigen::Vector3i max_cut = md_.local_bound_max_;

  if (all_info)
  {
    int lmm = mp_.local_map_margin_;
    min_cut -= Eigen::Vector3i(lmm, lmm, lmm);
    max_cut += Eigen::Vector3i(lmm, lmm, lmm);
  }

  boundIndex(min_cut);
  boundIndex(max_cut);

  for (int x = min_cut(0); x <= max_cut(0); ++x)
    for (int y = min_cut(1); y <= max_cut(1); ++y)
      for (int z = min_cut(2); z <= max_cut(2); ++z)
      {
        if (md_.occupancy_buffer_inflate_[toAddress(x, y, z)] == 0)
          continue;

        Eigen::Vector3d pos;
        indexToPos(Eigen::Vector3i(x, y, z), pos);
        if (pos(2) > mp_.visualization_truncate_height_)
          continue;

        pt.x = pos(0);
        pt.y = pos(1);
        pt.z = pos(2);
        cloud.push_back(pt);
      }

  // for (double x = 0; x <= 1; x = x+0.1)
  //   for (double y = -1; y <= 0;  y = y+1)
  //     for (double  z = 0; z <= 2;  z = z+0.1)
  //     {
  //       pt.x = x;
  //       pt.y = y;
  //       pt.z = z;
  //       cloud.push_back(pt);
  //     }
  cloud.width = cloud.points.size();
  cloud.height = 1;
  cloud.is_dense = true;
  cloud.header.frame_id = mp_.frame_id_;
  sensor_msgs::PointCloud2 cloud_msg;

  pcl::toROSMsg(cloud, cloud_msg);
  cloud_msg.header.stamp = mapstamp;
  map_inf_pub_.publish(cloud_msg);

  // ROS_INFO("pub map");
}

bool GridMap::odomValid() { return md_.has_odom_; }

bool GridMap::hasDepthObservation() { return md_.has_first_depth_; }

Eigen::Vector3d GridMap::getOrigin() { return mp_.map_origin_; }

// int GridMap::getVoxelNum() {
//   return mp_.map_voxel_num_[0] * mp_.map_voxel_num_[1] * mp_.map_voxel_num_[2];
// }

void GridMap::getRegion(Eigen::Vector3d &ori, Eigen::Vector3d &size)
{
  ori = mp_.map_origin_, size = mp_.map_size_;
}

void GridMap::extrinsicCallback(const nav_msgs::OdometryConstPtr &odom)
{
  Eigen::Quaterniond cam2body_q = Eigen::Quaterniond(odom->pose.pose.orientation.w,
                                                     odom->pose.pose.orientation.x,
                                                     odom->pose.pose.orientation.y,
                                                     odom->pose.pose.orientation.z);
  Eigen::Matrix3d cam2body_r_m = cam2body_q.toRotationMatrix();
  md_.cam2body_.block<3, 3>(0, 0) = cam2body_r_m;
  md_.cam2body_(0, 3) = odom->pose.pose.position.x;
  md_.cam2body_(1, 3) = odom->pose.pose.position.y;
  md_.cam2body_(2, 3) = odom->pose.pose.position.z;
  md_.cam2body_(3, 3) = 1.0;
}

void GridMap::depthOdomCallback(const sensor_msgs::ImageConstPtr &img,
                                const nav_msgs::OdometryConstPtr &odom)
{
  /* get pose */
  Eigen::Quaterniond body_q = Eigen::Quaterniond(odom->pose.pose.orientation.w,
                                                 odom->pose.pose.orientation.x,
                                                 odom->pose.pose.orientation.y,
                                                 odom->pose.pose.orientation.z);
  Eigen::Matrix3d body_r_m = body_q.toRotationMatrix();
  Eigen::Matrix4d body2world;
  body2world.block<3, 3>(0, 0) = body_r_m;
  body2world(0, 3) = odom->pose.pose.position.x;
  body2world(1, 3) = odom->pose.pose.position.y;
  body2world(2, 3) = odom->pose.pose.position.z;
  body2world(3, 3) = 1.0;

  Eigen::Matrix4d cam_T = body2world * md_.cam2body_;
  md_.camera_pos_(0) = cam_T(0, 3);
  md_.camera_pos_(1) = cam_T(1, 3);
  md_.camera_pos_(2) = cam_T(2, 3);
  md_.camera_r_m_ = cam_T.block<3, 3>(0, 0);

  /* get depth image */
  cv_bridge::CvImagePtr cv_ptr;
  cv_ptr = cv_bridge::toCvCopy(img, img->encoding);
  if (img->encoding == sensor_msgs::image_encodings::TYPE_32FC1)
  {
    (cv_ptr->image).convertTo(cv_ptr->image, CV_16UC1, mp_.k_depth_scaling_factor_);
  }
  cv_ptr->image.copyTo(md_.depth_image_);

  md_.occ_need_update_ = true;
  md_.flag_use_depth_fusion = true;
}
