// Tuan Luong
// 9-20-2023
//

#include <sl/Camera.hpp>
#include <chrono>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <random>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

Eigen::Matrix3f rot_pcd_to_world (sl::Vector3<float> & euler_rad){
	float cphi, sphi, cpsi, spsi;
	cphi = cos(euler_rad[0]);
	sphi = sin(euler_rad[0]);
	cpsi = cos(euler_rad[2]);
	spsi = sin(euler_rad[2]);
	Eigen::Matrix3f xrot, zrot, axis_align;
	// Construct axis align matrix. Just to swap camera axis to match world NED convention
	axis_align(0,0) = 0;
	axis_align(0,1) = 0;
	axis_align(0,2) = 1;
	axis_align(1,0) = 1;
	axis_align(1,1) = 0;
	axis_align(1,2) = 0;
	axis_align(2,0) = 0;
	axis_align(2,1) = 1;
	axis_align(2,2) = 0;
	// Constuct xrot matrix
	xrot(0,0) = 1;
	xrot(0,1) = 0;
	xrot(0,2) = 0;
	xrot(1,0) = 0;
	xrot(1,1) = cphi;
	xrot(1,2) = -sphi;
	xrot(2,0) = 0;
	xrot(2,1) = sphi;
	xrot(2,2) = cphi;
	// Construct yrot matrix
	zrot(0,0) = cpsi;
	zrot(0,1) = -spsi;
	zrot(0,2) = 0;
	zrot(1,0) = spsi;
	zrot(1,1) = cpsi;
	zrot(1,2) = 0;
	zrot(2,0) = 0;
	zrot(2,1) = 0;
	zrot(2,2) = 1;

	// Construct DCM matrix
	return  axis_align * zrot * xrot;
}

Eigen::Vector4f calc_plane (Eigen::Vector3f point1, Eigen::Vector3f point2, Eigen::Vector3f point3) {
	Eigen::Vector3f a_vec, b_vec, n;
	a_vec = point2 - point1;
	b_vec = point3 - point1;
	n = a_vec.cross(b_vec);
	n.normalize();
	float d = - n.dot(point1);
	Eigen::Vector4f ret;
	ret << n(0), n(1), n(2), d;
	return ret;
}

float calc_ctilt (Eigen::Vector4f & plane_param){
	Eigen::Vector3f test, vertical;
	test << plane_param(0), plane_param(1), plane_param(2);
	vertical << 0, 0, 1;
	return test.dot(vertical);
}

float calc_dist (Eigen::Vector4f plane_param, Eigen::Vector3f point) {
	Eigen::Vector3f n;
	n << plane_param(0), plane_param(1), plane_param(2);
	return n.dot(point) + plane_param(3);
}

Eigen::Vector4f ground_segmentation (pcl::PointCloud<pcl::PointXYZ>::ConstPtr pcl, float & dist_thres_m, float & cang_thres_rad, int &  max_iter){
	// Implement RANSAC for plane fitting
	// Improved by using pre rotated xyz array in world frame 
	// so plane limit can be implemented and reduce amount of impractical samples
	Eigen::Vector4f plane_param;
	plane_param << 0,0,1,0;
	int current_max_census, census, iter;
	current_max_census = 0;
	iter = 0;
	int a,b,c;
	std::random_device rd;
	std::mt19937 seed(rd());
	std::uniform_int_distribution<int> gen_ind(0, pcl->width - 1);
	while (iter < max_iter){
		census = 0;
		a = gen_ind(seed);
		b = gen_ind(seed);
		c = gen_ind(seed);
		Eigen::Vector3f vec_a, vec_b, vec_c, vec_d;
		vec_a(0) = pcl->points[a].x; vec_a(1) = pcl->points[a].y; vec_a(2) = pcl->points[a].z;
		vec_b(0) = pcl->points[b].x; vec_b(1) = pcl->points[b].y; vec_b(2) = pcl->points[b].z;
		vec_c(0) = pcl->points[c].x; vec_c(1) = pcl->points[c].y; vec_c(2) = pcl->points[c].z;
		Eigen::Vector4f temp = calc_plane(vec_a, vec_b, vec_c);
		if (calc_ctilt (temp) < cang_thres_rad){
			iter++;
			continue;
		}
		int num_test = 1000;
		for (int i = 0; i < num_test; i++){
			int test_ind = gen_ind(seed);
			vec_d(0) = pcl->points[test_ind].x; vec_d(1) = pcl->points[test_ind].y; vec_d(2) = pcl->points[test_ind].z;
			float test_dist = calc_dist(temp, vec_d);
			if (abs(test_dist) < dist_thres_m){
				census++;
			}
		}
		if (census > current_max_census){
			plane_param = temp;
			current_max_census = census;
		}
		iter++;
	}
	return plane_param;
}

// Classify points into 3 types: ground, negative_obstacle, positive_obstacle for a 2D scan view
// If positive obstacle is higher than some threshold, we can just ignore it to save processing time, and data bandwidth

void obstacle_class (pcl::PointCloud<pcl::PointXYZ>::ConstPtr pcl, pcl::PointCloud<pcl::PointXYZRGB>::Ptr gnd_pcl,pcl::PointCloud<pcl::PointXYZRGB>::Ptr pos_pcl, 
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr neg_pcl, Eigen::Vector4f & ground_param, float & dist_thres_m){
	Eigen::Vector3f point;
	pcl::PointXYZRGB pclpoint;
	for (int i = 0; i < pcl->width; i++){
		point(0) = pcl->points[i].x; point(1) = pcl->points[i].y; point(2) = pcl->points[i].z;
		float dist = calc_dist (ground_param, point);
		//std::cout<<dist<<"\n";
		// Is ground
		if (dist < dist_thres_m && dist > -dist_thres_m){
			pclpoint.x = point(0); pclpoint.y = point(1); pclpoint.z = point(2);
			// pack r/g/b into rgb
			std::uint8_t r = 0, g = 255, b = 0;
			std::uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
			pclpoint.rgb = *reinterpret_cast<float*>(&rgb);
			gnd_pcl -> push_back(pclpoint);
		}
		// Is negative
		else if (dist < 0.2 && dist >= dist_thres_m){
			pclpoint.x = point(0); pclpoint.y = point(1); pclpoint.z = point(2);
			std::uint8_t r = 0, g = 0, b = 255;
			std::uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
			pclpoint.rgb = *reinterpret_cast<float*>(&rgb);
			neg_pcl -> push_back(pclpoint);
		}
		else if (dist <= -dist_thres_m && dist > -0.2){
			pclpoint.x = point(0); pclpoint.y = point(1); pclpoint.z = point(2);
			std::uint8_t r = 255, g = 0, b = 0;
			std::uint32_t rgb = ((std::uint32_t)r << 16 | (std::uint32_t)g << 8 | (std::uint32_t)b);
			pclpoint.rgb = *reinterpret_cast<float*>(&rgb);
			pos_pcl -> push_back(pclpoint);
		}
	}
}

int main (int argc, char **argv) {
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcl(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr gnd_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pos_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr neg_pcl(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointXYZ pclpoint;
	sl::Camera zed;
	sl::InitParameters init_param;
	init_param.camera_resolution = sl::RESOLUTION::VGA;
	init_param.depth_mode = sl::DEPTH_MODE::ULTRA;
	init_param.camera_fps = 15;
	init_param.coordinate_units = sl::UNIT::METER;
	init_param.depth_minimum_distance = 0.4;
	init_param.depth_maximum_distance = 3.0;
	int mat_len, width, height;
	// TODO ADD MORE RESOLUTION OPTIONS
	if (init_param.camera_resolution == sl::RESOLUTION::HD1080){
		width = 1920;
		height = 1080;
	}
	if (init_param.camera_resolution == sl::RESOLUTION::HD720) {
		width = 1280;
		height = 720;
	}
	else if (init_param.camera_resolution == sl::RESOLUTION::VGA){
		width = 672;
		height = 376;
	}
	mat_len = width * height;

	auto returned_state = zed.open(init_param);
	if(returned_state != sl::ERROR_CODE::SUCCESS) {
		std::cout << "ERROR: " << returned_state << ", exit program.\n";
		return 1;
	}

	sl::PositionalTrackingParameters tracking_param;
	returned_state = zed.enablePositionalTracking(tracking_param);
	if (returned_state != sl::ERROR_CODE::SUCCESS) {
		std::cout << "ERROR: " << returned_state << ", exit program.\n";
		return 1;
	}

	sl::Pose pose;
	sl::Mat img, pcd;
	sl::float4 slpoint;
	Eigen::MatrixXf xyz_loc (3,mat_len);
	float dist_thres = 0.05f;
	float c_angle_thres = 0.9994f;
	int max_seg_iter = 1000;
	sl::Vector3<float> zed_euler_rad, zed_pos_m;
	auto t1 = std::chrono::high_resolution_clock::now();
	while(true){
		if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
			auto t1 = std::chrono::high_resolution_clock::now();
			zed.retrieveImage (img, sl::VIEW::LEFT);
			zed.retrieveMeasure (pcd, sl::MEASURE::XYZ);
			zed.getPosition (pose, sl::REFERENCE_FRAME::WORLD);
			zed_euler_rad = pose.getEulerAngles();
			zed_pos_m = pose.getTranslation();
			Eigen::Matrix3f dcm = rot_pcd_to_world(zed_euler_rad);
			int k = 0;
			pcl->clear();
			gnd_pcl -> clear();
			neg_pcl -> clear();
			pos_pcl -> clear();
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					pcd.getValue(i,j, &slpoint);
					if (std::isfinite(slpoint.x)){
						pclpoint.x = slpoint.x * dcm(0,0) + slpoint.y * dcm(0,1) + slpoint.z * dcm(0,2);
						pclpoint.y = slpoint.x * dcm(1,0) + slpoint.y * dcm(1,1) + slpoint.z * dcm(1,2);
						pclpoint.z = slpoint.x * dcm(2,0) + slpoint.y * dcm(2,1) + slpoint.z * dcm(2,2);
						pcl->push_back(pclpoint);
					} 
				}
			}
			Eigen::Vector4f ground_plane = ground_segmentation (pcl, dist_thres, c_angle_thres, max_seg_iter);
			obstacle_class(pcl, gnd_pcl, pos_pcl, neg_pcl, ground_plane, dist_thres);
			//std::cout << ground_plane(3) << "\n";
			auto t2 = std::chrono::high_resolution_clock::now();
			std::chrono::duration <double, std::milli> ms_double = t2 - t1;
			std::cout << ms_double.count() << "ms \n";
			//std::cout << gnd_pcl->width;
		}	
	}

	
	zed.disablePositionalTracking();
	zed.close();
	return 0;
}

