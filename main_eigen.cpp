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

Eigen::Matrix3f rot_pcd_to_world (sl::Vector3<float> & euler_rad){
	float cphi, sphi, cpsi, spsi;
	cphi = cos(euler_rad[0]);
	sphi = sin(euler_rad[0]);
	cpsi = cos(euler_rad[2]);
	spsi = sin(euler_rad[2]);
	Eigen::Matrix3f xrot, zrot, axis_align;
	// Construct axis align matrix. Just to swap camera axis to match world NED convention
	axis_align(0,2) = 1;
	axis_align(1,0) = 1;
	axis_align(2,1) = 1;
	// Constuct xrot matrix
	xrot(0,0) = 1;
	xrot(1,1) = cphi;
	xrot(1,2) = -sphi;
	xrot(2,1) = sphi;
	xrot(2,2) = cphi;
	// Construct yrot matrix
	zrot(0,0) = cpsi;
	zrot(0,1) = -spsi;
	zrot(1,0) = spsi;
	zrot(1,1) = cpsi;
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

Eigen::Vector4f ground_segmentation (Eigen::MatrixXf & xyz_loc, float & dist_thres_m, float & cang_thres_rad, int &  max_iter, int & num_valid_point){
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
	std::uniform_int_distribution<int> gen_ind(0, num_valid_point - 1);
	while (iter < max_iter){
		census = 0;
		a = gen_ind(seed);
		b = gen_ind(seed);
		c = gen_ind(seed);
		std::cout<<xyz_loc.col(a)<<"\n";
		Eigen::Vector4f temp = calc_plane(xyz_loc.col(a), xyz_loc.col(b), xyz_loc.col(c));
		if (calc_ctilt (temp) < cang_thres_rad){
			continue;
		}
		int num_test = 100;
		for (int i = 0; i < num_test; i++){
			int test_ind = gen_ind(seed);
			float test_dist = calc_dist(temp, xyz_loc.col(test_ind));
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

// I don't like how I am sorting now. Need to analyze if this is needed or find better way
int sort_xyz_class_matrix (Eigen::MatrixXf & xyz_loc, int & num_valid_point){
	Eigen::MatrixXf temp = xyz_loc;
	int j = 0;
	for (int i = 0; i < num_valid_point; i++){
		if (xyz_loc(2,i) == 0){
			temp.col(j) = xyz_loc.col(i);
			j++;
		}
	}
	for (int i = 0; i < num_valid_point; i++){
		if (xyz_loc(2,i) == 100){
			temp.col(j) = xyz_loc.col(i);
			j++;
		}
	}
	for (int i = 0; i < num_valid_point; i++){
		if (xyz_loc(2,i) == 200){
			temp.col(j) = xyz_loc.col(i);
			j++;
		}
	}
	xyz_loc = temp;
	return 0;
}

// Classify points into 3 types: ground, negative_obstacle, positive_obstacle for a 2D scan view
// Replace the 3rd collumn value in the XYZ matrix with: 0 - ground, 100 - negative_obstacle, 200 - positive_obstacle
// If positive obstacle is higher than some threshold, we can just ignore it to save processing time, and data bandwidth
// Return number of valid points
int obstacle_class (Eigen::MatrixXf & xyz_loc, Eigen::Vector4f & ground_param, float & dist_thres_m, int & num_valid_point){
	int k = 0;
	for (int i = 0; i < num_valid_point; i++) {
		float dist = calc_dist (ground_param, xyz_loc.col(i));
		// Is ground point
		if (dist < dist_thres_m && dist > dist_thres_m){
			xyz_loc(2,i) = 0;
			k++;
		}
		// Is negative obstacles
		else if(dist >= dist_thres_m && dist < 0.2){
			xyz_loc(2,i) = 200;
			k++;
		}
		// Is positive obstacles
		else if(dist <= -dist_thres_m && dist > -0.2) { 
			xyz_loc(2,i) = 100;
			k++;
		}
		else{
			xyz_loc(2,i) = 300;
		}
	}
	sort_xyz_class_matrix (xyz_loc, num_valid_point);
	return k;
}

int main (int argc, char **argv) {
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

	
	std::cout<<mat_len;
	sl::Pose pose;
	sl::Mat img, pcd;
	sl::float4 point3D;
	Eigen::MatrixXf xyz_loc (3,mat_len);
	float dist_thres = 0.08f;
	float c_angle_thres = 0.9994f;
	int max_seg_iter = 400;
	sl::Vector3<float> zed_euler_rad, zed_pos_m;
	auto t1 = std::chrono::high_resolution_clock::now();
	while (true) {
		if (zed.grab() == sl::ERROR_CODE::SUCCESS) {
			zed.retrieveImage (img, sl::VIEW::LEFT);
			zed.retrieveMeasure (pcd, sl::MEASURE::XYZ);
			zed.getPosition (pose, sl::REFERENCE_FRAME::WORLD);
			zed_euler_rad = pose.getEulerAngles();
			zed_pos_m = pose.getTranslation();
			Eigen::Matrix3f dcm = rot_pcd_to_world(zed_euler_rad);
			int k = 0;
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					pcd.getValue(i,j, &point3D);
					if (std::isfinite(point3D.x)){
						xyz_loc (0,k) = point3D.x * dcm(0,0) + point3D.y * dcm(0,1) + point3D.z * dcm(0,2);
						xyz_loc (1,k) = point3D.x * dcm(1,0) + point3D.y * dcm(1,1) + point3D.z * dcm(1,2);
						xyz_loc (2,k) = point3D.x * dcm(2,0) + point3D.y * dcm(2,1) + point3D.z * dcm(2,2);
						k++;
					} 
				}
			}	
			//Eigen::Vector4f ground_plane = ground_segmentation (xyz_loc, dist_thres, c_angle_thres, max_seg_iter, k);
			//int num_interest_point = obstacle_class (xyz_loc, ground_plane, dist_thres, k);
			auto t2 = std::chrono::high_resolution_clock::now();
			std::chrono::duration <double, std::milli> ms_double = t2 - t1;
			t1 = t2;
			std::cout << ms_double.count() << " ms\n";  	
		}

	}

	zed.disablePositionalTracking();
	zed.close();
	return 0;
}

