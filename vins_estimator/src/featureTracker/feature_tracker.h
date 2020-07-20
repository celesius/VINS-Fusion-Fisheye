/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *
 * Author: Qin Tong (qintonguav@gmail.com)
 *******************************************************/

#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "../utility/opencv_cuda.h"

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "../estimator/parameters.h"
#include "../utility/tic_toc.h"

#ifdef WITH_VWORKS
#include "vworks_feature_tracker.hpp"
#endif

#define PYR_LEVEL 3
#define WIN_SIZE cv::Size(21, 21)

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);
bool inBorder(const cv::Point2f &pt, cv::Size shape);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

typedef Eigen::Matrix<double, 8, 1> TrackFeatureNoId;
typedef pair<int, TrackFeatureNoId> TrackFeature;
typedef vector<TrackFeature> FeatureFramenoId;
typedef map<int, FeatureFramenoId> FeatureFrame;
class Estimator;
class FisheyeUndist;
map<int, cv::Point2f> pts_map(vector<int> ids, vector<cv::Point2f> cur_pts);

double distance(cv::Point2f pt1, cv::Point2f pt2);


template<CvMat>
class PinholeFeatureTracker {
//This class is for track points on a pinhole camera
protected:
    int base_id = 0;
    CvMat prev_img;
    cv::Mat mask;
    std::vector<cv::Mat> * prev_pyr = nullptr;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> predict_pts;
    vector<cv::Point2f> cur_pts;
    vector<cv::Point2f> cur_un_pts;
    vector<cv::Point2f> prev_un_pts;

    vector<cv::Point2f> pts_velocity;
    vector<cv::Point3f> cur_velocity;

    vector<int> ids;
    map<int, int> pts_status;
    map<int, int> prevPtsMap;
    map<int, int> cur_un_pts_map;
    set<int> removed_pts;
    vector<int> track_cnt;
    vector<camodocal::CameraPtr> m_camera;

public:
    PinholeFeatureTracker() {}
    virtual void track_image(CvMat img);
    virtual void track_current_to_right(CvMat img);
    virtual void setMask() {}

    bool inBorder(const cv::Point2f &pt);
    bool inBorder(const cv::Point2f &pt, cv::Size shape) const;
    
    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
        map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);

    vector<cv::Point3f> ptsVelocity3D(vector<int> &ids, vector<cv::Point3f> &pts, 
        map<int, cv::Point3f> &cur_id_pts, map<int, cv::Point3f> &prev_id_pts);


}

template<CvMat>
class IFeatureFrameTracker {


protected:
    int row, col;

    IFeatureTracker() {}
    Estimator * estimator = nullptr;
    virtual void setup_feature_frame(FeatureFrame & ff, 
        vector<int> ids, vector<cv::Point2f> cur_pts, vector<cv::Point3f> cur_un_pts, vector<cv::Point3f> cur_pts_vel, int camera_id);
    FeatureFrame setup_feature_frame();
    virtual void readIntrinsicParameter(const vector<string> &calib_file) = 0;
    virtual FeatureFrame trackImage(double _cur_time, const CvMat &_img, const CvMat &_img1) = 0;
    virtual FeatureFrame trackImage_fisheye(double _cur_time, const std::vector<cv::Mat> & fisheye_imgs_up, const std::vector<cv::Mat> & fisheye_imgs_down) = 0;

    virtual void setPrediction(map<int, Eigen::Vector3d> &predictPts) {};
    virtual void removeOutliers(set<int> &removePtsIds) {};

    double cur_time;
    double prev_time;

    void detectPoints(const CvMat & img, vector<cv::Point2f> & n_pts, 
        vector<cv::Point2f> & cur_pts, int require_pts, const CvMat & mask = CvMat());


    void setFeatureStatus(int feature_id, int status) {
        this->pts_status[feature_id] = status;
        if (status < 0) {

        }
    }

}





// class StereoFeatureTracker : public IFeatureFrameTracker<cv::Mat> {
//     void addPoints();
//     void undistortedPoints();
//     vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);
//     void drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
//         vector<int> &curLeftIds,
//         vector<cv::Point2f> &curLeftPts, 
//         vector<cv::Point2f> &curRightPts,
//         map<int, cv::Point2f> &prevLeftPtsMap);
//     void drawTrackImage(cv::Mat & img, vector<cv::Point2f> pts, vector<int> ids, map<int, cv::Point2f> prev_pts);
//     void setPrediction(map<int, Eigen::Vector3d> &predictPts);
//     void removeOutliers(set<int> &removePtsIds);
//     virtual void setPrediction(map<int, Eigen::Vector3d> &predictPts) override;
//     virtual void removeOutliers(set<int> &removePtsIds) override;

// }


