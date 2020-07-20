#include "feature_tracker.h"


template<CvMat>
class IFisheyeFeatureFrameTracker: public IFeatureFrameTracker {
protected:
    std::vector<FisheyeUndist> fisheys_undists;
    void drawTrackFisheye(const cv::Mat & img_up, const cv::Mat & img_down, 
                            CvMat imUpTop,
                            CvMat imDownTop,
                            CvMat imUpSide, 
                            CvMat imDownSide);
public:
    IFisheyeFeatureFrameTracker() {}

    vector<cv::Point3f> undistortedPtsTop(vector<cv::Point2f> &pts, FisheyeUndist & fisheye);
    vector<cv::Point3f> undistortedPtsSide(vector<cv::Point2f> &pts, FisheyeUndist & fisheye, bool is_downward);

    // void setMaskFisheye();
    // cv::Mat setMaskFisheye(cv::Size shape, vector<cv::Point2f> & cur_pts, vector<int> & track_cnt, vector<int> & ids);
    // void setMaskFisheye(cv::cuda::GpuMat & mask, cv::Size shape, vector<cv::Point2f> & cur_pts, 
    //     vector<int> & track_cnt, vector<int> & ids);

    void addPointsFisheye();

    cv::Size top_size;
    cv::Size side_size;


};



class FisheyeFeatureFrameTrackerCuda: public IFisheyeFeatureTracker<cv::cuda::GpuMat> {

public:
    FeatureFrame trackImage_fisheye(double _cur_time, 
        const std::vector<cv::cuda::GpuMat> & fisheye_imgs_up, 
        const std::vector<cv::cuda::GpuMat> & fisheye_imgs_down, bool is_blank_init = false);

    vector<cv::Point2f> opticalflow_track(cv::cuda::GpuMat & cur_img, 
                        cv::cuda::GpuMat & prev_img, vector<cv::Point2f> & prev_pts, 
                        vector<int> & ids, vector<int> & track_cnt,
                        bool is_lr_track, vector<cv::Point2f> prediction_points = vector<cv::Point2f>());

};

class FisheyeFeatureFrameTrackerOMP:: public IFisheyeFeatureTracker<cv::Mat> {
public:
    FeatureFrame trackImage(double _cur_time, const cv::Mat &_img, const cv::Mat &_img1 = cv::Mat());

    FeatureFrame trackImage_fisheye(double _cur_time, const std::vector<cv::Mat> & fisheye_imgs_up, const std::vector<cv::Mat> & fisheye_imgs_down);
    
    vector<cv::Point2f> opticalflow_track(vector<cv::Mat> * cur_pyr, 
                        vector<cv::Mat> * prev_pyr, vector<cv::Point2f> & prev_pts, 
                        vector<int> & ids, vector<int> & track_cnt, vector<cv::Point2f> prediction_points = vector<cv::Point2f>()) const;

    vector<cv::Point2f> opticalflow_track(cv::Mat & cur_img, vector<cv::Mat> * cur_pyr, 
                        cv::Mat & prev_img, vector<cv::Mat> * prev_pyr, vector<cv::Point2f> & prev_pts, 
                        vector<int> & ids, vector<int> & track_cnt, vector<cv::Point2f> prediction_points = vector<cv::Point2f>()) const;
}
