#include "utils/print.h"
#include "state/Propagator.h"
#include "utils/sensor_data.h"
#include <opencv2/core/core.hpp>
#include "cam/CamRadtan.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_msckf;

int main(){
    std::string gst_pipeline = 
        "libcamerasrc ! "
        "video/x-raw,width=320,height=240,framerate=30/1 ! "
        "videoconvert ! "
        "video/x-raw, format=GRAY8 ! "
        "appsink drop=true sync=false";
    
    cv::VideoCapture cap(gst_pipeline, cv::CAP_GSTREAMER);

    if(!cap.isOpened()){
        PRINT_ERROR(RED "Unable to open a GStreamer feed!\n" RESET);
        return EXIT_FAILURE;
    }

    while(true){
        if(cv::waitKey(10) == 27){
            break;
        }

        cv::Mat frame;
        cap >> frame;

        if(!frame.empty()){
            ov_core::CameraData message;
            message.timestamp = std::chrono::system_clock::now();
            message.sensor_ids.push_back(0);
            message.image.push_back(frame);
            message.masks.push_back(cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1));
        }
    }

    return 0;
}
