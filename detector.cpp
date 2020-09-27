#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <dlib/matrix.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>

#include "detector.h"

template <long num_filters, typename SUBNET> using con5d = dlib::con<num_filters,5,5,2,2,SUBNET>;
template <long num_filters, typename SUBNET> using con5  = dlib::con<num_filters,5,5,1,1,SUBNET>;

template <typename SUBNET> using downsampler  = dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<16,SUBNET>>>>>>>>>;
template <typename SUBNET> using rcon5  = dlib::relu<dlib::affine<con5<45,SUBNET>>>;

using DetectorNet = dlib::loss_mmod<dlib::con<1,9,9,1,1,rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

class Detector {

public:
    Detector(const char* model_path) {
        dlib::deserialize(model_path) >> net;
    }

    auto detect(const dlib::matrix<dlib::rgb_pixel>& image) {
        std::lock_guard<std::mutex> lock(net_mutex);
        return net(image);
    }

    auto detect(const std::vector<dlib::matrix<dlib::rgb_pixel>>& images) {
        std::lock_guard<std::mutex> lock(net_mutex);
        return net(images);
    }

private:
    DetectorNet net;
    std::mutex net_mutex;
};



detector_init_result* detector_init(char* model_file_path) {
    detector_init_result* result = (detector_init_result*)malloc(sizeof(detector_init_result));

    try {
        Detector* detector = new Detector(model_file_path);
        result->detector = (void*)detector;
        result->error_message = NULL;
    } catch (std::exception& e) {
        result->detector = NULL;
        result->error_message = strdup(e.what());
    }

    return result;
}

void detector_free(void* detector) {
    delete (Detector*)(detector);
}

detector_detect_result* detector_detect(void* detector, void* image) {
    detector_detect_result* result = (detector_detect_result*)malloc(sizeof(detector_detect_result));

    try {
        dlib::matrix<dlib::rgb_pixel> dlib_image;

        cv::Mat* opencv_image = (cv::Mat*)image;

        if (opencv_image->channels() > 1) {
            dlib::assign_image(dlib_image, dlib::cv_image<dlib::bgr_pixel>(*opencv_image));
        } else {
            dlib::assign_image(dlib_image, dlib::cv_image<uchar>(*opencv_image));
        }

        auto detections_raw = ((Detector*)(detector))->detect(dlib_image);

        detection* detections = (detection*)calloc(detections_raw.size(), sizeof(detection));
    
        for(long unsigned int i = 0; i < detections_raw.size(); i++) {
            detections[i].region.min.x = detections_raw[i].rect.left();
            detections[i].region.min.y = detections_raw[i].rect.top();
            detections[i].region.max.x = detections_raw[i].rect.right();
            detections[i].region.max.y = detections_raw[i].rect.bottom();
            detections[i].confidence = detections_raw[i].detection_confidence;
        }

        result->detections_count = detections_raw.size();
        result->detections = detections;
        result->error_message = NULL;

    } catch (std::exception& e) {
        result->detections_count = 0;
        result->detections = NULL;
        result->error_message = strdup(e.what());
    }

    return result;
}

detector_batch_detect_result* detector_batch_detect(void* detector, void* images, int images_count) {
    detector_batch_detect_result* result = (detector_batch_detect_result*)malloc(sizeof(detector_batch_detect_result));

    try {
        std::vector<dlib::matrix<dlib::rgb_pixel>> dlib_images(images_count);

        cv::Mat** opencv_images = (cv::Mat**)images;

        for (int i = 0; i < images_count; i++) {
            cv::Mat* opencv_image = opencv_images[i];
            dlib::matrix<dlib::rgb_pixel> dlib_image;

            if (opencv_image->channels() > 1) {
                dlib::assign_image(dlib_image, dlib::cv_image<dlib::bgr_pixel>(*opencv_image));
            } else {
                dlib::assign_image(dlib_image, dlib::cv_image<uchar>(*opencv_image));
            }

            dlib_images[i] = dlib_image;
        }

        auto batch_detections_raw = ((Detector*)(detector))->detect(dlib_images);

        batch_detection* batch_detections = (batch_detection*)calloc(batch_detections_raw.size(), sizeof(batch_detection));

        for(unsigned long i = 0; i < batch_detections_raw.size(); i++) {

            detection* detections = (detection*)calloc(batch_detections_raw[i].size(), sizeof(detection));

            for(unsigned long j = 0; j < batch_detections_raw[i].size(); j++) {
                detections[j].region.min.x = batch_detections_raw[i][j].rect.left();
                detections[j].region.min.y = batch_detections_raw[i][j].rect.top();
                detections[j].region.max.x = batch_detections_raw[i][j].rect.right();
                detections[j].region.max.y = batch_detections_raw[i][j].rect.bottom();
                detections[j].confidence = batch_detections_raw[i][j].detection_confidence;
            }

            batch_detections[i].detections = detections;
            batch_detections[i].detections_count = batch_detections_raw[i].size();
        }

        result->detections = batch_detections;
        result->detections_count = batch_detections_raw.size();
        result->error_message = NULL;

    } catch (std::exception& e) {
        result->detections = NULL;
        result->detections_count = 0;
        result->error_message = strdup(e.what());
    }

    return result;
}