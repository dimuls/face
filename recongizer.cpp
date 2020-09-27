#include <dlib/dnn.h>
#include <dlib/opencv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>

#include "recognizer.h"

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

const int IMAGE_SIZE = 150;

using RecognizerNet = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<IMAGE_SIZE>
                            >>>>>>>>>>>>;

static std::vector<dlib::matrix<dlib::rgb_pixel>> jitter_image(const dlib::matrix<dlib::rgb_pixel>& image, int count) {
    thread_local dlib::rand random;

    std::vector<dlib::matrix<dlib::rgb_pixel>> crops;
    for (int i = 0; i < count; i++) {
        crops.push_back(dlib::jitter_image(image, random));
    }

    return crops;
}

class Recognizer {

public:
    Recognizer(const char* shaper_model_file_path, const char* recognizer_model_file_path) {
        dlib::deserialize(shaper_model_file_path) >> shaper;
        dlib::deserialize(recognizer_model_file_path) >> net;
    }

    dlib::matrix<float,0,1> recognize(const dlib::matrix<dlib::rgb_pixel>& image, dlib::rectangle face_location, double padding, int jittering) {
        auto shape = shaper(image, face_location);

        dlib::matrix<dlib::rgb_pixel> chip;
        dlib::extract_image_chip(image, dlib::get_face_chip_details(shape, IMAGE_SIZE, padding), chip);

        dlib::matrix<float,0,1> descriptor;

        if (jittering > 0) {
            auto chips = jitter_image(chip, jittering);
            std::vector<dlib::matrix<float,0,1>> descriptors;
            {
                std::lock_guard lock(net_mutex);
                descriptors = net(chips);
            }
            descriptor = dlib::mean(dlib::mat(descriptors));
        } else {
            std::lock_guard lock(net_mutex);
            descriptor = net(chip);
        }

        return descriptor;
    }

private:
    dlib::shape_predictor shaper;

    RecognizerNet net;
    std::mutex net_mutex;
};


recognizer_init_result* recognizer_init(char* shaper_model_file_path, char* recognizer_model_file_path) {
    recognizer_init_result* result = (recognizer_init_result*)malloc(sizeof(recognizer_init_result));

    try {
        Recognizer* recognizer = new Recognizer(shaper_model_file_path, recognizer_model_file_path);
        result->recognizer = (void*)recognizer;
        result->error_message = NULL;
    } catch (std::exception& e) {
        result->recognizer = NULL;
        result->error_message = strdup(e.what());
    }

    return result;
}

void recognizer_free(void* recognizer) {
    delete (Recognizer*)(recognizer);
}

recognizer_recognize_result* recognizer_recognize(void* recognizer, void* image, rectangle* face_location, double padding, int jittering) {
    recognizer_recognize_result* result = (recognizer_recognize_result*)malloc(sizeof(recognizer_recognize_result));

    try {
        dlib::matrix<dlib::rgb_pixel> dlib_image;

        cv::Mat* opencv_image = (cv::Mat*)image;

        if (opencv_image->channels() > 1) {
            dlib::assign_image(dlib_image, dlib::cv_image<dlib::bgr_pixel>(*opencv_image));
        } else {
            dlib::assign_image(dlib_image, dlib::cv_image<uchar>(*opencv_image));
        }

        dlib::rectangle dlib_face_location;

        dlib_face_location.set_left(face_location->min.x);
        dlib_face_location.set_top(face_location->min.y);
        dlib_face_location.set_right(face_location->max.x);
        dlib_face_location.set_bottom(face_location->max.y);

        auto dlib_descriptor = ((Recognizer*)(recognizer))->recognize(dlib_image, dlib_face_location, padding, jittering);

        float* descriptor = (float*)calloc(dlib_descriptor.nr(), sizeof(float));

        for (int i = 0; i < dlib_descriptor.nr(); i++) {
            descriptor[i] = dlib_descriptor(i, 0);
        }

        result->descriptor = descriptor;
        result->error_message = NULL;

    } catch (std::exception& e) {
        result->descriptor = NULL;
        result->error_message = strdup(e.what());
        return result;
    }

    return result;
}