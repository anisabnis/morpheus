// Header file that consists of super-resolution utilities
// nginx header files should go before other, because they define 64-bit off_t

extern "C" {
  #include <nginx.h>
  #include <ngx_config.h>
  #include <ngx_core.h>
  #include <ngx_http.h>
  #include <libavcodec/avcodec.h>
  #include <libswscale/swscale.h>
  #include <libavutil/imgutils.h>
  #include <libavutil/opt.h>
  #include <libavformat/avformat.h>
  #include <libavformat/avio.h>
  #include <libavutil/file.h>
  #include <libavutil/dict.h>
}

#include <iostream>
#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn_superres;

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

typedef struct {
  ngx_int_t width;
  ngx_int_t height;
} ngx_dims_t;

// Add after existing struct/function declarations
struct SRResult {
    cv::Mat Y;    // uint8, out_H × out_W
    cv::Mat Cb;   // uint8, out_H/2 × out_W/2
    cv::Mat Cr;   // uint8, out_H/2 × out_W/2
};

// ---------------------------------------------------------------------------
// ORT session wrapper — constructed once per request from raw .onnx bytes.
// Holds env, session, and the I/O name strings.
// ---------------------------------------------------------------------------
struct OrtSuperResSession {
    Ort::Env     env;
    Ort::Session session;
    std::string  input_name;
    std::string  output_name;

    OrtSuperResSession(const void *model_data, size_t model_data_len);
};

Mat upscaleImage(Mat img, string modelName, string modelPath, int scale, ngx_http_request_t *r);
ngx_dims_t ngx_http_superresolution_file_cache_get_dimensions(ngx_str_t resolution, ngx_log_t *log);

// Legacy OpenCV-DNN signatures (kept so video.cpp compiles unchanged).
Mat upscaleImageFromONNX(Mat img, Net net, int scale, ngx_http_request_t *r);
std::vector<cv::Mat> upscaleBatchFromONNX(const std::vector<cv::Mat> &images, cv::dnn::Net &net, float scale, ngx_http_request_t *r);

// ORT-native entry points used by the video pipeline.
cv::Mat upscaleImageORT(const cv::Mat &imageBGR, OrtSuperResSession &ort, float scale, ngx_http_request_t *r);
std::vector<cv::Mat> upscaleBatchORT(const std::vector<cv::Mat> &images, OrtSuperResSession &ort, float scale, ngx_http_request_t *r);
OrtSuperResSession &get_or_create_ort_session(const std::string &uri, const void *model_data, size_t model_data_len, ngx_http_request_t *r);
SRResult upscaleImageORT_YUV(const cv::Mat &Y_u8, const cv::Mat &Cb_u8, const cv::Mat &Cr_u8,
                              OrtSuperResSession &ort, float scale, ngx_http_request_t *r);
// Pre-loads all espcn.onnx models found under /usr/local/data/models/ into
// the ORT session cache. Call once from the nginx worker init process hook.
extern "C" void preload_all_ort_sessions(ngx_log_t *log);
