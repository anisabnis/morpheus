#include "ngx_http_superresolution_util.hpp"
#include <mutex>
#include <unordered_map>
#include <filesystem>
#include <fstream>

// ---------------------------------------------------------------------------
// YCrCb helpers (unchanged)
// ---------------------------------------------------------------------------

void preprocess_YCrCb(InputArray inpImg, OutputArray outImg)
{
    if ( inpImg.type() == CV_8UC1 )
    {
        inpImg.getMat().convertTo(outImg, CV_32F, 1.0 / 255.0);
    }
    else if ( inpImg.type() == CV_32FC1 )
    {
        inpImg.getMat().convertTo(outImg, CV_32F, 1.0 / 255.0);
    }
    else if ( inpImg.type() == CV_32FC3 )
    {
        Mat img_float;
        inpImg.getMat().convertTo(img_float, CV_32F, 1.0 / 255.0);
        cvtColor(img_float, outImg, COLOR_BGR2YCrCb);
    }
    else if ( inpImg.type() == CV_8UC3 )
    {
        Mat ycrcb;
        cvtColor(inpImg, ycrcb, COLOR_BGR2YCrCb);
        ycrcb.convertTo(outImg, CV_32F, 1.0 / 255.0);
    }
    else
    {
        CV_Error(Error::StsBadArg, String("Not supported image type: ") + typeToString(inpImg.type()));
    }
}

void reconstruct_YCrCb(InputArray inpImg, InputArray origImg, OutputArray outImg, int scale)
{
    if ( origImg.type() == CV_32FC3 )
    {
        Mat orig_channels[3];
        split(origImg.getMat(), orig_channels);

        Mat Cr, Cb;
        cv::resize(orig_channels[1], Cr, cv::Size(), scale, scale);
        cv::resize(orig_channels[2], Cb, cv::Size(), scale, scale);

        std::vector <Mat> channels;
        channels.push_back(inpImg.getMat());
        channels.push_back(Cr);
        channels.push_back(Cb);

        Mat merged_img;
        merge(channels, merged_img);

        Mat merged_8u_img;
        merged_img.convertTo(merged_8u_img, CV_8U, 255.0);

        cvtColor(merged_8u_img, outImg, COLOR_YCrCb2BGR);
    }
    else if ( origImg.type() == CV_32FC1 )
    {
        inpImg.getMat().convertTo(outImg, CV_8U, 255.0);
    }
    else
    {
        CV_Error(Error::StsBadArg, String("Not supported image type: ") + typeToString(origImg.type()));
    }
}

// Cache key: (video_name, quality, identifier)
struct OrtSessionKey {
    std::string video;
    std::string quality;
    std::string identifier;

    bool operator==(const OrtSessionKey &o) const {
        return video == o.video && quality == o.quality && identifier == o.identifier;
    }
};

struct OrtSessionKeyHash {
    size_t operator()(const OrtSessionKey &k) const {
        size_t h = std::hash<std::string>{}(k.video);
        h ^= std::hash<std::string>{}(k.quality) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<std::string>{}(k.identifier) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// Per-worker static cache — no mutex needed (nginx workers are single-threaded per worker)
static std::unordered_map<OrtSessionKey, std::unique_ptr<OrtSuperResSession>, OrtSessionKeyHash>
    g_ort_session_cache;

// Parse key from URL like /vrvideos_24/RedsVideo1/q1/1/segment0000.mp4
//                                       [video]  [q] [id]
static OrtSessionKey parse_ort_key(const std::string &uri) {
    OrtSessionKey key;
    // Split by '/'
    std::vector<std::string> parts;
    std::stringstream ss(uri);
    std::string token;
    while (std::getline(ss, token, '/')) {
        if (!token.empty()) parts.push_back(token);
    }
    // /vrvideos_24/RedsVideo1/q1/1/segment0000.mp4
    //  [0]         [1]        [2][3][4]
    if (parts.size() >= 4) {
        key.video      = parts[1];   // RedsVideo1
        key.quality    = parts[2];   // q1
        key.identifier = parts[3];   // 1
    }
    return key;
}

// Call this instead of constructing OrtSuperResSession directly
OrtSuperResSession &get_or_create_ort_session(const std::string &uri,
                                               const void *model_data,
                                               size_t model_data_len,
                                               ngx_http_request_t *r)
{
    OrtSessionKey key = parse_ort_key(uri);

    auto it = g_ort_session_cache.find(key);
    if (it != g_ort_session_cache.end()) {
        ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
            "SR_ORT cache hit: video=%s quality=%s id=%s  cache_size=%zu",
            key.video.c_str(), key.quality.c_str(), key.identifier.c_str(),
            g_ort_session_cache.size());
        return *it->second;
    }

    // Cache miss — create new session (pays the ~100ms CUDA warm-up once)
    ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
        "SR_ORT cache miss: video=%s quality=%s id=%s — creating session",
        key.video.c_str(), key.quality.c_str(), key.identifier.c_str());

    auto session = std::make_unique<OrtSuperResSession>(model_data, model_data_len);
    auto &ref = *session;
    g_ort_session_cache[key] = std::move(session);
    return ref;
}


// ---------------------------------------------------------------------------
// Dimension parser (unchanged)
// ---------------------------------------------------------------------------

ngx_dims_t
ngx_http_superresolution_file_cache_get_dimensions(ngx_str_t resolution, ngx_log_t *log)
{
  u_char *m, *n;
  m = n = resolution.data;
  ngx_dims_t dim;
  dim.width = -1;
  dim.height = -1;

  for (size_t i=0; i < resolution.len; i++){
    if(m != NULL && (*m == 'x' || *m == 'X')){
      dim.width = ngx_atoi(n, m-n);
      n = m+1;
    }
    ++m;
  }

  if (m > n){
    dim.height = ngx_atoi(n, m-n);
  }
  return dim;
}

Mat upscaleImage(Mat img, string modelName, string modelPath, int scale, ngx_http_request_t *r){
  DnnSuperResImpl sr;
  sr.readModel(modelPath);
  sr.setModel(modelName,scale);
  Mat outputImage;
  sr.upsample(img, outputImage);
  return outputImage;
}

// ---------------------------------------------------------------------------
// OrtSuperResSession constructor
// ---------------------------------------------------------------------------

OrtSuperResSession::OrtSuperResSession(const void *model_data, size_t model_data_len)
    : env(ORT_LOGGING_LEVEL_WARNING, "espcn"),
      session(nullptr)
{
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

    // CUDA execution provider
    OrtCUDAProviderOptions cuda_opts{};
    cuda_opts.device_id = 0;
    // Allow ORT to pre-allocate a CUDA memory arena for fast allocation
    cuda_opts.arena_extend_strategy = 0;  // kNextPowerOfTwo
    opts.AppendExecutionProvider_CUDA(cuda_opts);

    session = Ort::Session(env, model_data, model_data_len, opts);

    // Cache input/output names (ORT 1.16 API: AllocatedStringPtr)
    Ort::AllocatorWithDefaultOptions allocator;
    input_name  = std::string(session.GetInputNameAllocated(0, allocator).get());
    output_name = std::string(session.GetOutputNameAllocated(0, allocator).get());
}

// ---------------------------------------------------------------------------
// Internal helper: run one ORT forward pass on a pre-built float32 Y blob.
//
// blob layout: [N, 1, H, W]  (NCHW, float32, values in [0,1])
// Returns output blob [N, 1, H*scale, W*scale].
// ---------------------------------------------------------------------------
static cv::Mat ort_forward(OrtSuperResSession &ort,
                           const float *input_data, size_t N, size_t H, size_t W)
{
    //std::array<int64_t, 4> input_shape = {(int64_t)N, 1, (int64_t)H, (int64_t)W};
    std::array<int64_t, 4> input_shape = {(int64_t)N, (int64_t)H, (int64_t)W, 1};
    size_t numel = N * H * W;

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info,
        const_cast<float *>(input_data), numel,
        input_shape.data(), input_shape.size());

    const char *input_names[]  = { ort.input_name.c_str()  };
    const char *output_names[] = { ort.output_name.c_str() };

    auto output_tensors = ort.session.Run(
        Ort::RunOptions{nullptr},
        input_names,  &input_tensor, 1,
        output_names, 1);

    // Copy output into a cv::Mat so the ORT buffer can be freed
    auto &out = output_tensors[0];
    auto shape = out.GetTensorTypeAndShapeInfo().GetShape();
    // shape: [N, out_H, out_W, 1]
    int out_H = (int)shape[1];
    int out_W = (int)shape[2];

    cv::Mat result((int)(N * out_H), out_W, CV_32FC1);
    const float *src = out.GetTensorData<float>();
    std::memcpy(result.data, src, N * out_H * out_W * sizeof(float));
    return result;
}

// ---------------------------------------------------------------------------
// upscaleImageORT — single frame, ORT path
// ---------------------------------------------------------------------------

/*
cv::Mat upscaleImageORT(const cv::Mat &imageBGR, OrtSuperResSession &ort, float scale, ngx_http_request_t *r)
{
    int H = imageBGR.rows;
    int W = imageBGR.cols;

    // BGR -> YCrCb
    cv::Mat ycrcb;
    cv::cvtColor(imageBGR, ycrcb, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> channels(3);
    cv::split(ycrcb, channels);

    // Y channel: uint8 -> float32 [0,1], shape [H,W]
    cv::Mat Y_f32;
    channels[0].convertTo(Y_f32, CV_32FC1, 1.0 / 255.0);

    // Run inference: input [1,1,H,W]
    cv::Mat out_blob = ort_forward(ort, (float *)Y_f32.data, 1, H, W);
    // out_blob rows = out_H, cols = out_W
    int out_H = out_blob.rows;
    int out_W = out_blob.cols;

    // Clamp and convert to uint8
    cv::Mat Y_out_8u;
    cv::threshold(out_blob, out_blob, 1.0f, 1.0f, cv::THRESH_TRUNC);
    cv::threshold(out_blob, out_blob, 0.0f, 0.0f, cv::THRESH_TOZERO);
    out_blob.convertTo(Y_out_8u, CV_8UC1, 255.0);

    // Bicubic-upscale CrCb
    cv::Mat Cr_up, Cb_up;
    cv::resize(channels[1], Cr_up, cv::Size(out_W, out_H), 0, 0, cv::INTER_CUBIC);
    cv::resize(channels[2], Cb_up, cv::Size(out_W, out_H), 0, 0, cv::INTER_CUBIC);

    // Merge and convert back
    cv::Mat ycrcb_out;
    std::vector<cv::Mat> out_channels = {Y_out_8u, Cr_up, Cb_up};
    cv::merge(out_channels, ycrcb_out);

    cv::Mat bgr_out;
    cv::cvtColor(ycrcb_out, bgr_out, cv::COLOR_YCrCb2BGR);
    return bgr_out;
}
*/

cv::Mat upscaleImageORT(const cv::Mat &imageBGR, OrtSuperResSession &ort, float scale, ngx_http_request_t *r)
{
    using clk = std::chrono::high_resolution_clock;
    auto t0 = clk::now();

    int H = imageBGR.rows;
    int W = imageBGR.cols;

    // BGR -> YCrCb
    cv::Mat ycrcb;
    cv::cvtColor(imageBGR, ycrcb, cv::COLOR_BGR2YCrCb);
    auto t1 = clk::now();

    std::vector<cv::Mat> channels(3);
    cv::split(ycrcb, channels);
    auto t2 = clk::now();

    // Y channel: uint8 -> float32 [0,1]
    cv::Mat Y_f32;
    channels[0].convertTo(Y_f32, CV_32FC1, 1.0 / 255.0);
    auto t3 = clk::now();

    // ORT inference
    cv::Mat out_blob = ort_forward(ort, (float *)Y_f32.data, 1, H, W);
    auto t4 = clk::now();

    int out_H = out_blob.rows;
    int out_W = out_blob.cols;

    // Clamp and convert to uint8
    cv::Mat Y_out_8u;
    cv::threshold(out_blob, out_blob, 1.0f, 1.0f, cv::THRESH_TRUNC);
    cv::threshold(out_blob, out_blob, 0.0f, 0.0f, cv::THRESH_TOZERO);
    out_blob.convertTo(Y_out_8u, CV_8UC1, 255.0);
    auto t5 = clk::now();

    // Bicubic-upscale CrCb
    cv::Mat Cr_up, Cb_up;
    cv::resize(channels[1], Cr_up, cv::Size(out_W, out_H), 0, 0, cv::INTER_CUBIC);
    cv::resize(channels[2], Cb_up, cv::Size(out_W, out_H), 0, 0, cv::INTER_CUBIC);
    auto t6 = clk::now();

    // Merge and convert back
    cv::Mat ycrcb_out;
    std::vector<cv::Mat> out_channels = {Y_out_8u, Cr_up, Cb_up};
    cv::merge(out_channels, ycrcb_out);
    cv::Mat bgr_out;
    cv::cvtColor(ycrcb_out, bgr_out, cv::COLOR_YCrCb2BGR);
    auto t7 = clk::now();

    auto ms = [](auto a, auto b){
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, 0,
        "[SR timing] cvtColor=%.2fms  split=%.2fms  convertTo=%.2fms  "
        "ORT=%.2fms  clamp+8u=%.2fms  crcb_resize=%.2fms  merge+cvt=%.2fms  "
        "TOTAL=%.2fms\n",
        ms(t0,t1), ms(t1,t2), ms(t2,t3),
        ms(t3,t4), ms(t4,t5), ms(t5,t6), ms(t6,t7),
        ms(t0,t7));

    return bgr_out;
}


SRResult upscaleImageORT_YUV(const cv::Mat &Y_u8, const cv::Mat &Cb_u8, const cv::Mat &Cr_u8,
                              OrtSuperResSession &ort, float scale, ngx_http_request_t *r)
{
    cv::Mat Y_f32;
    Y_u8.convertTo(Y_f32, CV_32FC1, 1.0 / 255.0);

    cv::Mat out_blob = ort_forward(ort, (float *)Y_f32.data, 1, Y_u8.rows, Y_u8.cols);

    cv::threshold(out_blob, out_blob, 1.0f, 1.0f, cv::THRESH_TRUNC);
    cv::threshold(out_blob, out_blob, 0.0f, 0.0f, cv::THRESH_TOZERO);

    SRResult result;
    out_blob.convertTo(result.Y, CV_8UC1, 255.0);

    // Use ACTUAL ORT output dimensions — not H*scale which may differ by 1 pixel
    int actual_out_H = result.Y.rows;
    int actual_out_W = result.Y.cols;

    cv::resize(Cb_u8, result.Cb, cv::Size(actual_out_W/2, actual_out_H/2), 0, 0, cv::INTER_CUBIC);
    cv::resize(Cr_u8, result.Cr, cv::Size(actual_out_W/2, actual_out_H/2), 0, 0, cv::INTER_CUBIC);

    return result;
}

// ---------------------------------------------------------------------------
// upscaleBatchORT — N frames in one GPU forward pass
// ---------------------------------------------------------------------------

std::vector<cv::Mat> upscaleBatchORT(const std::vector<cv::Mat> &images,
                                     OrtSuperResSession &ort,
                                     float scale,
                                     ngx_http_request_t *r)
{
    if (images.empty()) return {};

    int H = images[0].rows;
    int W = images[0].cols;
    int N = (int)images.size();
    int out_H = (int)(H * scale);
    int out_W = (int)(W * scale);

    // Build one contiguous [N, 1, H, W] float32 buffer on the CPU
    std::vector<float> input_buf((size_t)N * H * W);

    std::vector<cv::Mat> crcb_upscaled(N);

    for (int i = 0; i < N; i++) {
        cv::Mat ycrcb;
        cv::cvtColor(images[i], ycrcb, cv::COLOR_BGR2YCrCb);

        std::vector<cv::Mat> channels(3);
        cv::split(ycrcb, channels);

        // Normalize Y into the flat buffer (row-major NCHW)
        cv::Mat Y_f32;
        channels[0].convertTo(Y_f32, CV_32FC1, 1.0 / 255.0);
        std::memcpy(input_buf.data() + (size_t)i * H * W,
                    Y_f32.data,
                    (size_t)H * W * sizeof(float));

        // Bicubic CrCb upscale (CPU, cheap)
        cv::Mat crcb_pair;
        std::vector<cv::Mat> crcb_ch = {channels[1], channels[2]};
        cv::merge(crcb_ch, crcb_pair);
        cv::resize(crcb_pair, crcb_upscaled[i], cv::Size(out_W, out_H), 0, 0, cv::INTER_CUBIC);
    }

    // One batched GPU forward pass [N, 1, H, W] -> [N, 1, out_H, out_W]
    cv::Mat out_blob = ort_forward(ort, input_buf.data(), N, H, W);
    // out_blob is laid out as N rows of out_H each, width out_W

    std::vector<cv::Mat> results(N);
    for (int i = 0; i < N; i++) {
        // Row slice for frame i: rows [i*out_H .. (i+1)*out_H)
        cv::Mat Y_out_f32 = out_blob.rowRange(i * out_H, (i + 1) * out_H);

        cv::threshold(Y_out_f32, Y_out_f32, 1.0f, 1.0f, cv::THRESH_TRUNC);
        cv::threshold(Y_out_f32, Y_out_f32, 0.0f, 0.0f, cv::THRESH_TOZERO);

        cv::Mat Y_out_8u;
        Y_out_f32.convertTo(Y_out_8u, CV_8UC1, 255.0);

        std::vector<cv::Mat> crcb_channels(2);
        cv::split(crcb_upscaled[i], crcb_channels);

        cv::Mat ycrcb_out;
        std::vector<cv::Mat> out_ch = {Y_out_8u, crcb_channels[0], crcb_channels[1]};
        cv::merge(out_ch, ycrcb_out);

        cv::cvtColor(ycrcb_out, results[i], cv::COLOR_YCrCb2BGR);
    }

    return results;
}

// ---------------------------------------------------------------------------
// Legacy OpenCV-DNN wrappers — these now just call the ORT path.
// They require the onnx bytes to be accessible; since the legacy callers
// pass a cv::dnn::Net we cannot re-create the session here without the raw
// bytes.  These stubs are kept only so existing call sites that still use
// the old cv::dnn::Net path compile.  The video pipeline should be updated
// to call upscaleImageORT / upscaleBatchORT directly.
// ---------------------------------------------------------------------------

Mat upscaleImageFromONNX(Mat imageBGR, Net net, int scale, ngx_http_request_t *r)
{
    // Original OpenCV-DNN implementation kept as fallback
    cv::Mat preproc_img;
    preprocess_YCrCb(imageBGR, preproc_img);

    Mat ycbcr_channels[3];
    split(preproc_img, ycbcr_channels);
    Mat Y = ycbcr_channels[0];

    cv::Mat blob;
    dnn::blobFromImage(Y, blob, 1.0);
    net.setInput(blob);
    cv::Mat blob_output = net.forward();

    std::vector<Mat> model_outs;
    dnn::imagesFromBlob(blob_output, model_outs);
    Mat out_img = model_outs[0];

    cv::Mat result;
    reconstruct_YCrCb(out_img, preproc_img, result, scale);
    return result;
}

std::vector<cv::Mat> upscaleBatchFromONNX(const std::vector<cv::Mat> &images,
                                           cv::dnn::Net &net,
                                           float scale,
                                           ngx_http_request_t *r)
{
    // Original batch implementation kept as fallback
    if (images.empty()) return {};

    int h = images[0].rows;
    int w = images[0].cols;
    int out_h = (int)(h * scale);
    int out_w = (int)(w * scale);

    std::vector<cv::Mat> y_channels(images.size());
    std::vector<cv::Mat> upscaled_crcb(images.size());

    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat ycrcb;
        cv::cvtColor(images[i], ycrcb, cv::COLOR_BGR2YCrCb);
        std::vector<cv::Mat> channels(3);
        cv::split(ycrcb, channels);
        channels[0].convertTo(y_channels[i], CV_32FC1, 1.0 / 255.0);
        cv::Mat crcb;
        std::vector<cv::Mat> crcb_channels = {channels[1], channels[2]};
        cv::merge(crcb_channels, crcb);
        cv::resize(crcb, upscaled_crcb[i], cv::Size(out_w, out_h), 0, 0, cv::INTER_CUBIC);
    }

    cv::Mat blob = cv::dnn::blobFromImages(y_channels, 1.0, cv::Size(), cv::Scalar(), false, false, CV_32F);
    net.setInput(blob);
    cv::Mat output = net.forward();

    std::vector<cv::Mat> results(images.size());
    for (size_t i = 0; i < images.size(); i++) {
        cv::Mat y_out(out_h, out_w, CV_32FC1, output.ptr<float>(i, 0));
        cv::Mat y_out_8u;
        y_out.convertTo(y_out_8u, CV_8UC1, 255.0);
        std::vector<cv::Mat> crcb_channels(2);
        cv::split(upscaled_crcb[i], crcb_channels);
        std::vector<cv::Mat> merged_channels = {y_out_8u, crcb_channels[0], crcb_channels[1]};
        cv::Mat ycrcb_out;
        cv::merge(merged_channels, ycrcb_out);
        cv::cvtColor(ycrcb_out, results[i], cv::COLOR_YCrCb2BGR);
    }
    return results;
}

// ---------------------------------------------------------------------------
// preload_all_ort_sessions
//
// Called once per nginx worker process at startup.
// Walks /usr/local/data/models/ and pre-loads every espcn.onnx it finds.
//
// Expected directory layout:
//   /usr/local/data/models/{video}/{ignored}/{quality}/{identifier}/espcn.onnx
//
// This maps to OrtSessionKey { video, quality, identifier }, which is the
// same decomposition used by parse_ort_key() when serving requests from URLs
// of the form: /vrvideos_24/{video}/{quality}/{identifier}/segment….mp4
// ---------------------------------------------------------------------------

extern "C" void preload_all_ort_sessions(ngx_log_t *log)
{
    namespace fs = std::filesystem;
    const fs::path models_root = "/usr/local/data/models";

    if (!fs::exists(models_root) || !fs::is_directory(models_root)) {
        ngx_log_error(NGX_LOG_WARN, log, 0,
            "SR_PRELOAD models root does not exist: %s", models_root.c_str());
        return;
    }

    int loaded = 0, skipped = 0, failed = 0;

    // Recursively find every espcn.onnx
    for (auto &entry : fs::recursive_directory_iterator(
             models_root,
             fs::directory_options::skip_permission_denied))
    {
        if (!entry.is_regular_file()) continue;
        if (entry.path().filename() != "espcn.onnx") continue;

        // Compute path components relative to models_root:
        // Expected: [video] [ignored] [quality] [identifier] espcn.onnx
        fs::path rel = fs::relative(entry.path(), models_root);
        std::vector<std::string> parts;
        for (auto &comp : rel) {
            std::string s = comp.string();
            if (!s.empty() && s != ".") parts.push_back(s);
        }

        // Minimum: video / quality / identifier / espcn.onnx  (4 parts)
        // With the extra middle dir:  video / ignored / quality / identifier / espcn.onnx (5 parts)
        OrtSessionKey key;
        if (parts.size() >= 5) {
            // /models/{video}/{ignored}/{quality}/{identifier}/espcn.onnx
            key.video      = parts[0];
            key.quality    = parts[2];
            key.identifier = parts[3];
        } else if (parts.size() == 4) {
            // /models/{video}/{quality}/{identifier}/espcn.onnx
            key.video      = parts[0];
            key.quality    = parts[1];
            key.identifier = parts[2];
        } else {
            ngx_log_error(NGX_LOG_WARN, log, 0,
                "SR_PRELOAD unexpected path depth, skipping: %s",
                entry.path().c_str());
            skipped++;
            continue;
        }

        // Skip if already cached (e.g. preload called twice)
        if (g_ort_session_cache.count(key)) {
            skipped++;
            continue;
        }

        // Read the .onnx file into memory
        std::ifstream f(entry.path(), std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            ngx_log_error(NGX_LOG_WARN, log, 0,
                "SR_PRELOAD cannot open model file: %s", entry.path().c_str());
            failed++;
            continue;
        }
        std::streamsize sz = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<char> buf(sz);
        if (!f.read(buf.data(), sz)) {
            ngx_log_error(NGX_LOG_WARN, log, 0,
                "SR_PRELOAD read failed for: %s", entry.path().c_str());
            failed++;
            continue;
        }

        // Create the ORT session and insert into cache
        try {
            auto session = std::make_unique<OrtSuperResSession>(
                static_cast<const void *>(buf.data()),
                static_cast<size_t>(sz));

            ngx_log_error(NGX_LOG_NOTICE, log, 0,
                "SR_PRELOAD loaded: video=%s quality=%s id=%s  path=%s",
                key.video.c_str(), key.quality.c_str(),
                key.identifier.c_str(), entry.path().c_str());

            g_ort_session_cache[key] = std::move(session);
            loaded++;
        } catch (const Ort::Exception &e) {
            ngx_log_error(NGX_LOG_ERR, log, 0,
                "SR_PRELOAD ORT session creation failed for %s: %s",
                entry.path().c_str(), e.what());
            failed++;
        } catch (const std::exception &e) {
            ngx_log_error(NGX_LOG_ERR, log, 0,
                "SR_PRELOAD exception for %s: %s",
                entry.path().c_str(), e.what());
            failed++;
        }
    }

    ngx_log_error(NGX_LOG_NOTICE, log, 0,
        "SR_PRELOAD complete: loaded=%d  skipped=%d  failed=%d  cache_size=%zu",
        loaded, skipped, failed, g_ort_session_cache.size());
}
