#ifndef _NVIDIA_GPU_MONITOR_H
#define _NVIDIA_GPU_MONITOR_H

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

#include <chrono>
#include <iostream>
#include <thread>

#include "nvml.h"
#include "utils.h"

#endif // _NVIDIA_GPU_MONITOR_H
