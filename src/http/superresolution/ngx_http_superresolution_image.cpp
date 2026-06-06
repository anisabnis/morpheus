#include "ngx_http_superresolution_util.hpp"

extern "C" ngx_int_t ngx_http_superresolution_scale_image(ngx_http_request_t *r, ngx_chain_t *in, char* buf, int buflen);


// Call this function if you know that the content is image
ngx_int_t
ngx_http_superresolution_scale_image(ngx_http_request_t *r, ngx_chain_t *in, char* buf, int buflen)
{
  Mat matImg;
  matImg = imdecode(Mat(1, buflen, CV_8UC1, buf), IMREAD_UNCHANGED);

  ngx_dims_t dim = ngx_http_superresolution_file_cache_get_dimensions(r->headers_in.resolution, r->connection->log);
  float scale_factor = ((float) dim.width)/matImg.size().width;

  Mat result;
  if (fabs(scale_factor - 1) < 0.01){
    return NGX_OK;
  }
  else if (fabs(scale_factor - 2) < 0.01){
    string path = "/usr/local/nginx/models/FSRCNN_x2.pb";
    string modelName = "fsrcnn";
    int scale = 2;
    result = upscaleImage(matImg, modelName, path, scale, r);    
  }
  else if(fabs(scale_factor - 4) < 0.01){
    string path = "/usr/local/nginx/models/FSRCNN_x4.pb";
    string modelName = "fsrcnn";
    int scale = 4;
    result = upscaleImage(matImg, modelName, path, scale, r);    
  }
  else if(fabs(scale_factor - 0.5) < 0.01){
    ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "Downscale by a factor of 2");
    resize(matImg, result, cv::Size(), 0.5, 0.5);
  }
  else if(fabs(scale_factor - 0.25) < 0.01){
    ngx_log_debug(NGX_LOG_DEBUG_HTTP, r->connection->log, 0, "Downscale by a factor of 4");
    resize(matImg, result, cv::Size(), 0.25, 0.25);
  }
  
  // Move to the last chain
  ngx_chain_t *cl;
  for (cl = in; cl; cl = cl->next) {
    if (cl->buf->last_buf) {
      break;
    }
  }

  if (!cl->buf->in_file){
    return NGX_DECLINED;
  }
  
  std::vector<uchar> buffer;
  cv::imencode(".png", result, buffer);
  u_char *newbuf = (u_char *) malloc(buffer.size());
  std::copy(buffer.begin(), buffer.end(), newbuf);

  ngx_buf_t *tmpbuf;
  tmpbuf = (ngx_buf_t *) ngx_pcalloc(r->pool, sizeof(ngx_buf_t));
  if (tmpbuf == NULL) {
        return NGX_DECLINED;
  }

  tmpbuf->pos = newbuf;
  tmpbuf->last = tmpbuf->pos + buffer.size();
  tmpbuf->start = tmpbuf->pos;
  tmpbuf->end = tmpbuf->last;
  tmpbuf->last_buf = 1;
  tmpbuf->memory = 1;
  cl->buf = tmpbuf;     
  
  return NGX_OK;  
}

