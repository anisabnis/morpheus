/*
 * Copyright (c) 2001 Fabrice Bellard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
 
/**
 * @file
 * video decoding with libavcodec API example
 *
 * @example decode_video.c
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavformat/avformat.h>
}

#include "precomp.hpp"
#include <opencv2/dnn_superres.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <chrono>

using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;
using namespace std;
using namespace cv;
using namespace dnn;
using namespace dnn_superres;

#define INBUF_SIZE 4096


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


static void encode(AVCodecContext *enc_ctx, AVFrame *frame, AVPacket *pkt,
                   FILE *outfile)
{
    int ret;
 
    ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0) {
        fprintf(stderr, "Error sending a frame for encoding\n");
        exit(1);
    }
 
    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "Error during encoding\n");
            exit(1);
        }
 
        fwrite(pkt->data, 1, pkt->size, outfile);
        av_packet_unref(pkt);
    }
}


static void pgm_save(unsigned char *buf, int wrap, int xsize, int ysize,
                     char *filename)
{
    FILE *f;
    int i;
 
    f = fopen(filename,"wb");
    fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
    for (i = 0; i < ysize; i++)
        fwrite(buf + i * wrap, 1, xsize, f);
    fclose(f);
}

void reconstruct_Y(InputArray inpImg, InputArray origImg, OutputArray outImg, int scale){
  inpImg.getMat().convertTo(outImg, CV_8U, 255.0);
}

Mat upscaleImage(Mat imageBGR, string modelName, string modelPath, int scale){

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;
  
  DnnSuperResImpl sr;
  //sr.readModel(modelPath);
  sr.setModel(modelName, scale);

  cv::dnn::Net net = cv::dnn::readNetFromTensorflow(modelPath);
  net.setPreferableTarget(DNN_TARGET_CUDA);
  net.setPreferableBackend(DNN_BACKEND_CUDA);
  
  cv::Mat preproc_img;
  preprocess_YCrCb(imageBGR, preproc_img);
  
  Mat ycbcr_channels[3];
  split(preproc_img, ycbcr_channels);

  Mat Y = ycbcr_channels[0];
  
  cv::Mat blob;
  dnn::blobFromImage(Y, blob, 1.0);

  auto t1 = high_resolution_clock::now();
  net.setInput(blob);
  cv::Mat blob_output = net.forward();

  auto t2 = high_resolution_clock::now();
  auto ms_int = duration_cast<milliseconds>(t2 - t1);
  printf("\n\n%d\n\n", ms_int);

  std::vector <Mat> model_outs;
  dnn::imagesFromBlob(blob_output, model_outs);
  
  Mat out_img = model_outs[0];
  //Mat result;
  //reconstruct_Y(out_img, preproc_img, result, scale);
  
  //Reconstruct: upscale the Cr and Cb space and merge the three layer
  cv::Mat result;
  reconstruct_YCrCb(out_img, preproc_img, result, scale);

  return result;
  //return result;
}


static int decode_packet(AVPacket *pPacket, AVCodecContext *pCodecContext, AVFrame *pFrame,
			 AVFrame *enc_frame, FILE *outfile, AVCodecContext *enc_ctx, AVPacket *enc_pkt)
{
  // Supply raw packet data as input to a decoder
  // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga58bc4bf1e0ac59e27362597e467efff3
  int response = avcodec_send_packet(pCodecContext, pPacket);
  int ret;
  
  if (response < 0) {
    //printf("Error while sending a packet to the decoder: %s", av_err2str(response));
    return response;
  }

  while (response >= 0)
  {
    // Return decoded output data (into a frame) from a decoder
    // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga11e6542c4e66d3028668788a1a74217c
    response = avcodec_receive_frame(pCodecContext, pFrame);
    if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
      break;
    } else if (response < 0) {
      return response;
    }

    int width = pFrame->width;
    int height = pFrame->height;
    
    cv::Mat image(height, width, CV_8UC3);
    int cvLinesizes[1];
    cvLinesizes[0] = image.step1();            
      
    SwsContext *conversion = sws_getContext(width, height, (AVPixelFormat)pFrame->format, width, height,
					    AVPixelFormat::AV_PIX_FMT_BGR24, SWS_FAST_BILINEAR, NULL, NULL, NULL);

    sws_scale(conversion, pFrame->data, pFrame->linesize, 0, height, &image.data, cvLinesizes);
    sws_freeContext(conversion);

    Mat result;
    string path = "./simple_frozen_graph.pb";
    string modelName = "custom";
    float scale = 4;
    result = upscaleImage(image, modelName, path, scale);

    cvLinesizes[0] = result.step1();            
    
    // Convert to ffmpeg frame
    AVFrame *pFrameBGR = av_frame_alloc();      
    pFrameBGR->format = AVPixelFormat::AV_PIX_FMT_YUV420P;

    pFrameBGR->width  = result.size().width;
    width = result.size().width;
    pFrameBGR->height = result.size().height;
    height = result.size().height;
    ret = av_frame_get_buffer(pFrameBGR, 0);

    if (ret < 0) {
      fprintf(stderr, "Could not allocate the video frame data\n");
      exit(1);
    }
        
    SwsContext *conversion1 = sws_getContext(width, height, AVPixelFormat::AV_PIX_FMT_BGR24, width, height,
					     AVPixelFormat::AV_PIX_FMT_YUV420P, SWS_FAST_BILINEAR, NULL, NULL, NULL);

    sws_scale(conversion1, &result.data, cvLinesizes, 0, height, pFrameBGR->data, pFrameBGR->linesize);
    sws_freeContext(conversion1);      

    encode(enc_ctx, pFrameBGR, enc_pkt, outfile);      
    av_frame_free(&pFrameBGR);

  }
  return 0;
}

 
int main(int argc, char **argv)
{
    const char *filename, *outfilename;
    const AVCodec *codec;
    const AVCodec *codec1;
    AVCodecContext *c= NULL;
    AVCodecContext *c1= NULL;
    FILE *f;
    AVFrame *frame;
    AVFrame *enc_frame;
    uint8_t *data;
    size_t   data_size;
    int ret;
    int eof;
    AVPacket *pkt;
    AVPacket *enc_pkt;
    FILE *out_file;
    
    if (argc <= 2) {
        fprintf(stderr, "Usage: %s <input file> <output file>\n"
                "And check your input file is encoded by mpeg1video please.\n", argv[0]);
        exit(0);
    }
    filename    = argv[1];
    outfilename = argv[2];

    printf("args %s %s\n", filename, outfilename);
    
    out_file = fopen(outfilename, "wb");
    if (!out_file) {
        fprintf(stderr, "sab Could not open %s\n", outfilename);
        exit(1);
    }    
    
    pkt = av_packet_alloc();
    if (!pkt)
        exit(1);

    enc_pkt = av_packet_alloc();
    if (!enc_pkt)
        exit(1);

    /* set end of buffer to 0 (this ensures that no overreading happens for damaged MPEG streams) */
    AVFormatContext *pFormatContext = avformat_alloc_context();
    if (!pFormatContext) {
      printf("ERROR could not allocate memory for Format Context");
      return -1;
    }

    if (avformat_open_input(&pFormatContext, filename, NULL, NULL) != 0) {
      printf("ERROR could not open the file");
      return -1;
    }

    if (avformat_find_stream_info(pFormatContext,  NULL) < 0) {
      printf("ERROR could not get the stream info");
      return -1;
    }

    AVCodecParameters *pCodecParameters =  NULL;
    int video_stream_index = -1;

    for (int i = 0; i < pFormatContext->nb_streams; i++){
      AVCodecParameters *pLocalCodecParameters =  NULL;
      pLocalCodecParameters = pFormatContext->streams[i]->codecpar;

      const AVCodec *pLocalCodec = NULL;

      // finds the registered decoder for a codec ID
      // https://ffmpeg.org/doxygen/trunk/group__lavc__decoding.html#ga19a0ca553277f019dd5b0fec6e1f9dca
      pLocalCodec = avcodec_find_decoder(pLocalCodecParameters->codec_id);
      
      if (pLocalCodec == NULL) {
	printf("ERROR unsupported codec!");
	// In this example if the codec is not found we just skip it
	continue;
      }

      // when the stream is a video we store its index, codec parameters and codec
      if (pLocalCodecParameters->codec_type == AVMEDIA_TYPE_VIDEO) {
	if (video_stream_index == -1) {
	  video_stream_index = i;
	  codec = (AVCodec *) pLocalCodec;
	  pCodecParameters = pLocalCodecParameters;
	}
      }
    }

    if (video_stream_index == -1) {
      printf("File %s does not contain a video stream!", argv[1]);
      return -1;
    }

    c = avcodec_alloc_context3(codec);
    if (!c){
      printf("failed to allocated memory for AVCodecContext");
      return -1;
    }

    if (avcodec_parameters_to_context(c, pCodecParameters) < 0){
      printf("failed to copy codec params to codec context");
      return -1;
    }

    // Initialize the AVCodecContext to use the given AVCodec.
    // https://ffmpeg.org/doxygen/trunk/group__lavc__core.html#ga11f785a188d7d9df71621001465b0f1d
    if (avcodec_open2(c, codec, NULL) < 0)
    {
      printf("failed to open codec through avcodec_open2");
      return -1;
    }

    // Now set up the encoder
    codec1 = avcodec_find_encoder(pCodecParameters->codec_id);
    if (!codec1) {
      printf("could not find encoder ");
      return -1;
    }  
 
    c1 = avcodec_alloc_context3(codec1);
    if (!c1) {
      return -1;
    }

    if (avcodec_parameters_to_context(c1, pCodecParameters)) {
      printf("could not find encoder context");
      return ret;
    }
    
     c1 = avcodec_alloc_context3(codec1);
     if (!c1) {
         fprintf(stderr, "Could not allocate video codec context\n");
         exit(1);
     }
  
     /* put sample parameters */
     /* resolution must be a multiple of two */
     /* frames per second */
     c1->time_base = (AVRational){1, 30};
     c1->framerate = (AVRational){30, 1};
     c1->width = (int) 256 * 2;
     c1->height = (int) 144 * 2;
     /* emit one intra frame every ten frames
      * check frame pict_type before passing frame
      * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
      * then gop_size is ignored and the output of encoder
      * will always be I frame irrespective to gop_size
      */
     c1->gop_size = 10;
     c1->max_b_frames = 1;
     c1->pix_fmt = AV_PIX_FMT_YUV420P;
     //c1->pix_fmt = AVPixelFormat::AV_PIX_FMT_GRAY8;
     //c->pix_fmt = c->pix_fmt;
     
     if (codec1->id == AV_CODEC_ID_H264)
         av_opt_set(c1->priv_data, "preset", "slow", 0);
  
     /* open it */
     ret = avcodec_open2(c1, codec1, NULL);
     if (ret < 0) {
       //fprintf(stderr, "Could not open codec: %s\n", av_err2str(ret));
         exit(1);
     }
     
    // AVFormatContext *fmt_ctx = NULL;
    // if(avformat_open_input(&fmt_ctx, filename, NULL, NULL) != 0) {
    //   fprintf(stderr, "fmt_ctx error\n");
    // }
    
    // if (avformat_find_stream_info(fmt_ctx, NULL) < 0) {
    //   return -1;
    // }

    // fprintf(stderr, "AM I here 2 \n");
    
    // int video_stream = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    // if (video_stream < 0) {
    //   return -1;
    // }

    // fprintf(stderr, "AM I here 3 \n");
    
    // AVCodecParameters *origin_par = NULL;
    // fprintf(stderr, "AM I here 4 %d \n", video_stream);
    // origin_par = fmt_ctx->streams[video_stream]->codecpar;
    // fprintf(stderr, "AM I here 5 %d \n", video_stream);
    
    // /* find the MPEG-2 video decoder */
    // //codec = avcodec_find_decoder(AV_CODEC_ID_MPEG2VIDEO); //done
    // printf("(origin_par->codec_id %d %d" , origin_par->codec_id, AV_CODEC_ID_H264 );

    // fprintf(stderr, "AM I here 6 %d \n", video_stream);
    // codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    // //codec = avcodec_find_decoder(AV_CODEC_ID_MPEG4); //done
    // if (!codec) {
    //     fprintf(stderr, "Codec not found\n");
    //     exit(1);
    // }

    // parser = av_parser_init(codec->id);
    // if (!parser) {
    //     fprintf(stderr, "parser not found\n");
    //     exit(1);
    // }
    
    // c = avcodec_alloc_context3(codec); // done
    // if (!c) {
    //     fprintf(stderr, "Could not allocate video codec context\n");
    //     exit(1);
    // }
 
    // /* open it */
    // if (avcodec_open2(c, codec, NULL) < 0) { //done
    //     fprintf(stderr, "Could not open codec\n");
    //     exit(1);
    // }

    // if (codec->id == AV_CODEC_ID_H264)
    //   av_opt_set(c->priv_data, "preset", "slow", 0);

    
    // printf("codec width %d and height %d", c->width, c->height);
    
    /* find the mpeg2video encoder */
    // Setup the encoder
    //codec1 = avcodec_find_encoder(AV_CODEC_ID_MPEG2VIDEO);
    //codec = avcodec_find_encoder(origin_par->codec_id);
    // codec1 = avcodec_find_encoder(AV_CODEC_ID_H264);
    //  if (!codec1) {
    //      fprintf(stderr, "Codec1 not found\n");
    //      exit(1);
    //  }
  
    //  c1 = avcodec_alloc_context3(codec1);
    //  if (!c1) {
    //      fprintf(stderr, "Could not allocate video codec context\n");
    //      exit(1);
    //  }
  
    //  /* put sample parameters */
    //  //c1->bit_rate = 400000;
    //  /* resolution must be a multiple of two */
    //  /* frames per second */
    //  c1->time_base = (AVRational){1, 30};
    //  c1->framerate = (AVRational){30, 1};
    //  c1->width = (int) 256 * 2;
    //  c1->height = (int) 144 * 2;
    //  /* emit one intra frame every ten frames
    //   * check frame pict_type before passing frame
    //   * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
    //   * then gop_size is ignored and the output of encoder
    //   * will always be I frame irrespective to gop_size
    //   */
    //  c1->gop_size = 10;
    //  c1->max_b_frames = 1;
    //  c1->pix_fmt = AV_PIX_FMT_YUV420P;
    //  //c1->pix_fmt = AVPixelFormat::AV_PIX_FMT_GRAY8;
    //  //c->pix_fmt = c->pix_fmt;
     
    //  if (codec1->id == AV_CODEC_ID_H264)
    //      av_opt_set(c1->priv_data, "preset", "slow", 0);
  
    //  /* open it */
    //  ret = avcodec_open2(c1, codec1, NULL);
    //  if (ret < 0) {
    //    //fprintf(stderr, "Could not open codec: %s\n", av_err2str(ret));
    //      exit(1);
    //  }

    f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "ani Could not open %s\n", filename);
        exit(1);
    }

    frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate video frame\n");
        exit(1);
    }

    enc_frame = av_frame_alloc();
    if (!enc_frame) {
      fprintf(stderr, "Could not allocate video encode frame\n");
      exit(1);
    }

    enc_pkt = av_packet_alloc();
    if (!enc_pkt)
        exit(1);

    
    int response = 0;
    while (av_read_frame(pFormatContext, pkt) >= 0)
    {
      if (pkt->stream_index == video_stream_index) {
	response = decode_packet(pkt, c, frame, enc_frame, out_file, c1, enc_pkt);
	if (response < 0)
	  break;
      }
      av_packet_unref(pkt);
    }    
    // int ret1;
  
    // do {
    //     /* read raw data from the input file */
    //     data_size = fread(inbuf, 1, INBUF_SIZE, f);
    // 	//printf("data size : %d \n", data_size);
    //     if (ferror(f))
    //         break;
    //     eof = !data_size;

    //     /* use the parser to split the data into frames */
    //     data = inbuf;
    //     while (data_size > 0 || eof) {
    //         ret = av_parser_parse2(parser, c, &pkt->data, &pkt->size,
    //                                data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);

    //         if (ret < 0) {
    //             fprintf(stderr, "Error while parsing\n");
    //             exit(1);
    //         }
    //         data      += ret;
    //         data_size -= ret;

    //         if (pkt->size){
    // 	      ret1 = decode(c, frame, pkt, enc_frame, out_file, c1, enc_pkt);
    // 	      if (ret1 == 0){
    // 		continue;
    // 	      }
    // 	    }
    //         else if (eof)
    //             break;

    //     }
    // } while (!eof);
 
    // /* flush the decoder */
    // decode(c, frame, NULL, enc_frame, out_file, c1, enc_pkt);
    // fclose(f);
 
    //av_parser_close(parser);
    avcodec_free_context(&c);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    av_packet_free(&enc_pkt);
    return 0;
}
