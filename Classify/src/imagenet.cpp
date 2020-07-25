/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "videoSource.h"
#include "videoOutput.h"
#include <tensorNet.h>
#include "cudaFont.h"
#include "imageNet.h"
//
//#include <signal.h>
//
//
//#ifdef HEADLESS
//#define IS_HEADLESS() "headless"	// run without display
//#else
//#define IS_HEADLESS() (const char*)NULL
//#endif
//
//
//bool signal_recieved = false;
//
//void sig_handler(int signo) {
//    if (signo == SIGINT) {
//        LogVerbose("received SIGINT\n");
//        signal_recieved = true;
//    }
//}
//
//int usage() {
//    printf("usage: imagenet [--help] [--network=NETWORK] ...\n");
//    printf("                input_URI [output_URI]\n\n");
//    printf("Classify a video/image stream using an image recognition DNN.\n");
//    printf("See below for additional arguments that may not be shown above.\n\n");
//    printf("positional arguments:\n");
//    printf("    input_URI       resource URI of input stream  (see videoSource below)\n");
//    printf("    output_URI      resource URI of output stream (see videoOutput below)\n\n");
//
//    printf("%s", imageNet::Usage());
//    printf("%s", videoSource::Usage());
//    printf("%s", videoOutput::Usage());
//    printf("%s", Log::Usage());
//
//    return 0;
//}

#include <opencv2/opencv.hpp>
#include <glog/logging.h>

int main(int argc, char **argv) {
//    /*
//     * create font for image overlay
//     */
//    cudaFont *font = cudaFont::Create();
//
//    if (!font) {
//        LogError("imagenet:  failed to load font for overlay\n");
//        return 0;
//    }


    /*
     * create recognition network
     */
//	imageNet* net = imageNet::Create(cmdLine);
    imageNet::NetworkType type = imageNet::NetworkType::GOOGLENET;
    precisionType ptype = precisionType::TYPE_FP16;
    std::string prototxt_path = "/home/xzh/jetson-inference/data/networks/googlenet.prototxt";
    std::string model_path = "/home/xzh/jetson-inference/data/networks/bvlc_googlenet.caffemodel";
    std::string class_path = "/home/xzh/jetson-inference/data/networks/ilsvrc12_synset_words.txt";
    std::string inputName = "data";
    std::string outputName = "prob";
    size_t maxBatchSize = 1;
    deviceType dtype = DEVICE_GPU;
    imageNet *net = imageNet::Create(prototxt_path.data(), model_path.data(), NULL, class_path.data(),
                                     inputName.data(), outputName.data(), maxBatchSize, ptype, dtype, true);

    if (!net) {
        LogError("imagenet:  failed to initialize imageNet\n");
        return 0;
    }

    cv::Mat image = cv::imread("/home/xzh/Downloads/34b226c558bea6bba4dcbfb891943059.jpg");
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
//    LOG(INFO)<<"image type " <<image.type()<<"\n";
//    cv::imshow("aa", image);
//
//    cv::waitKey(-1);
//    cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
//    cv::imshow("bb", image);
//
//    cv::waitKey(-1);
    LOG(INFO) << "start classify";
    float confidence = 0.0f;
//    const int img_class = net->Classify(image, input->GetWidth(), input->GetHeight(), &confidence);
    const unsigned int bytes = image.cols * image.rows * sizeof(uchar) * 3;
    LOG(INFO) << "image.cols " << image.cols << "   image.rows " << image.rows << " byte size " << bytes;
    uchar *inputImage;
    cudaMalloc((uchar **) &inputImage, bytes);
    cudaMemcpy(inputImage, image.data, bytes, cudaMemcpyHostToDevice);
    const int img_class = net->Classify(inputImage, image.cols, image.rows, imageFormat::IMAGE_RGB8, &confidence);
    LOG(ERROR) << net->GetClassDesc(img_class);
    cudaFree(inputImage);
//    /*
//     * processing loop
//     */
//    while (!signal_recieved) {
//        // capture next image image
//        uchar3 *image = NULL;
//
//        if (!input->Capture(&image, 1000)) {


//            LogError("imagenet:  failed to capture next frame\n");
//            continue;
//        }
//
//        // classify image
//        float confidence = 0.0f;
//        const int img_class = net->Classify(image, input->GetWidth(), input->GetHeight(), &confidence);
//
//        if (img_class >= 0) {
//            LogVerbose("imagenet:  %2.5f%% class #%i (%s)\n", confidence * 100.0f, img_class,
//                       net->GetClassDesc(img_class));
//
//            if (font != NULL) {
//                char str[256];
//                sprintf(str, "%05.2f%% %s", confidence * 100.0f, net->GetClassDesc(img_class));
//
//                font->OverlayText(image, input->GetWidth(), input->GetHeight(),
//                                  str, 5, 5, make_float4(255, 255, 255, 255), make_float4(0, 0, 0, 100));
//            }
//        }
//
//        // render outputs
//        if (output != NULL) {
//            output->Render(image, input->GetWidth(), input->GetHeight());
//
//            // update status bar
//            char str[256];
//            sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
//                    NV_TENSORRT_PATCH, net->GetNetworkName(), net->GetNetworkFPS());
//            output->SetStatus(str);
//
//            // check if the user quit
//            if (!output->IsStreaming())
//                signal_recieved = true;
//        }
//
//        // check for EOS
//        if (!input->IsStreaming())
//            signal_recieved = true;
//
//        // print out timing info
//        net->PrintProfilerTimes();
//    }
//
//
//
//    SAFE_DELETE(output);
    SAFE_DELETE(net);

//    LogVerbose("imagenet:  shutdown complete.\n");
    return 0;
}

