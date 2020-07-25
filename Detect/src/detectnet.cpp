/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <opencv2/opencv.hpp>
#include "detectNet.h"
#include <glog/logging.h>

using namespace cv;
//#include <glog/logging.h>
/*
 * detectNet -- loading detection network model from:
          -- model        networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff
          -- input_blob   'Input'
          -- output_blob  'NMS'
          -- output_count 'NMS_1'
          -- class_labels networks/SSD-Mobilenet-v2/ssd_coco_labels.txt
          -- threshold    0.500000
          -- batch_size   1
          Create("networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff", "networks/SSD-Mobilenet-v2/ssd_coco_labels.txt",
          threshold, "Input", Dims3(3,300,300),
          "NMS", "NMS_1", maxBatchSize, precision, device, allowGPUFallback);
 */
int main(int argc, char **argv) {
//    imageNet::NetworkType type = imageNet::NetworkType::GOOGLENET;
    precisionType ptype = precisionType::TYPE_FP32;
    deviceType dtype = deviceType::DEVICE_GPU;
    std::string model_path = "/home/xzh/jetson-inference/data/networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff";
    std::string class_path = "/home/xzh/jetson-inference/data/networks/SSD-Mobilenet-v2/ssd_coco_labels.txt";
    float threshold = 0.5;
    std::string inputName = "Input";
    std::string outputName = "NMS";
    std::string numDetections = "NMS_1";
//    detectNet *net = detectNet::Create(model_path.data(),class_path.data(),threshold,inputName.data(),Dims3(3,300,300),
//            outputName.data(),numDetections.data(),1,ptype,dtype,true);

    detectNet *net = detectNet::Create(
            "/home/xzh/jetson-inference/data/networks/SSD-Mobilenet-v2/ssd_mobilenet_v2_coco.uff",
            "/home/xzh/jetson-inference/data/networks/SSD-Mobilenet-v2/ssd_coco_labels.txt", 0.5, "Input",
            Dims3(3, 300, 300),
            "NMS", "NMS_1", 1, ptype, dtype, true);

    if (!net) {
        LogError("detectnet:  failed to load detectNet model\n");
        return 0;
    } else {
//        LOG(INFO)<<"detectNet sucess";
    }
    cv::Mat origimage = cv::imread("/home/xzh/Pictures/1.jpg");
    cv::Mat image;
    cv::cvtColor(origimage, image, cv::COLOR_BGR2RGB);
    cv::imshow("image1",origimage);
    cv::waitKey(-1);
    detectNet::Detection *detections = NULL;
    uint32_t overlay = detectNet::OVERLAY_LABEL | detectNet::OVERLAY_BOX;
    LOG(INFO) << "start classify";
    float confidence = 0.0f;
    const unsigned int bytes = image.cols * image.rows * sizeof(uchar) * 3;
    LOG(INFO) << "image.cols " << image.cols << "   image.rows " << image.rows << " byte size " << bytes;
    uchar *inputImage;
    cudaMalloc((uchar **) &inputImage, bytes);
    cudaMemcpy(inputImage, image.data, bytes, cudaMemcpyHostToDevice);
    int detectionsNum = net->Detect(inputImage, image.cols, image.rows, imageFormat::IMAGE_RGB8, &detections,
                                    overlay);
    if (detectionsNum > 0) {
        for (int n = 0; n < detectionsNum; n++) {
            LOG(INFO) << detections[n].ClassID << "  " << net->GetClassDesc(detections[n].ClassID) << "  "
                      << detections[n].Confidence;
            LOG(INFO) << detections[n].Left << "  " << detections[n].Top << "  "
                      << detections[n].Right << "  " << detections[n].Bottom << "  " << detections[n].Width() << "  " <<
                      detections[n].Height();
            cv::Rect rect(detections[n].Left, detections[n].Top, detections[n].Width(), detections[n].Height());
            cv::rectangle(origimage, rect, Scalar(255, 0, 0), 1, LINE_8, 0);
        }
    }
    cv::imshow("image",origimage);
    cv::waitKey(-1);
    // parse overlay flags
//    const uint32_t overlayFlags = detectNet::OverlayFlagsFromStr(cmdLine.GetString("overlay", "box,labels,conf"));

//    /*
//     * processing loop
//     */
//    while (!signal_recieved) {
//        // capture next image image
//        uchar3 *image = NULL;
//
//        if (!input->Capture(&image, 1000)) {
//            LogError("detectnet:  failed to capture video frame\n");
//            continue;
//        }
//
//        // detect objects in the frame
//        detectNet::Detection *detections = NULL;
//
//        const int numDetections = net->Detect(image, input->GetWidth(), input->GetHeight(), &detections, overlayFlags);
//
//        if (numDetections > 0) {
//            LogVerbose("%i objects detected\n", numDetections);
//
//            for (int n = 0; n < numDetections; n++) {
//                LogVerbose("detected obj %i  class #%u (%s)  confidence=%f\n", n, detections[n].ClassID,
//                           net->GetClassDesc(detections[n].ClassID), detections[n].Confidence);
//                LogVerbose("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, detections[n].Left,
//                           detections[n].Top, detections[n].Right, detections[n].Bottom, detections[n].Width(),
//                           detections[n].Height());
//            }
//        }
//
//        // render outputs
//        if (output != NULL) {
//            output->Render(image, input->GetWidth(), input->GetHeight());
//
//            // update the status bar
//            char str[256];
//            sprintf(str, "TensorRT %i.%i.%i | %s | Network %.0f FPS", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
//                    NV_TENSORRT_PATCH, precisionTypeToStr(net->GetPrecision()), net->GetNetworkFPS());
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
//    /*
//     * destroy resources
//     */
//    LogVerbose("detectnet:  shutting down...\n");
//
//    SAFE_DELETE(input);
//    SAFE_DELETE(output);
//    SAFE_DELETE(net);

//    LogVerbose("detectnet:  shutdown complete.\n");
    return 0;
}

