/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <fstream>
#include <json/json.h>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <random>
#include <string>
#include <vector>

#include "yolov5.h"

int coco80_to_coco91_class(int id)
{
    // # converts 80-index (val2014) to 91-index (paper)
    // # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    // # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    // # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    // # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    // # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    std::vector<int> x = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                          22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                          46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
                          67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};
    return x[id];
}

std::vector<float> xyxy2xywh(float x0, float x1, float x2, float x3)
{
    // # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    // y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    return {x0, x1, x2 - x0, x3 - x1};
}

std::vector<std::string> readCocoPaths(std::string coco_file_path)
{
    std::vector<std::string> result;
    const std::string        test_file_list = "./data/coco_val_2017_list.txt";
    std::ifstream            coco_test_file(test_file_list);
    std::string              line;
    if (coco_file_path.back() != '/')
    {
        coco_file_path += '/';
    }
    if (coco_test_file)
    {
        while (getline(coco_test_file, line))
        {
            result.push_back(coco_file_path + line);
        }
    }
    return result;
}

class InputParser
{
  public:
    InputParser(int &argc, char **argv)
    {
        for (int i = 1; i < argc; ++i)
            this->tokens.push_back(std::string(argv[i]));
    }
    std::string getCmdOption(const std::string &option) const
    {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end())
        {
            return *itr;
        }
        static std::string empty_string("");
        return empty_string;
    }
    bool cmdOptionExists(const std::string &option) const
    {
        return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
    }

  private:
    std::vector<std::string> tokens;
};

int main(int argc, char **argv)
{
    InputParser input(argc, argv);
    if (input.cmdOptionExists("-h"))
    {
        printf("Usage 1: ./validate_coco --engine path_to_engine_or_loadable  --coco_path path_to_coco_dataset "
               "--backend cudla_fp16/cudla_int8\n");
        printf("Usage 2: ./validate_coco --engine path_to_engine_or_loadable  --image path_to_image --backend "
               "cudla_fp16/cudla_int8\n");
        return 0;
    }
    std::string engine_path = input.getCmdOption("--engine");
    if (engine_path.empty())
    {
        printf("Error: please specify the loadable path with --engine");
        return 0;
    }
    std::string backend_str = input.getCmdOption("--backend");
    std::string coco_path   = input.getCmdOption("--coco_path");
    std::string image_path  = input.getCmdOption("--image");

    Yolov5Backend backend = Yolov5Backend::CUDLA_FP16;
    if (backend_str == "cudla_fp16")
    {
        backend = Yolov5Backend::CUDLA_FP16;
    }
    if (backend_str == "cudla_int8")
    {
        backend = Yolov5Backend::CUDLA_INT8;
    }

    yolov5 yolov5_infer(engine_path, backend);

    std::vector<cv::Mat>            bgr_imgs;
    std::vector<std::string>        imgPathList = readCocoPaths(coco_path);
    std::vector<std::vector<float>> results;

    if (!image_path.empty())
    {
        printf("Run Yolov5 DLA pipeline for %s\n", image_path.c_str());
        cv::Mat one_img = cv::imread(image_path);
        bgr_imgs.push_back(one_img);
        std::vector<cv::Mat> nchwMats = yolov5_infer.preProcess4Validate(bgr_imgs);

        yolov5_infer.infer();
        results = yolov5_infer.postProcess4Validation(0.25, 0.5f);
        printf("Num object detect: %ld\n", results.size());
        for (auto &item : results)
        {
            // left, top, right, bottom, label, confident
            cv::rectangle(one_img, cv::Point(item[0], item[1]), cv::Point(item[2], item[3]), cv::Scalar(0, 255, 0), 2,
                          16);
        }
        printf("detect result has been write to result.jpg\n");
        cv::imwrite("result.jpg", one_img);
        return 0;
    }

    Json::Value      root;
    Json::FastWriter writer;

    for (size_t i = 0; i < imgPathList.size(); i++)
    {
        cv::Mat one_img = cv::imread(imgPathList[i]);
        bgr_imgs.push_back(one_img);
        std::vector<cv::Mat> nchwMats = yolov5_infer.preProcess4Validate(bgr_imgs);

        printf("\r%ld / %ld  ", i, imgPathList.size());
        fflush(stdout);

        yolov5_infer.infer();
        results = yolov5_infer.postProcess4Validation(0.001f, 0.65f);
        printf("Num object detect: %ld\n", results.size());

        // processing the name. eg: ./images/train2017/000000000250.jpg will be processed as 250
        int image_id = stoi(imgPathList[i].substr(imgPathList[i].length() - 16,
                                                  imgPathList[i].find_last_of(".") - (imgPathList[i].length() - 16)));
        for (size_t k = 0; k < results.size(); k++)
        {
            Json::Value OneResult;
            Json::Value bboxObj;
            OneResult["image_id"]    = image_id;
            OneResult["category_id"] = coco80_to_coco91_class(results[k][4]);
            OneResult["score"]       = results[k][5];

            std::vector<float> point = xyxy2xywh(results[k][0], results[k][1], results[k][2], results[k][3]);
            bboxObj.append(point[0]);
            bboxObj.append(point[1]);
            bboxObj.append(point[2]);
            bboxObj.append(point[3]);
            OneResult["bbox"] = bboxObj;
            root.append(OneResult);
        }
        bgr_imgs.clear();
        results.clear();
    }

    std::string   json_file = writer.write(root);
    std::ofstream out("./predict.json");
    out << json_file;
    std::cout << "predict result has been written to ./predict.json " << std::endl;

    return 0;
}