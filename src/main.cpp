#include <iostream>
#include <unistd.h>
#include <memory>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "detector.h"
#include "cxxopts.hpp"

using namespace std;
using namespace cv;

#define LOCAL_STREAM "local_stream"
#define IP_STREAM "ip_stream"

std::vector<std::string> LoadNames(const std::string& path) {
    // load class names
    std::vector<std::string> class_names;
    std::ifstream infile(path);
    if (infile.is_open()) {
        std::string line;
        while (getline (infile,line)) {
            class_names.emplace_back(line);
        }
        infile.close();
    }
    else {
        std::cerr << "Error loading the class names!\n";
    }

    return class_names;
}


void Demo(cv::Mat& img,
        const std::vector<std::vector<Detection>>& detections,
        const std::vector<std::string>& class_names,
        bool label = true, bool image_show = false) {

    if (!detections.empty()) {
        for (const auto& detection : detections[0]) {
            const auto& box = detection.bbox;
            float score = detection.score;
            int class_idx = detection.class_idx;

            cv::rectangle(img, box, cv::Scalar(0, 0, 255), 2);

            if (label) {
                std::stringstream ss;
                ss << std::fixed << std::setprecision(2) << score;
                std::string s = class_names[class_idx] + " " + ss.str();

                auto font_face = cv::FONT_HERSHEY_DUPLEX;
                auto font_scale = 1.0;
                int thickness = 1;
                int baseline=0;
                auto s_size = cv::getTextSize(s, font_face, font_scale, thickness, &baseline);
                cv::rectangle(img,
                        cv::Point(box.tl().x, box.tl().y - s_size.height - 5),
                        cv::Point(box.tl().x + s_size.width, box.tl().y),
                        cv::Scalar(0, 0, 255), -1);
                cv::putText(img, s, cv::Point(box.tl().x, box.tl().y - 5),
                            font_face , font_scale, cv::Scalar(255, 255, 255), thickness);
            }
        }
    }
    if (image_show) {
        cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
        cv::imshow("Result", img);
        cv::waitKey(0);
    }
}

void do_object_detection(string streamUrl, Detector detector, float conf_thres, float iou_thres, vector<string> class_names, bool view_img) {
    cout << "do object detection:" << streamUrl << endl;
    Mat src;
    // float scaling_factor = 0.5;
    cout << "opencv videocapture opening stream" << endl;

    VideoCapture capture(0);
    cout << "opencv videocapture opening stream......" << endl;
    if (!capture.isOpened()) {
        cout << "opencv videocapture open stream failed" << endl;
        return;
    }
    cout << "opencv videocapture open stream success" << endl;

    capture.set(CAP_PROP_FRAME_WIDTH, 1280);
    capture.set(CAP_PROP_FRAME_HEIGHT, 720);

    capture >> src;

    // run once to warm up
    std::cout << "Run once on empty image" << std::endl;
    auto temp_img = cv::Mat::zeros(src.rows, src.cols, CV_32FC3);
    detector.Run(temp_img, 1.0f, 1.0f);

    VideoWriter writer;
    // select desired codec (must be available at runtime)
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
    // framerate of the created video stream
    double fps = capture.get(CAP_PROP_FPS);
    cout << "video writer fps: " << fps << endl;
    // name of the output video file
    string filename = "/home/cds/mycode/avdevelopment/libtorch-yolov5/build/live.avi";
    bool isColor = (src.type() == CV_8UC3);
    // src.size()
    // writer.open(filename, codec, fps, Size(round(scaling_factor * src.cols), round(scaling_factor * src.rows)), isColor);
    writer.open(filename, codec, fps, src.size(), isColor);
    // check if we succeeded
    if (!writer.isOpened()) {
        cout << "opencv could not open the output video file for write" << endl;
        return;
    }
    cout << "opencv open the output video file for writing success" << endl;
    while (capture.isOpened()) {
        bool ok = capture.read(src);
        if (!ok || src.empty()) {
            continue;
        }

        // Mat dst;
        // resize(src, dst, Size(), scaling_factor, scaling_factor, INTER_AREA);

        auto result = detector.Run(src, conf_thres, iou_thres);
        // visualize detections
        Demo(src, result, class_names, true);
        if (view_img) {
            namedWindow("Result", WINDOW_AUTOSIZE);
            imshow("Result", src);
            waitKey(30);
        } else {
            writer.write(src);
            cout << "writing data to deatination file" << endl;
        }
    }
    capture.release();
}

bool is_path_exist(const std::string &path) {
    cout << "check file......" << endl;
    if (access(path.c_str(), 0) == F_OK) {
        cout << "file exist" << endl;
        return true;
    }
    cout << "file is not existed" << endl;
    return false;
}

int main(int argc, const char* argv[]) {
    cxxopts::Options parser(argv[0], "A LibTorch inference implementation of the yolov5");

    // TODO: add other args
    parser.allow_unrecognised_options().add_options()
            ("weights", "model.torchscript.pt path", cxxopts::value<std::string>())
            ("source", "source", cxxopts::value<std::string>())
            ("conf-thres", "object confidence threshold", cxxopts::value<float>()->default_value("0.4"))
            ("iou-thres", "IOU threshold for NMS", cxxopts::value<float>()->default_value("0.5"))
            ("gpu", "Enable cuda device or cpu", cxxopts::value<bool>()->default_value("false"))
            ("view-img", "display results", cxxopts::value<bool>()->default_value("false"))
            ("h,help", "Print usage");

    auto opt = parser.parse(argc, argv);

    if (opt.count("help")) {
        std::cout << parser.help() << std::endl;
        exit(0);
    }

    // check if gpu flag is set
    bool is_gpu = opt["gpu"].as<bool>();

    // set device type - CPU/GPU
    torch::DeviceType device_type;
    if (torch::cuda::is_available() && is_gpu) {
        device_type = torch::kCUDA;
    } else {
        device_type = torch::kCPU;
    }

    // load class names from dataset for visualization
    std::vector<std::string> class_names = LoadNames("../weights/coco.names");
    if (class_names.empty()) {
        return -1;
    }

    // load network
    std::string weights = opt["weights"].as<std::string>();
    auto detector = Detector(weights, device_type);

    // load input image
    std::string source = opt["source"].as<std::string>();
    // set up threshold
    float conf_thres = opt["conf-thres"].as<float>();
    float iou_thres = opt["iou-thres"].as<float>();
    bool view_img = opt["view-img"].as<bool>();
    if (is_path_exist(source)) {
        cv::Mat img = cv::imread(source);
        if (img.empty()) {
            std::cerr << "Error loading the image!\n";
            return -1;
        }
        // run once to warm up
        std::cout << "Run once on empty image" << std::endl;
        auto temp_img = cv::Mat::zeros(img.rows, img.cols, CV_32FC3);
        detector.Run(temp_img, 1.0f, 1.0f);
        // inference
        auto result = detector.Run(img, conf_thres, iou_thres);

        // visualize detections
        if (view_img) {
            Demo(img, result, class_names, true, true);
        }

        cv::destroyAllWindows();
    } else {
        if (source != LOCAL_STREAM && source != IP_STREAM) {
            std::cerr << "Param source is invalid!\n";
            return -1;
        } else {
            if (source == LOCAL_STREAM) {
                do_object_detection(source, detector, conf_thres, iou_thres, class_names, view_img);
            }
        }
    }

    return 0;
}