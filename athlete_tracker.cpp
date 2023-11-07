#include <iostream>
#include <dlib/image_processing.h>
#include <dlib/opencv/cv_image.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>
#include "boost/program_options.hpp"


#define VIDEO_FILE "../../input/ingebrigsten_munich_2022.mp4"
#define MODEL_PROTOTXT "../../model/MobileNetSSD_deploy.prototxt"
#define MODEL_MODEL "../../model/MobileNetSSD_deploy.caffemodel"
#define SCALE (1 / 127.5)  // scale for image color values when creating a blob
#define MAX_N_DETECTIONS 1  // maximum number of object detections per frame

using namespace std;
using namespace cv;
using namespace dlib;
namespace po = boost::program_options;

string classes[21] = {
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"
};

int main(int argc, char* argv[]) {
    
    // flag to enable autodetection of the athlete
    bool autodetect = false;

    // CLI config
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("input-file", po::value<string>(), "Input file")
        ("output-file", po::value<string>(), "Output file (*.avi)")
        ("autodetect,a", po::bool_switch(&autodetect), "enable athlete autodetection")
        ("confidence,c", po::value<float>(), "min. detection confidence")
    ;

    // Parse parameters
    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (const po::error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    // display CLI help
    if (vm.count("help")) {
        cout << "Usage: " << argv[0] << " [options] <description of positional 1> <description of positional 2> ...\n";
        cout << desc;
        return 0;
    }
    
    // Input-file is mandatory
    if (!vm.count("input-file")) {
        cout << "Please provide an input-file." << endl;
        return 1;
    }

    // Confidence
    float min_confidence = 0.9;
    if (vm.count("confidence")) {
        min_confidence = vm["confidence"].as<float>();
    }

    // Open the input file
    cv::VideoCapture cap(vm["input-file"].as<string>()); 
    cv::Mat frame;
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video file." << std::endl;
        return 1;
    }

    // Open output
    VideoWriter output;
    if (vm.count("output-file")) {
        int frame_width = static_cast<int>(cap.get(3));
        int frame_height = static_cast<int>(cap.get(4));
        Size frame_size(frame_width, frame_height);
        int fps = 20;
        output = VideoWriter(vm["output-file"].as<string>(), VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame_size);
    }

    // init the correlation tracker
    dlib::correlation_tracker tracker;
    bool tracker_started = false;
    dlib::drectangle drect;
    bool athlete_detected = false;

    // Capture first frame
    while (cap.isOpened()) {
        if (!cap.read(frame)) {
            cout << "Video has ended.";
            break;
        }
                
        // Define a region of interest (ROI) for tracking        
        if (!athlete_detected) {
            if (autodetect) {
                // initalize the object detector
                dnn::Net object_detector = dnn::readNetFromCaffe(MODEL_PROTOTXT, MODEL_MODEL);
                int image_height = frame.rows;
                int image_width = frame.cols;

                // Create a blob that will be passed to object detection model
                Mat blob = dnn::blobFromImage(frame, SCALE, Size(image_width, image_height), 150);
                object_detector.setInput(blob);
                Mat detections = object_detector.forward();

                // Process the output to extract object detections
                Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
                for (int i = 0; i < min(MAX_N_DETECTIONS, detectionMat.rows); ++i) {
                    float confidence = detectionMat.at<float>(i, 2);
                    string label = classes[static_cast<int>(detectionMat.at<float>(i, 1))];

                    if (confidence > min_confidence && label == "person") {  // Adjust the confidence threshold as needed
                        int left = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                        int top = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                        int right = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                        int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                        // fire up the correlation tracker
                        drect = dlib::drectangle(left, top, right, bottom);
                        athlete_detected = true;

                        // draw rectangle
                        cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 2);
                    }
                }
            }
            else {
                Rect2d r = selectROI(frame);
                drect = dlib::drectangle(r.x, r.y, r.x+r.width, r.y+r.height);
                athlete_detected = true;
            }
        }
        else {
            if (!tracker_started) {
                // start the tracker    
                tracker.start_track(cv_image<rgb_pixel>(frame), drect);
                tracker_started = true;
            }
            else {
                // Update the tracker with the new frame
                tracker.update(cv_image<rgb_pixel>(frame));

                // Get the tracking result
                dlib::drectangle trackedRect = tracker.get_position();

                // Draw a rectangle around the tracked object
                cv::rectangle(frame, cv::Point(trackedRect.left(), trackedRect.top()), cv::Point(trackedRect.right(), trackedRect.bottom()), cv::Scalar(0, 0, 255), 2);

                // Print coordinates
                cout << "x=" << trackedRect.left() << ", y=" << trackedRect.top() << ", width=" << trackedRect.width() << ", height=" << trackedRect.height() << endl;
            }
        }
       
        // Display the tracked
        cv::imshow("Tracker", frame);
       
        // Write to output file (only AVI working)
        if (vm.count("output-file")) {
            output.write(frame);
        }

        if (cv::waitKey(10) == 'q') {
            break;
        }

    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
