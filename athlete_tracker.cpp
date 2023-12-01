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

    // Extract relevant video properties
    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
    int fps = static_cast<int>(cap.get(CAP_PROP_FPS));    
    
    cout << "Input video width: " << frame_width << "px, height: " << frame_height << "px, fps: " << fps << endl << endl;

    // Open output 
    VideoWriter output;
    if (vm.count("output-file")) {   
        Size frame_size(frame_width, frame_height);
        output = VideoWriter(vm["output-file"].as<string>(), VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame_size);
    }

    // init the correlation tracker
    dlib::correlation_tracker tracker;
    dlib::drectangle tracker_drect;
    bool tracker_started = false;
    bool athlete_detected = false;  

    // Capture first frame
    while (cap.isOpened()) {
        if (!cap.read(frame)) {
            cout << "Video has ended.";
            break;
        }
                
        // Define a region of interest (ROI) for tracking        
        if (!athlete_detected) {

            // If autodetection has been enabled start the object detector and mark the object with the highest confidence for the
            // class person
            if (autodetect) {
                // initialize the object detector
                dnn::Net object_detector = dnn::readNetFromCaffe(MODEL_PROTOTXT, MODEL_MODEL);
                
                // Create a blob that will be passed to object detection model
                Mat blob = dnn::blobFromImage(frame, SCALE, Size(frame_width, frame_height), 150);
                object_detector.setInput(blob);
                Mat detections = object_detector.forward();

                // Process the output to extract object detections. We only want the one with the highest confidence
                Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
                for (int i = 0; i < min(MAX_N_DETECTIONS, detectionMat.rows); ++i) {

                    // extract confidence and label from detection matrix
                    float confidence = detectionMat.at<float>(i, 2);
                    string label = classes[static_cast<int>(detectionMat.at<float>(i, 1))];

                    if (label == "person" && confidence > min_confidence) {  // Adjust the confidence threshold as needed
                        
                        // set flag that athlete has been detected, object detector won't be triggered again
                        athlete_detected = true;

                        // extract all four corners of the bounding box
                        int left = static_cast<int>(detectionMat.at<float>(i, 3) * frame_width);
                        int top = static_cast<int>(detectionMat.at<float>(i, 4) * frame_height);
                        int right = static_cast<int>(detectionMat.at<float>(i, 5) * frame_width);
                        int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * frame_height);

                        // Initialize the tracker rectangle
                        tracker_drect = dlib::drectangle(left, top, right, bottom);

                        // draw rectangle
                        cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 255), 2);
                    }
                }
            }
            else {
                // If autodetect is not activate let the user manually create a rectangle around the athlete that shall be tracked
                Rect2d r = selectROI(frame);
                // Initialize the tracker rectangle
                tracker_drect = dlib::drectangle(r.x, r.y, r.x+r.width, r.y+r.height);
                athlete_detected = true;
            }
        }
        else {
            if (!tracker_started) {
                // start the tracker if it hasn't been started, yet
                tracker.start_track(cv_image<rgb_pixel>(frame), tracker_drect);
                tracker_started = true;
            }
            else {
                // Update the tracker with the new frame
                tracker.update(cv_image<rgb_pixel>(frame));

                // Get the tracking result
                tracker_drect = tracker.get_position();

                // Draw a rectangle around the tracked object
                cv::rectangle(frame, cv::Point(tracker_drect.left(), tracker_drect.top()), cv::Point(tracker_drect.right(), tracker_drect.bottom()), cv::Scalar(0, 0, 255), 2);

                // Print coordinates
                cout << "x=" << tracker_drect.left() << ", y=" << tracker_drect.top() << ", width=" << tracker_drect.width() << ", height=" << tracker_drect.height() << endl;
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
