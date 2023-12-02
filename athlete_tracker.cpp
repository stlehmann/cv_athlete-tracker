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
#define MODEL_MEAN 127.5  // mean substraction value for model
#define MODEL_SCALE (1 / 127.5)  // scale for image color values when creating a blob

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
        cout << "Minimum confidence for athlete detection is " << min_confidence << endl;
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
    std::vector<dlib::drectangle> tracked_rects;
    std::vector<dlib::correlation_tracker> trackers;
    bool trackers_started = false;
    bool athlete_detected = false;

    // Capture first frame
    while (cap.isOpened()) {
        if (!cap.read(frame)) {
            cout << "Video has ended.";
            break;
        }

        int resized_height = 300;
        float aspect_ratio = static_cast<float>(resized_height) / static_cast<float>(frame_height);
        int resized_width = int(static_cast<float>(frame_width) * aspect_ratio);

        // resize frame to 300px height
        cv::Mat resized_frame;
        cv::resize(frame, resized_frame, Size(resized_width, resized_height));

        // keep aspect ratio but crop the x axis to 300px
        int cropped_left = resized_frame.cols - 300;
        cv::Rect cropped_rect(cropped_left, 0, 300, 300);
        cv::Mat cropped_frame = resized_frame(cropped_rect);
               
        // Define a region of interest (ROI) for tracking        
        if (!athlete_detected) {

            // If autodetection has been enabled start the object detector and mark the object with the highest confidence for the
            // class person
            if (autodetect) {
                // initialize the object detector
                dnn::Net object_detector = dnn::readNetFromCaffe(MODEL_PROTOTXT, MODEL_MODEL);
                
                // Create a blob that will be passed to object detection model
                Mat blob = dnn::blobFromImage(cropped_frame, MODEL_SCALE, Size(cropped_frame.cols, cropped_frame.rows), MODEL_MEAN, false);
                object_detector.setInput(blob);
                Mat detections = object_detector.forward();

                // Process the output to extract object detections
                Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());
                for (int i = 0; i < detectionMat.rows; ++i) {

                    // Extract confidence and label from detection matrix
                    float confidence = detectionMat.at<float>(i, 2);
                    string label = classes[static_cast<int>(detectionMat.at<float>(i, 1))];

                    if (label == "person" && confidence > min_confidence) {  // Adjust the confidence threshold as needed

                        cout << "Athlete " << i << " detected with confidence " << confidence << endl;
                        
                        // Extract all four corners of the bounding box, rescale to original size of the frame
                        int left = static_cast<int>((detectionMat.at<float>(i, 3) * cropped_frame.cols + cropped_left) / aspect_ratio);
                        int top = static_cast<int>(detectionMat.at<float>(i, 4) * cropped_frame.rows / aspect_ratio);
                        int right = static_cast<int>((detectionMat.at<float>(i, 5) * cropped_frame.cols + cropped_left) / aspect_ratio);
                        int bottom = static_cast<int>(detectionMat.at<float>(i, 6) * cropped_frame.rows / aspect_ratio);

                        // Add to tracked rectangles vector
                        tracked_rects.push_back(dlib::drectangle(left, top, right, bottom));

                        // Set flag that athlete has been detected, object detector won't be triggered again
                        athlete_detected = true;
                    }
                }
            }
            else {
                // If autodetect is not activate let the user manually create a rectangle around the athlete that shall be tracked
                Rect2d r = selectROI(frame);
                // Initialize the tracker rectangle
                tracked_rects.push_back(dlib::drectangle(r.x, r.y, r.x+r.width, r.y+r.height));
                athlete_detected = true;
            }
        }
        else {
            if (!trackers_started) {
                // Create a vector of correlation trackers for all found athletes
                for (size_t i=0; i< tracked_rects.size(); ++i) {
                    dlib::correlation_tracker tracker;
                    tracker.start_track(cv_image<rgb_pixel>(frame), tracked_rects[i]);
                    trackers.push_back(tracker);
                }
                trackers_started = true;
            }
            else {
                for (size_t i=0; i < trackers.size(); ++i) {
                    // Update each tracker with the new frame
                    trackers[i].update(cv_image<rgb_pixel>(frame));
                    
                    // Get the tracking result
                    dlib::rectangle drect = trackers[i].get_position();
                    tracked_rects[i] = drect;

                    // crop athlete for further processing
                    cv::Mat cropped_athlete = frame(cv::Rect(drect.left(), drect.top(), drect.width(), drect.height()));
                    cv::imshow("Athlete" + to_string(i + 1), cropped_athlete);

                    // Draw a rectangle around the tracked object
                    cv::rectangle(
                        frame,
                        cv::Point(drect.left(), drect.top()),
                        cv::Point(drect.right(), drect.bottom()),
                        cv::Scalar(0, 0, 255),
                        2
                    );

                    // Add label
                    cv::putText(
                        frame,
                        "Athlete" + std::to_string(i + 1),
                        cv::Point(drect.left(), drect.top() - 10),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cv::Scalar(0, 0, 255)
                    );

                    // Print coordinates
                    cout << "Athlete " << i+1 << ": x=" << tracked_rects[i].left() << ", y=" << tracked_rects[i].top() << ", width=" << tracked_rects[i].width() << ", height=" << tracked_rects[i].height() << endl;
                }
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
