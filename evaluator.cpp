#include <iostream>

// getopt
#include <unistd.h>
#include <queue>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>

#include <fstream> // std::ifstream

using namespace std;
using namespace cv;

typedef enum
{
    ACC = 0,
    DIST = 1
} metric_t;

const string label_path = "/media/ergale/SSD/Universita/Parma - Scienze Informatiche/1INF/2SEM/Visione artificiale per il veicolo/LaneDetection/test/labelcpp/valid/images-2014-12-22-12-35-10_mapping_280S_ramps/"; // Path of the ground truth lanes
// const string detected_path = "/media/ergale/SSD/Universita/Parma - Scienze Informatiche/1INF/2SEM/Visione artificiale per il veicolo/LaneDetection/test2/";                                                          // Path of the detected lanes
const string detected_path = "/media/ergale/SSD/Universita/Parma - Scienze Informatiche/1INF/2SEM/Visione artificiale per il veicolo/LaneDetection/output/";                                                 // Path of the detected lanes
const string image_path = "/media/ergale/SSD/Universita/Parma - Scienze Informatiche/1INF/2SEM/Visione artificiale per il veicolo/LaneDetection/image/color/images-2014-12-22-12-35-10_mapping_280S_ramps/"; // Path of the images
const string image_extension = "_color_rect.png";                                                                                                                                                            // Extension of the images
string image_name = "";
metric_t metric;
int pixel_margin = 50;
bool show_image = false;
bool all_images_in_dir = false;

bool parse_inputs(int argc, char **argv)
{
    int c;
    while ((c = getopt(argc, argv, "hi:m:p:sa")) != -1)
        switch (c)
        {
        case 'i':
            image_name = optarg;
            break;
        case 'm':
            if (atoi(optarg) == 1)
                metric = DIST;
            else
                metric = ACC;
            break;
        case 'p':
            pixel_margin = atoi(optarg);
            break;
        case 's':
            show_image = true;
            break;
        case 'a':
            all_images_in_dir = true;
            break;
        case 'h':
        default:
            std::cout << "usage: " << argv[0] << " -i <image_name> -m <number> -p <number>" << std::endl;
            std::cout << "exit:  type q" << std::endl
                      << std::endl;
            std::cout << "Allowed options:" << std::endl
                      << "   -h                       produce help message" << std::endl
                      << "   -i arg                   image name." << std::endl
                      << "   -m arg                   [optional] specificy the metrics: 0 (default) for the accuracy, 1 for the pixel distance" << std::endl
                      << "   -p arg                   [optional] pixel margin (default 15)"
                      << "   -s                       [optional] show image"
                      << "   -a                       [optional] process all images in the directory" << std::endl;
            return false;
        }

    if (image_name == "" && !all_images_in_dir)
    {
        std::cout << "The image name must be specified!" << std::endl;
        std::cout << "usage: " << argv[0] << " -i <image_name> -m <number> -p <number>" << std::endl;
        return false;
    }
    return true;
}

vector<double> get_line_from_file(ifstream &ifs)
{
    vector<double> line(417);

    for (int i = 0; i < 417; i++)
    {
        ifs >> line[i];
    }
    return line;
}

int get_lanes(string filename, vector<vector<double>> &lanes)
{
    ifstream lanes_file(filename, ios_base::in);

    if (lanes_file.is_open() == false)
    {
        cout << "File not found: " << filename << endl;
        return -1;
    }

    lanes[0] = get_line_from_file(lanes_file);
    lanes[1] = get_line_from_file(lanes_file);
    lanes[2] = get_line_from_file(lanes_file);
    lanes[3] = get_line_from_file(lanes_file);
    return 0;
}

double calculate_accuracy(vector<double> line1, vector<double> line2)
{
    double acc = 0;
    for (int i = 0; i < 417; i++)
    {
        if (abs(line1[i] - line2[i]) < pixel_margin)
            acc++;
    }
    return acc / 417;
}

int calculate_acc_metric(vector<vector<double>> detected_lanes, vector<vector<double>> ground_truth_lanes)
{
    cout << "Calculating accuracy metric..." << endl;

    double acc_l1 = calculate_accuracy(detected_lanes[0], ground_truth_lanes[0]);
    double acc_l0 = calculate_accuracy(detected_lanes[1], ground_truth_lanes[1]);
    double acc_r0 = calculate_accuracy(detected_lanes[2], ground_truth_lanes[2]);
    double acc_r1 = calculate_accuracy(detected_lanes[3], ground_truth_lanes[3]);

    cout << "l1 accuracy: " << acc_l1 << "\n"
         << "l0 accuracy: " << acc_l0 << "\n"
         << "r0 accuracy: " << acc_r0 << "\n"
         << "r1 accuracy: " << acc_r1 << "\n";

    cout << "Accuracy: " << (acc_l1 + acc_l0 + acc_r0 + acc_r1) / 4 << endl;
    return 0;
}

double calculate_distance(vector<double> line1, vector<double> line2)
{
    double dist = 0;
    int n = 0;
    for (int i = 0; i < 417; i++)
    {
        if (line1[i] != -1 && line2[i] != -1)
        {
            dist += abs(line1[i] - line2[i]);
            n++;
        }
    }
    if (n == 0)
        return 0;
    return dist / n;
}

int calculate_dist_metric(vector<vector<double>> detected_lanes, vector<vector<double>> ground_truth_lanes)
{
    double dist_l1 = calculate_distance(detected_lanes[0], ground_truth_lanes[0]);
    double dist_l0 = calculate_distance(detected_lanes[1], ground_truth_lanes[1]);
    double dist_r0 = calculate_distance(detected_lanes[2], ground_truth_lanes[2]);
    double dist_r1 = calculate_distance(detected_lanes[3], ground_truth_lanes[3]);

    cout << "l1 distance: " << dist_l1 << "\n"
         << "l0 distance: " << dist_l0 << "\n"
         << "r0 distance: " << dist_r0 << "\n"
         << "r1 distance: " << dist_r1 << "\n";

    cout << "Distance: " << (dist_l1 + dist_l0 + dist_r0 + dist_r1) / 4 << endl;
    return 0;
}

void show_images(vector<vector<double>> detected_lanes, vector<vector<double>> ground_truth_lanes)
{
    Mat image_detected_lanes = imread(image_path + image_name + image_extension);
    if (image_detected_lanes.empty())
    {
        cout << "Image not found: " << image_path + image_name + image_extension << endl;
        return;
    }
    Mat image_ground_lanes = image_detected_lanes.clone();

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 417; j++)
        {
            if (detected_lanes[i][j] != -1)
            {
                circle(image_detected_lanes, Point(detected_lanes[i][j], j + 300), 1, Scalar(0, 0, 255), -1);
            }
            if (ground_truth_lanes[i][j] != -1)
            {
                circle(image_ground_lanes, Point(ground_truth_lanes[i][j], j + 300), 1, Scalar(0, 255, 0), -1);
            }
        }
    }
    imshow("Detected lanes", image_detected_lanes);
    imshow("Ground truth lanes", image_ground_lanes);
    waitKey(0);
}

int main(int argc, char **argv)
{
    if (parse_inputs(argc, argv) == false)
    {
        return -1;
    }

    vector<vector<double>> detected_lanes(4, vector<double>()); // l1,l0,r0,r1
    vector<vector<double>> gt_lanes(4, vector<double>());       // l1,l0,r0,r1

    if (get_lanes(detected_path + image_name, detected_lanes) == -1)
    {
        std::cout << "The detected lanes file does not exist!" << std::endl;
        return -1;
    }

    if (get_lanes(label_path + image_name, gt_lanes) == -1)
    {
        std::cout << "The ground truth lanes file does not exist!" << std::endl;
        return -1;
    }

    if (metric == ACC)
    {
        calculate_acc_metric(detected_lanes, gt_lanes);
    }
    else
    {
        calculate_dist_metric(detected_lanes, gt_lanes);
    }

    if (show_image)
    {
        show_images(detected_lanes, gt_lanes);
    }
    return 0;
}