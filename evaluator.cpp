#include <iostream>
#include <dirent.h>

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

typedef struct
{
    string image_name;
    metric_t metric;
    int pixel_margin;
    bool show_image;
    bool all_images_in_dir;
} args_t;

const string label_path = "/media/ergale/SSD/Universita/Parma - Scienze Informatiche/1INF/2SEM/Visione artificiale per il veicolo/LaneDetection/test2/";                                                     // Path of the ground truth lanes
const string detected_path = "/media/ergale/SSD/Universita/Parma - Scienze Informatiche/1INF/2SEM/Visione artificiale per il veicolo/LaneDetection/output/";                                                 // Path of the detected lanes
const string image_path = "/media/ergale/SSD/Universita/Parma - Scienze Informatiche/1INF/2SEM/Visione artificiale per il veicolo/LaneDetection/image/color/images-2014-12-22-12-35-10_mapping_280S_ramps/"; // Path of the images
const string image_extension = "_color_rect.png";                                                                                                                                                            // Extension of the images

args_t args_params = {.image_name = "",
                      .metric = ACC,
                      .pixel_margin = 50,
                      .show_image = false,
                      .all_images_in_dir = false};

bool parse_inputs(int argc, char **argv)
{
    int c;
    while ((c = getopt(argc, argv, "hi:m:p:sa")) != -1)
        switch (c)
        {
        case 'i':
            args_params.image_name = optarg;
            break;
        case 'm':
            if (atoi(optarg) == 1)
                args_params.metric = DIST;
            else
                args_params.metric = ACC;
            break;
        case 'p':
            args_params.pixel_margin = atoi(optarg);
            break;
        case 's':
            args_params.show_image = true;
            break;
        case 'a':
            args_params.all_images_in_dir = true;
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

    if (args_params.image_name == "" && args_params.all_images_in_dir == false)
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

void calculate_accuracy(vector<double> gt_line, vector<double> detected_line, double &precision, double &recall)
{
    double tp = 0;
    double fp = 0;
    double fn = 0;
    double tn = 0;
    double counter = 0;
    for (int i = 0; i < 417; i++)
    {
        if (gt_line[i] == -1 && detected_line[i] == -1)
            tn++;
        else if (gt_line[i] == -1 && detected_line[i] != -1)
            fp++;
        else if (gt_line[i] != -1 && detected_line[i] == -1)
            fn++;
        else if (gt_line[i] != -1 && detected_line[i] != -1 && abs(gt_line[i] - detected_line[i]) < args_params.pixel_margin)
            tp++;
        else // Too far
            fn++;
    }
    if (tp + fp == 0)
        precision = 1;
    else
        precision = tp / (tp + fp);

    if (tp + fn == 0)
        recall = 1;
    else
        recall = tp / (tp + fn);
}

int calculate_acc_metric(vector<vector<double>> detected_lanes, vector<vector<double>> ground_truth_lanes)
{
    cout << "Calculating accuracy metric..." << endl;

    double precision_l1, recall_l1, precision_l0, recall_l0, precision_r1, recall_r1, precision_r0, recall_r0;
    calculate_accuracy(ground_truth_lanes[0], detected_lanes[0], precision_l1, recall_l1);
    calculate_accuracy(ground_truth_lanes[1], detected_lanes[1], precision_l0, recall_l0);
    calculate_accuracy(ground_truth_lanes[2], detected_lanes[2], precision_r1, recall_r1);
    calculate_accuracy(ground_truth_lanes[3], detected_lanes[3], precision_r0, recall_r0);
    cout << "L1: precision:" << precision_l1 << " recall:" << recall_l1 << endl;
    cout << "L0: precision:" << precision_l0 << " recall:" << recall_l0 << endl;
    cout << "R1: precision:" << precision_r1 << " recall:" << recall_r1 << endl;
    cout << "R0: precision:" << precision_r0 << " recall:" << recall_r0 << endl;
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

void show_images(string image_name, vector<vector<double>> detected_lanes, vector<vector<double>> ground_truth_lanes)
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

bool calculate_metric_on_image(string image_name)
{
    vector<vector<double>> detected_lanes(4, vector<double>()); // l1,l0,r0,r1
    vector<vector<double>> gt_lanes(4, vector<double>());       // l1,l0,r0,r1

    if (get_lanes(detected_path + image_name, detected_lanes) == -1)
    {
        std::cout << "The detected lanes file does not exist!" << std::endl;
        return false;
    }

    if (get_lanes(label_path + image_name, gt_lanes) == -1)
    {
        std::cout << "The ground truth lanes file does not exist!" << std::endl;
        return false;
    }

    if (args_params.metric == ACC)
    {
        calculate_acc_metric(detected_lanes, gt_lanes);
    }
    else
    {
        calculate_dist_metric(detected_lanes, gt_lanes);
    }

    if (args_params.show_image)
    {
        show_images(image_name, detected_lanes, gt_lanes);
    }

    return true;
}

vector<string> get_all_images_in_dir()
{
    vector<string> image_names;
    DIR *dir;
    struct dirent *diread;
    if ((dir = opendir(detected_path.c_str())) != nullptr)
    {
        while ((diread = readdir(dir)) != nullptr)
        {
            string im_name = string(diread->d_name);
            image_names.push_back(im_name);
        }
        closedir(dir);
    }
    else
    {
        perror("opendir");
    }
    return image_names;
}

int main(int argc, char **argv)
{
    if (parse_inputs(argc, argv) == false)
    {
        return -1;
    }
    bool ok = false;

    if (args_params.all_images_in_dir)
    {
        vector<string> image_names = get_all_images_in_dir();
        for (int i = 0; i < image_names.size(); i++)
        {
            cout << "Image: " << image_names[i] << endl;
            ok = calculate_metric_on_image(image_names[i]);
            if (ok == false)
            {
                cout << "Error in image: " << image_names[i] << endl;
                return -1;
            }
        }
    }
    else
    {
        ok = calculate_metric_on_image(args_params.image_name);
        if (ok == false)
        {
            cout << "Error in image: " << args_params.image_name << endl;
            return -1;
        }
    }
    return 0;
}