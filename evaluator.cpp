#include <iostream>

// getopt
#include <unistd.h>
#include <queue>
#include <string>

#include <fstream> // std::ifstream

using namespace std;

typedef enum
{
    ACC = 0,
    DIST = 1
} metric_t;

const string label_path = "/media/ergale/SSD/Universita/Parma - Scienze Informatiche/1INF/2SEM/Visione artificiale per il veicolo/LaneDetection/test/labelcpp/valid/images-2014-12-22-12-35-10_mapping_280S_ramps/"; // Path of the ground truth laness
const string detected_path = "/media/ergale/SSD/Universita/Parma - Scienze Informatiche/1INF/2SEM/Visione artificiale per il veicolo/LaneDetection/test2/";                                                          // Path of the detected lanes
string image_name = "";
metric_t metric;
int pixel_margin = 50;

bool parse_inputs(int argc, char **argv)
{
    int c;
    while ((c = getopt(argc, argv, "hi:m:p:")) != -1)
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
        case 'h':
        default:
            std::cout << "usage: " << argv[0] << " -i <image_name> -m <number> -p <number>" << std::endl;
            std::cout << "exit:  type q" << std::endl
                      << std::endl;
            std::cout << "Allowed options:" << std::endl
                      << "   -h                       produce help message" << std::endl
                      << "   -i arg                   image name." << std::endl
                      << "   -m arg                   [optional] specificy the metrics: 0 (default) for the accuracy, 1 for the pixel distance" << std::endl
                      << "   -p arg                   [optional] pixel margin (default 15)" << std::endl;
            return false;
        }

    if (image_name == "")
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

int get_lanes(string filename, vector<double> &l1, vector<double> &l0, vector<double> &r0, vector<double> &r1)
{
    ifstream lanes_file(filename, ios_base::in);

    if (lanes_file.is_open() == false)
    {
        cout << "File not found: " << filename << endl;
        return -1;
    }

    l1 = get_line_from_file(lanes_file);
    l0 = get_line_from_file(lanes_file);
    r0 = get_line_from_file(lanes_file);
    r1 = get_line_from_file(lanes_file);
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

int calculate_acc_metric()
{
    cout << "Calculating accuracy metric..." << endl;

    vector<double> detected_l1, detected_l0, detected_r0, detected_r1;
    vector<double> gt_l1, gt_l0, gt_r0, gt_r1;

    if (get_lanes(detected_path + image_name, detected_l1, detected_l0, detected_r0, detected_r1) == -1)
    {
        std::cout << "The detected lanes file does not exist!" << std::endl;
        return -1;
    }

    if (get_lanes(label_path + image_name, gt_l1, gt_l0, gt_r0, gt_r1) == -1)
    {
        std::cout << "The ground truth lanes file does not exist!" << std::endl;
        return -1;
    }

    double acc_l1 = calculate_accuracy(gt_l1, detected_l1);
    double acc_l0 = calculate_accuracy(gt_l0, detected_l0);
    double acc_r0 = calculate_accuracy(gt_r0, detected_r0);
    double acc_r1 = calculate_accuracy(gt_r1, detected_r1);

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

int calculate_dist_metric()
{
    std::cout << "Calculating pixel distance metric..." << std::endl;

    vector<double> detected_l1, detected_l0, detected_r0, detected_r1;
    vector<double> gt_l1, gt_l0, gt_r0, gt_r1;

    if (get_lanes(detected_path + image_name, detected_l1, detected_l0, detected_r0, detected_r1) == -1)
    {
        std::cout << "The detected lanes file does not exist!" << std::endl;
        return -1;
    }

    if (get_lanes(label_path + image_name, gt_l1, gt_l0, gt_r0, gt_r1) == -1)
    {
        std::cout << "The ground truth lanes file does not exist!" << std::endl;
        return -1;
    }

    double dist_l1 = calculate_distance(gt_l1, detected_l1);
    double dist_l0 = calculate_distance(gt_l0, detected_l0);
    double dist_r0 = calculate_distance(gt_r0, detected_r0);
    double dist_r1 = calculate_distance(gt_r1, detected_r1);

    cout << "l1 distance: " << dist_l1 << "\n"
         << "l0 distance: " << dist_l0 << "\n"
         << "r0 distance: " << dist_r0 << "\n"
         << "r1 distance: " << dist_r1 << "\n";

    cout << "Distance: " << (dist_l1 + dist_l0 + dist_r0 + dist_r1) / 4 << endl;
    return 0;
}

int main(int argc, char **argv)
{
    if (parse_inputs(argc, argv) == false)
    {
        return -1;
    }

    if (metric == ACC)
    {
        calculate_acc_metric();
    }
    else
    {
        calculate_dist_metric();
    }
    return 0;
}