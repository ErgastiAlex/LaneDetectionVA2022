#include <iostream>
#include <dirent.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/LU>

// getopt
#include <unistd.h>
#include <queue>
#include <string>
#include <float.h>
#include <fstream>

#include "linedetector.hh"

using namespace cv;
using namespace std;

/*
[[1381.9564208984375, 0, 648.1651000976562],
[0, 1394.720703125, 369.2019348144531],
[0, 0, 1]]
*/

const string output_path = "../output/";
const string image_path = "../image/color/images-2014-12-22-12-35-10_mapping_280S_ramps/";
const int ipm_width = 400;
const int ipm_height = 1000;
const double scale = 0.25;
const double line_distance_threshold = 20;
const double angle_threshold = 87.0 / 180.0 * CV_PI;                                                               // Only vertical line
const vector<Scalar> lane_colors = {Scalar(0, 0, 255), Scalar(0, 255, 0), Scalar(255, 0, 0), Scalar(255, 255, 0)}; // red, green, blue, yellow

args_t args = {
    .process_all_image = false,
    .line_length = 4,
    .line_width = 1,
    .image_name = "",
};

int main(int argc, char **argv)
{
    parse_args(argc, argv);
    if (args.process_all_image)
    {
        vector<string> image_names = get_all_images_in_dir();

        for (int i = 0; i < image_names.size(); i++)
        {
            cout << image_names[i] << endl;
            process_image(image_names[i], 1);
        }
    }
    else
    {
        process_image(args.image_name, 0);
    }
    return 0;
}

vector<string> get_all_images_in_dir()
{
    vector<string> image_names;
    DIR *dir;
    struct dirent *diread;
    if ((dir = opendir(image_path.c_str())) != nullptr)
    {
        while ((diread = readdir(dir)) != nullptr)
        {
            string im_name = string(diread->d_name);
            int pos = im_name.find("_color");
            if (pos != -1)
            {
                image_names.push_back(im_name.substr(0, pos));
            }
        }
        closedir(dir);
    }
    else
    {
        perror("opendir");
    }
    return image_names;
}

void process_image(string image_name, int waitKeyTimer)
{
    Mat image = imread(image_path + image_name + "_color_rect.png");
    blur(image, image, Size(5, 5));

    if (image.empty())
    {
        cout << "image not found" << endl;
        return;
    }
    imshow("Image", image);

#if DEBUG
    Mat image_ipm = ipm(image, ipm_width, ipm_height);
    imshow("image_ipm", image_ipm);
    waitKey(0);
#endif

    Mat binary_img = binarization(image);
    Mat img_ipm = ipm(binary_img, ipm_width, ipm_height);

#if DEBUG
    imshow("IPM", img_ipm);
    waitKey(0);
#endif

    img_ipm = clean_ipm_from_noise(img_ipm);
    vector<Vec4i> lines = get_all_lines_in_the_image(img_ipm);
    vector<vector<double>> lane_lines = get_four_lanes(lines, img_ipm.cols / 2, img_ipm.rows);
    draw_lanes_on_image(image, lane_lines);

    imshow("Image with lines", image);
    waitKey(waitKeyTimer);

    writeOutputFile(image_name, lane_lines);
}

bool parse_args(int argc, char **argv)
{
    int c;
    while ((c = getopt(argc, argv, "hi:w:e:a")) != -1)
        switch (c)
        {
        case 'i':
            args.image_name = optarg;
            break;
        case 'w':
            args.line_width = atoi(optarg);
        case 'e':
            args.line_length = atoi(optarg);
            break;
        case 'a':
            args.process_all_image = true;
            break;
        case 'h':
        default:
            std::cout << "usage: " << argv[0] << " -i <image_name> -w <line-width> -e <line-height>" << std::endl;
            std::cout << "exit:  type q" << std::endl
                      << std::endl;
            std::cout << "Allowed options:" << std::endl
                      << "   -h                       produce help message" << std::endl
                      << "   -i arg                   image name." << std::endl
                      << "   -w arg                   [optional] specify the line width" << std::endl
                      << "   -e arg                   [optional] specify the line height" << std::endl
                      << "   -a                       [optional] process all images in the directory" << std::endl;
            return false;
        }

    if (args.image_name == "" && args.process_all_image == false)
    {
        std::cout << "The image name must be specified!" << std::endl;
        std::cout << "usage: " << argv[0] << " -i <image_name>" << std::endl;
        return false;
    }
    return true;
}

Mat binarization(Mat im_color)
{
    Mat r_binary = get_r_binary(im_color);
    Mat s_binary = get_s_binary(im_color);
    Mat mag_binary = get_mag_binary(im_color);
    Mat dir_binary = get_dir_binary(im_color);
    Mat grad_x_bin = get_grad_bin(im_color, 1, 0);
    Mat grad_y_bin = get_grad_bin(im_color, 0, 1);

    Mat im_bin(im_color.rows, im_color.cols, CV_8UC1, Scalar(0));

    bitwise_and(mag_binary, dir_binary, im_bin);
    // bitwise_or(dir_binary, im_bin, im_bin);
    bitwise_or(im_bin, grad_x_bin, im_bin);
    bitwise_or(im_bin, s_binary, im_bin);
#if DEBUG
    imshow("grad_x_bin", grad_x_bin);
    imshow("grad_y_bin", grad_y_bin);
    imshow("dir_binary", dir_binary);
    imshow("r_binary", r_binary);
    imshow("s_binary", s_binary);
    imshow("im_bin", im_bin);
    waitKey(0);
#endif

    return im_bin;
}

Mat get_r_binary(Mat image)
{
    Mat BGR_channel[3];
    split(image, BGR_channel);
    Mat r_binary(image.rows, image.cols, CV_8UC1, Scalar(0));
    inRange(BGR_channel[2], 150, 255, r_binary);
    return r_binary;
}

Mat get_s_binary(Mat image)
{
    Mat HSV(image.rows, image.cols, CV_8UC3, Scalar(0, 0, 0));
    cvtColor(image, HSV, COLOR_BGR2HSV);
    Mat HSV_channel[3];
    split(HSV, HSV_channel);
    Mat s_binary(image.rows, image.cols, CV_8UC1, Scalar(0));
    inRange(HSV_channel[1], 100, 255, s_binary);

    return s_binary;
}

Mat get_grad_bin(Mat image, int x_order, int y_order, int min_threshold, int max_threshold)
{
    Mat im_gray(image.rows, image.cols, CV_8UC1, Scalar(0));
    cvtColor(image, im_gray, COLOR_BGR2GRAY);

    Mat sobel(im_gray.rows, im_gray.cols, CV_32FC1);
    Sobel(im_gray, sobel, CV_32FC1, x_order, y_order, 3);

    sobel.convertTo(sobel, CV_8UC1);
    Mat grad_bin(sobel.rows, sobel.cols, CV_8UC1, Scalar(0));

    double min, max;
    minMaxLoc(sobel, &min, &max);

    sobel = sobel / max * 255;

    inRange(sobel, min_threshold, max_threshold, grad_bin);

    return grad_bin;
}

Mat get_mag_binary(Mat image, int sobel_kernel, int min_threshold, int max_threshold)
{
    Mat im_gray(image.rows, image.cols, CV_8UC1, Scalar(0));
    cvtColor(image, im_gray, COLOR_BGR2GRAY);
    Mat sobel_x(im_gray.rows, im_gray.cols, CV_32FC1);
    Mat sobel_y(im_gray.rows, im_gray.cols, CV_32FC1);
    Sobel(im_gray, sobel_x, CV_32FC1, 1, 0, sobel_kernel);
    Sobel(im_gray, sobel_y, CV_32FC1, 0, 1, sobel_kernel);

    Mat mag(im_gray.rows, im_gray.cols, CV_32FC1);
    mag = abs(sobel_x) + abs(sobel_y);
    mag.convertTo(mag, CV_8UC1);
    inRange(mag, min_threshold, max_threshold, mag);
    return mag;
}

Mat get_dir_binary(Mat image, int sobel_kernel, double min_threshold, double max_threshold)
{
    Mat im_gray(image.rows, image.cols, CV_8UC1, Scalar(0));
    cvtColor(image, im_gray, COLOR_BGR2GRAY);
    Mat sobel_x(im_gray.rows, im_gray.cols, CV_32FC1);
    Mat sobel_y(im_gray.rows, im_gray.cols, CV_32FC1);
    Sobel(im_gray, sobel_x, CV_32FC1, 1, 0, sobel_kernel);
    Sobel(im_gray, sobel_y, CV_32FC1, 0, 1, sobel_kernel);

    Mat dir(im_gray.rows, im_gray.cols, CV_32FC1);
    phase(sobel_x, sobel_y, dir);

    dir.convertTo(dir, CV_8UC1);
    inRange(dir, min_threshold, max_threshold, dir);
    return dir;
}

Mat ipm(Mat image, int width, int height)
{
    Eigen::Matrix3d projection_matrix;
    projection_matrix << 1381.9564208984375, 0, 648.1651000976562,
        0, 1394.720703125, 369.2019348144531,
        0, 0, 1;

    Mat img_ipm(height, width, image.type(), Scalar(0, 0, 0));

    for (int x = -width / 2; x < width / 2; x++)
    {
        for (int y = 0; y < height; y++)
        {
            Eigen::Vector3d position;
            position << x * scale, 10, y * scale;
            Eigen::Vector3d image_pos;

            image_pos = projection_matrix * position;
            int u = image_pos(0) / image_pos(2);
            int v = image_pos(1) / image_pos(2);

            if (u >= 0 && u < image.cols && v >= 300 && v < image.rows)
            {
                if (image.type() == CV_8UC1)
                    img_ipm.at<uchar>(height - y, x + width / 2) = image.at<uchar>(v, u);
                else if (image.type() == CV_8UC3)
                    img_ipm.at<Vec3b>(height - y, x + width / 2) = image.at<Vec3b>(v, u);
            }
        }
    }

    return img_ipm;
}

Mat clean_ipm_from_noise(Mat ipm)
{
    Mat element;

#if DEBUG
    imshow("ipm with noise", ipm);
    waitKey(0);
#endif

    // Remove noise
    Mat ipm_noise(ipm.rows, ipm.cols, CV_8UC1, Scalar(0));
    element = getStructuringElement(MORPH_RECT, Size(30, 30));
    // To close the gap of the noise around the road
    morphologyEx(ipm, ipm_noise, MORPH_CLOSE, element);

#if DEBUG
    imshow("ipm noise", ipm_noise);
    waitKey(0);
#endif

    // Remove the line
    erode(ipm_noise, ipm_noise, element);

#if DEBUG
    imshow("ipm noise without lines", ipm_noise);
    waitKey(0);
#endif

    // Expaned the noise
    dilate(ipm_noise, ipm_noise, element);

#if DEBUG
    imshow("ipm noise with noise increase", ipm_noise);
    waitKey(0);
#endif

    // Remove the noise from the original image
    ipm = ipm - ipm_noise;

#if DEBUG
    imshow("ipm cleaned", ipm);
    waitKey(0);
#endif

    return ipm;
}

vector<Vec4i> get_all_lines_in_the_image(Mat ipm)
{
    vector<Vec4i> lines;
    // Each line is a vector of 4 elements, the first two are the start point and the last two are the end point.
    // Rho and theta are the resolution parameters. Each cell for theta is equal to a variation of 3.14/180. Each cell for rho is equal to a variation of 1 pixel.
    HoughLinesP(ipm, lines, 1, CV_PI / 180, 30, 60, 10);

    vector<Vec4i> lines_filtered;
    for (int i = 0; i < lines.size(); i++)
    {
        int width = lines[i][2] - lines[i][0];
        if (width != 0)
        {
            double slope = abs((lines[i][3] - lines[i][1])) / width;
            double angle = atan(slope);

            if (angle < angle_threshold)
            {
                continue;
            }
        }
        lines_filtered.push_back(lines[i]);
    }
    lines = lines_filtered;

#if DEBUG
    Mat all_line_image(ipm.rows, ipm.cols, CV_8UC3, Scalar(0, 0, 0));
    cout << "lines.size()=" << lines.size() << endl;
    for (size_t i = 0; i < lines.size(); i++)
    {
        line(all_line_image, Point(lines[i][0], lines[i][1]),
             Point(lines[i][2], lines[i][3]), Scalar(0, 0, 255), 3, 8);
    }
    imshow("Detected Lines", all_line_image);
    waitKey(0);
#endif

    return lines;
}

vector<vector<double>> get_four_lanes(vector<Vec4i> lines, int x_scale, int y_scale)
{

    vector<int> labels;
    int numberOfLines = cv::partition(lines, labels, areSameLane);

    vector<vector<double>> all_lanes(numberOfLines, vector<double>(417, 0)); // pixel from 300 to 717
    vector<vector<int>> counter(numberOfLines, vector<int>(417, 0));

    vector<int> lanes_x_coordinate_in_ipm(numberOfLines, 0);
    vector<int> lanes_x_coordinate_counter(numberOfLines, 0);

    // Calculate the lane lines in the original image
    for (int i = 0; i < lines.size(); i++)
    {
        int label_number = labels[i];

        Vec4i new_points = get_lines_coordinates_from_ipm(lines[i], x_scale, y_scale);

        bool is_line_used = calc_line_points(new_points, all_lanes[label_number], counter[label_number]);
        if (is_line_used)
        {
            lanes_x_coordinate_in_ipm[label_number] += (lines[i][0] + lines[i][2]) / 2;
            lanes_x_coordinate_counter[label_number]++;
        }
    }

    vector<lane_t> right_lanes; // Ordered as r0,r1,r2
    vector<lane_t> left_lanes;  // Ordered as l2,l1,l0

    // Group all lanes of the same type together
    for (int i = 0; i < numberOfLines; i++)
    {
        for (int j = 0; j < 417; j++)
        {
            if (counter[i][j] > 0)
            {
                all_lanes[i][j] /= counter[i][j];
            }
        }

        double x_coordinate = lanes_x_coordinate_in_ipm[i] / lanes_x_coordinate_counter[i];

        if (x_coordinate > ipm_width / 2)
        {
            add_lanes_to_position_vector(right_lanes, x_coordinate, i);
        }
        else
        {
            add_lanes_to_position_vector(left_lanes, x_coordinate, i);
        }
    }

#if DEBUG
    cout << "right_lanes.size()=" << right_lanes.size() << endl;
    cout << "left_lanes.size()=" << left_lanes.size() << endl;
#endif

    vector<vector<double>> filtered_lanes = filter_lanes_by_position(right_lanes, left_lanes, all_lanes);

#if DEBUG
    // Draw lines with different colors
    Mat all_line_image(ipm_height, ipm_width, CV_8UC3, Scalar(0, 0, 0));
    for (size_t i = 0; i < lines.size(); i++)
    {
        line(all_line_image, Point(lines[i][0], lines[i][1]),
             Point(lines[i][2], lines[i][3]), lane_colors[labels[i]], 3, 8);
    }
    imshow("Detected grouped lines", all_line_image);
    waitKey(0);
#endif
    return filtered_lanes;
}

bool areSameLane(const Vec4i &_l1, const Vec4i &_l2)
{
    point_t start_l0 = {(double)_l1[0], (double)_l1[1]};
    point_t end_l0 = {(double)_l1[2], (double)_l1[3]};

    point_t start_l1 = {(double)_l2[0], (double)_l2[1]};
    point_t end_l1 = {(double)_l2[2], (double)_l2[3]};

    double slope_l0 = (end_l0.y - start_l0.y) / (end_l0.x - start_l0.x);
    double slope_l1 = (end_l1.y - start_l1.y) / (end_l1.x - start_l1.x);

    if (abs(end_l0.x - end_l1.x) < line_distance_threshold && abs(start_l0.x - start_l1.x) < line_distance_threshold)
    {
        return true;
    }
    else
    {
        return false;
    }
}

Vec4i get_lines_coordinates_from_ipm(Vec4i lines, int x_scale, int y_scale)
{
    Eigen::Matrix3d projection_matrix;
    projection_matrix << 1381.9564208984375, 0, 648.1651000976562,
        0, 1394.720703125, 369.2019348144531,
        0, 0, 1;

    Eigen::Vector3d position0;
    position0 << (lines[0] - x_scale) * scale, 10, (y_scale - lines[1]) * scale;

    Eigen::Vector3d position1;
    position1 << (lines[2] - x_scale) * scale, 10, (y_scale - lines[3]) * scale;

    Eigen::Vector3d image_pos0;
    image_pos0 = projection_matrix * position0;

    Eigen::Vector3d image_pos1;
    image_pos1 = projection_matrix * position1;

    Vec4i new_points;

    new_points[0] = image_pos0(0) / image_pos0(2);
    new_points[1] = image_pos0(1) / image_pos0(2);

    new_points[2] = image_pos1(0) / image_pos1(2);
    new_points[3] = image_pos1(1) / image_pos1(2);

    return new_points;
}

bool calc_line_points(Vec4i points, vector<double> &lanes, vector<int> &counter)
{
    // This point is the nearest from the camera
    point_t start = {(double)points[0], (double)points[1]};

    // This point is the farthest from the camera
    point_t end = {(double)points[2], (double)points[3]};

    if (end.y < start.y)
    {
        point_t tmp = start;
        start = end;
        end = tmp;
    }

#if DEBUG
    cout << "start.x=" << start.x << endl;
    cout << "start.y=" << start.y << endl;
    cout << "end.x=" << end.x << endl;
    cout << "end.y=" << end.y << endl;
#endif

    if (start.x == end.x || start.y == end.y)
    { // Vertical line or horizontal line
        return false;
    }
    if (end.y - start.y <= 2)
    { // Line too short
        return false;
    }

    double height = end.y - start.y;
    double slope = (end.x - start.x) / height;

    if (end.y >= 717)
    {
        end.y = 717;
    }
    if (start.y <= 300)
    {
        start.y = 300;
    }

    for (int y = start.y; y < end.y; y++)
    {
        double x = start.x + slope * (y - start.y);
        lanes[y - 300] += x;
        counter[y - 300]++;
    }
    return true;
}

void add_lanes_to_position_vector(vector<lane_t> &position_vector, double x_coordinate, int lane_index)
{
    for (size_t i = 0; i < position_vector.size(); i++)
    {
        // Insert the lane in order. The lanes are sorted by their slope
        if (position_vector[i].x_coordinate >= x_coordinate)
        {
            position_vector.insert(position_vector.begin() + i, {x_coordinate, lane_index});
            return;
        }
    }
    position_vector.push_back({x_coordinate, lane_index});
}

vector<vector<double>> filter_lanes_by_position(vector<lane_t> right_lanes, vector<lane_t> left_lanes, vector<vector<double>> all_lanes)
{
    vector<vector<double>> filtered_lanes(4, vector<double>(417, -1)); // l1, l0, r0, r1
    // If the lane is the closest right lane to the car, then the slope is a big positive number.
    // Distant right lanes have a smaller positive slope compared to the closer one.

    for (int i = 0; i < right_lanes.size() && i < 2; i++)
    {
        // right_lanes= r0,r1,r2
        int lane_index = right_lanes[i].lane_index;

        // Add right lane inside the third (r0) and fourth(r1) position
        filtered_lanes[3 + i] = all_lanes[lane_index];
    }

    int lane_index = -1;
    if (left_lanes.size() >= 2) // We have l2,l1,l0
    {
        for (int i = 1; i >= 0; i--)
        {
            // If i=1, then get the last element, if i=0 then gest the second last element
            lane_index = left_lanes[left_lanes.size() - 2 + i].lane_index;
            // Add left lane inside the first (l1) and second (l0) position
            filtered_lanes[i] = all_lanes[lane_index];
        }
    }
    else
    {
        // To solve the problem of having just one left lane (l0)
        lane_index = left_lanes[left_lanes.size() - 1].lane_index;
        // Add only l0
        filtered_lanes[1] = all_lanes[lane_index];
    }

    return filtered_lanes;
}

void draw_lanes_on_image(Mat image, vector<vector<double>> lanes)
{
    for (int i = 0; i < lanes.size(); i++)
    {
        for (int y = 0; y < lanes[i].size(); y++)
        {
            if (lanes[i][y] == 0)
                continue;

            int x = lanes[i][y];
            if (x < 0 || x >= image.cols)
                continue;

            circle(image, Point(x, y + 300), 1, lane_colors[i], 4, 8, 0);
        }
    }
}

void writeOutputFile(string image_name, vector<vector<double>> lanes)
{
    ofstream output_file;
    output_file.open(output_path + image_name);
    for (int i = 0; i < lanes.size(); i++)
    {
        for (int y = 0; y < lanes[i].size(); y++)
        {
            if (lanes[i][y] == 0)
            {
                output_file << "-1"
                            << " ";
                continue;
            }
            output_file << lanes[i][y] << " ";
        }
        output_file << endl;
    }
    output_file.close();
}