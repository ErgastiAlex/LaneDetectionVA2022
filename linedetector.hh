
using namespace cv;
using namespace std;

typedef struct
{
    bool process_all_image;
    int line_length;
    int line_width;
    string image_name;
} args_t;

typedef struct
{
    double x;
    double y;
} point_t;

typedef struct
{
    double x_coordinate;
    int lane_index;
} lane_t;

bool parse_args(int argc, char **argv);
vector<string> get_all_images_in_dir();
void process_image(string image_name, int waitKeyTimer);
Mat ipm(Mat image, int width, int height);
void draw_lines_from_ipm(Mat image, Mat ipm_with_lines);
Mat binarization(Mat im_color);
Mat get_r_binary(Mat image);
Mat get_s_binary(Mat image);
Mat get_grad_bin(Mat image, int x_order, int y_order, int min_threshold = 50, int max_threshold = 255);
Mat get_mag_binary(Mat image, int sobel_kernel = 3, int min_threshold = 50, int max_threshold = 255);
Mat get_dir_binary(Mat image, int sobel_kernel = 3, double min_threshold = 1, double max_threshold = CV_PI / 2);
Mat clean_ipm_from_noise(Mat ipm);
vector<Vec4i> get_all_lines_in_the_image(Mat ipm);
bool areSameLane(const Vec4i &_l1, const Vec4i &_l2);
Vec4i get_lines_coordinates_from_ipm(Vec4i lines, int x_scale, int y_scale);
bool calc_line_points(Vec4i points, vector<double> &lanes, vector<int> &counter);
void add_lanes_to_position_vector(vector<lane_t> &position_vector, double x_coordinate, int lane_index);
// Return l1,l0,r0,r1 lines in the image
vector<vector<double>> get_four_lanes(vector<Vec4i> lines, int x_scale, int y_scale);
vector<vector<double>> filter_lanes_by_position(vector<lane_t> right_lanes, vector<lane_t> left_lanes, vector<vector<double>> all_lanes);
void draw_lanes_on_image(Mat image, vector<vector<double>> lanes);
void writeOutputFile(string image_name, vector<vector<double>> lanes);
