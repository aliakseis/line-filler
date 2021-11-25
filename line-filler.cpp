// line-filler.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"


#include <iostream>
#include <optional>

auto get_ball_structuring_element(int radius) {
    /* Get a ball shape structuring element with specific radius for morphology operation.
        The radius of ball usually equals to(leaking_gap_size / 2).

        # Arguments
        radius : radius of ball shape.

        # Returns
        an array of ball structuring element.
    */
    return cv::getStructuringElement(cv::MORPH_ELLIPSE, { 2 * radius + 1, 2 * radius + 1 });
}

/*
def get_unfilled_point(image):
    """Get points belong to unfilled(value==255) area.

    # Arguments
        image: an image.

    # Returns
        an array of points.
    """
    y, x = np.where(image == 255)

    return np.stack((x.astype(int), y.astype(int)), axis=-1)
*/

std::optional<cv::Point> get_unfilled_point(const cv::Mat& image) {
    for (int y = 0; y < image.rows; ++y)
        for (int x = 0; x < image.cols; ++x)
            if (image.at<uchar>(y, x) == 255)
            {
                return std::optional<cv::Point>({ x, y });
            }
    return {};
}

auto exclude_area(const cv::Mat& image, int radius) {
    /*Perform erosion on image to exclude points near the boundary.
        We want to pick part using floodfill from the seed point after dilation.
        When the seed point is near boundary, it might not stay in the fill, and would
        not be a valid point for next floodfill operation.So we ignore these points with erosion.

        # Arguments
        image : an image.
        radius : radius of ball shape.

        # Returns
        an image after dilation.
        */
    cv::Mat result;
    cv::morphologyEx(image, result, cv::MORPH_ERODE, get_ball_structuring_element(radius), { -1, -1 }, 1);
    return result;
}

auto trapped_ball_fill_single(const cv::Mat& image, const cv::Point& seed_point, int radius) {
    /*Perform a single trapped ball fill operation.

        # Arguments
        image : an image.the image should consist of white background, black lines and black fills.
        the white area is unfilled area, and the black area is filled area.
        seed_point : seed point for trapped - ball fill, a tuple(integer, integer).
        radius : radius of ball shape.
        # Returns
        an image after filling.
        */
    auto ball = get_ball_structuring_element(radius);

        //pass1 = np.full(image.shape, 255, np.uint8)
        //pass2 = np.full(image.shape, 255, np.uint8)

    cv::Mat im_inv;
    cv::bitwise_not(image, im_inv);

        // Floodfill the image
    cv::Mat mask1;
    cv::copyMakeBorder(im_inv, mask1, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);
        //_, pass1, _, _ = 
    cv::Mat pass1(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
    cv::floodFill(pass1, mask1, seed_point, 0);// , 0, 0, 4);

        // Perform dilation on image.The fill areas between gaps became disconnected.
    cv::morphologyEx(pass1, pass1, cv::MORPH_DILATE, ball, { -1, -1 }, 1);
    cv::Mat mask2;
    cv::copyMakeBorder(pass1, mask2, 1, 1, 1, 1, cv::BORDER_CONSTANT, 0);

        // Floodfill with seed point again to select one fill area.
        //_, pass2, _, rect = 
    cv::Mat pass2(image.rows, image.cols, CV_8UC1, cv::Scalar(255));
    cv::Rect rect;
    cv::floodFill(pass2, mask2, seed_point, 0, &rect);// 0, 0, 4)
        // Perform erosion on the fill result leaking - proof fill.
    cv::morphologyEx(pass2, pass2, cv::MORPH_ERODE, ball, { -1, -1 }, 1);

    return pass2;
}

auto trapped_ball_fill_multi(const cv::Mat& image, int radius, bool mean = true, int max_iter = 1000) {
    /*Perform multi trapped ball fill operations until all valid areas are filled.

        # Arguments
        image : an image.The image should consist of white background, black lines and black fills.
        the white area is unfilled area, and the black area is filled area.
        radius : radius of ball shape.
        method : method for filtering the fills.
        'max' is usually with large radius for select large area such as background.
        max_iter : max iteration number.
        # Returns
        an array of fills' points.
    */
    //    print('trapped-ball ' + str(radius))

    cv::Mat unfill_area;
    image.copyTo(unfill_area);
        //filled_area, filled_area_size, result = [], [], []

    std::vector<std::vector<cv::Point>> filled_area;

    for (int i = 0; i < max_iter; ++i) {
        //points = get_unfilled_point(exclude_area(unfill_area, radius))
        //    if not len(points) > 0:
        //break
        auto point = get_unfilled_point(exclude_area(unfill_area, radius));
        if (!point)
            break;

        auto fill = trapped_ball_fill_single(unfill_area, *point, radius);
        cv::bitwise_and(unfill_area, fill, unfill_area);

            //filled_area.append(np.where(fill == 0))
            //filled_area_size.append(len(np.where(fill == 0)[0]))
        fill = fill == 0;
        std::vector<cv::Point> locations;   // output, locations of non-zero pixels
        cv::findNonZero(fill, locations);

        filled_area.push_back(std::move(locations));
    }

    /*
        filled_area_size = np.asarray(filled_area_size)

        if method == 'max':
    area_size_filter = np.max(filled_area_size)
        elif method == 'median' :
        area_size_filter = np.median(filled_area_size)
        elif method == 'mean' :
        area_size_filter = np.mean(filled_area_size)
        else:
    area_size_filter = 0

        result_idx = np.where(filled_area_size >= area_size_filter)[0]

        for i in result_idx :
    result.append(filled_area[i])

        return result
        */

    return filled_area;
}


int main(int argc, char** argv)
{
    if (argc < 2)
        return 1;

    try {
        auto img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

        cv::Mat binary;
        cv::threshold(img, binary, 220, 255, cv::THRESH_BINARY);

        auto fill = trapped_ball_fill_multi(binary, 3);

        cv::Mat res = cv::Mat::zeros(img.rows, img.cols, CV_8UC3);

        cv::RNG rng(215526);
        for (const auto& area : fill)
        {
            auto color = cv::Vec3b(rng.uniform(30, 255), rng.uniform(30, 255), rng.uniform(30, 255));
            for (auto& p : area)
                res.at<cv::Vec3b>(p) = color;
        }

        cv::imshow("result", res);
        cv::waitKey();
    }
    catch (const std::exception& ex) {
        std::cerr << typeid(ex).name() << ": " << ex.what() << '\n';
    }
}
