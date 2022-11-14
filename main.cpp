#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#define kIMAGE_SIZE 28
#define kCHANNELS 1

using namespace cv;
using namespace std;


int main(int argc, const char **argv){
    if (argc != 3) {
        std::cerr << "Usage: classifier <path-to-exported-script-module> "
        << "<path-to-lable-file>"
        << std::endl;
        return -1;
    }

    // ------------------------------------------------------------------------------------------------------------------------ //
    // ------------------------------------------------------------------------------------------------------------------------ //
    // Load the model: AlexNet:
    torch::jit::script::Module model = torch::jit::load( argv[1]);


    // ------------------------------------------------------------------------------------------------------------------------ //
    // ------------------------------------------------------------------------------------------------------------------------ //
    // Load Image:
    cv::Mat image = cv::imread( argv[2]);  // CV_8UC3
    if (image.empty() || !image.data) {
        cout << "Can't load or open the image" << endl;
        return -1;
    }

    cv::cvtColor( image, image, cv::COLOR_BGR2GRAY);

    // scale image to fit
    cv::Size scale( kIMAGE_SIZE, kIMAGE_SIZE);
    cv::resize( image, image, scale);

    // convert [unsigned int] to [float]
    image.convertTo( image, CV_32FC3, 1.0f / 255.0f);


    // ------------------------------------------------------------------------------------------------------------------------ //
    // ------------------------------------------------------------------------------------------------------------------------ //
    // Inference phase:
    try{
        auto input_tensor = torch::from_blob( image.data, {1, kIMAGE_SIZE, kIMAGE_SIZE, kCHANNELS});     // , torch::kFloat32);
        input_tensor = input_tensor.permute({0, 3, 1, 2});
        // input_tensor[0][0] = input_tensor[0][0].div_( 5).sub_( 0.5).div_( 0.25);

        input_tensor = input_tensor.contiguous().view( {-1, 1, 28, 28} );
        torch::Tensor out_tensor = model.forward( {input_tensor} ).toTensor();

        int pred = torch::argmax( out_tensor[0]).item<int>();
        std::cout << "The Predicted Label for the Image '" << argv[2] <<  "' is :: " << pred << "\n\n";
    }
    catch (const c10::Error& e) {
        std::cerr << " ecountered error in the inference phase \n";
        return -1;
    }

    return 0;
}