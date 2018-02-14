/**************************************************
 * Tune
 **************************************************/
#define K_MINIBATCH_SIZE   1
#define K_EPOCHES          5
/**************************************************/

#include "util.hpp"
#include "framework.hpp"

using namespace cc;
using namespace cc::activation;

int main(int argc, char *argv[]) {
    network_t nn;

    //
    // lenet
    //
    nn  << convolutional_layer(28, 28, 5, 1, 6)    // input=28x28x6, window=5, output=24x24x6, cnhw
        << pooling_layer(24, 24, 2, 6)             // input=24x24x6, window=2, output=12x12x6, cnhw
        << convolutional_layer(12, 12, 5, 6, 10)   // input=12x12x6, window=5, output=8x8x10,  cnhw
        << pooling_layer(8, 8, 2, 10)              // input=8x8x10,  window=2, output=4x4x10,  cnhw
        << relocation_layer(4, 4, 10)              // input=4x4x10,            output=4x4x10,  cnhw<->nchw
        << fully_connected_layer(4 * 4 * 10, 100)  // input=4x4x10,            output=100,     nchw
        << activation_layer<relu>(100)             // input=100,               output=100,     nchw
        << fully_connected_layer(100, 10)          // input=100,               output=10,      nchw
        << softmax_layer(10);                      // input=10,                output=10,      nchw

    //
    // train phase
    //
    std::vector<vec_t>   train_images;
    std::vector<label_t> train_labels;
    parse_mnist("./data/train-images-idx3-ubyte", train_images,
                "./data/train-labels-idx1-ubyte", train_labels);

    train_images.resize(10000);
    train_labels.resize(10000);

    gradient_descent optimizer;
    optimizer.alpha *= std::min(4.0f, static_cast<float_t>(std::sqrt(K_MINIBATCH_SIZE)));

    auto on_minibatch_enumerate = [&]() {};
    auto on_epoch_enumerate     = [&]() {};

    timer t;
    nn.train<cross_entropy_multiclass>(optimizer, train_images, train_labels, K_MINIBATCH_SIZE, K_EPOCHES, on_minibatch_enumerate, on_epoch_enumerate);
    t.lap("train total");

    //
    // inference phase
    //
    std::vector<vec_t>   test_images;
    std::vector<label_t> test_labels;
    parse_mnist("./data/t10k-images-idx3-ubyte", test_images,
                "./data/t10k-labels-idx1-ubyte", test_labels);

    nn.inference(test_images, test_labels, K_MINIBATCH_SIZE);
    t.lap("inference total");
}

