#pragma once

namespace cc {

void parse_mnist(const char *fn_images, std::vector<vec_t>   &images,
                 const char *fn_labels, std::vector<label_t> &labels) {
    size_t w = 28;
    size_t h = 28;

    const size_t train_size = read_ubyte_dataset(fn_images, fn_labels, w, h);

    std::vector<uint8_t> images_ubyte(train_size * h * w);
    std::vector<uint8_t> labels_ubyte(train_size);

    read_ubyte_dataset(fn_images, fn_labels, w, h, &images_ubyte[0], &labels_ubyte[0]);

    for (size_t n = 0; n < train_size; n++) {
        vec_t image(h * w);
        for (size_t i = 0; i < h * w; i++) {
            image[i] = images_ubyte[i + n * h * w] / 255.0f;
        }
        label_t label = labels_ubyte[n];

        images.push_back(image);
        labels.push_back(label);
    }
}
} // namespace
