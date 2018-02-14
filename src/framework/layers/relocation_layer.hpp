namespace cc {

class relocation_layer : public layer_t {
public:
    size_t width;
    size_t height;
    size_t hw;
    size_t hh;

    relocation_layer(const size_t width,
                     const size_t height,
                     const size_t channel)
        : layer_t(channel, channel),
          width(width),
          height(height) {
    }

    vec_t fp(const vec_t &input, size_t batch_size) override {
        this->input = input;
        output.clear();
        output.resize(batch_size * output_channel * height * width);

        const size_t N = batch_size;
        const size_t C = input_channel;
        const size_t H = height;
        const size_t W = width;

        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        const size_t i_i = w +
                                           h * W +
                                           n * W * H +
                                           c * W * H * N;
                        const size_t o_i = w +
                                           h * W +
                                           c * W * H +
                                           n * W * H * C;
                        output[o_i] = input[i_i];
                    }
                }
            }
        }

        if (next_layer.get() != nullptr) {
            return next_layer->fp(output, batch_size);
        }
        return output;
    }

    void bp(const vec_t &delta, size_t batch_size) override {
        if (prev_layer.lock() != nullptr) {
            this->delta.clear();
            this->delta.resize(batch_size * input_channel * height * width);

            const size_t N = batch_size;
            const size_t C = input_channel;
            const size_t H = height;
            const size_t W = width;

            for (size_t n = 0; n < N; n++) {
                for (size_t c = 0; c < C; c++) {
                    for (size_t h = 0; h < H; h++) {
                        for (size_t w = 0; w < W; w++) {
                            const size_t i_i = w +
                                               h * W +
                                               c * W * H +
                                               n * W * H * C;
                            const size_t o_i = w +
                                               h * W +
                                               n * W * H +
                                               c * W * H * N;
                            this->delta[o_i] = delta[i_i];
                        }
                    }
                }
            }

            prev_layer.lock()->bp(this->delta, batch_size);
        }
    }

    void update(optimizer &o) override {
        if (next_layer != nullptr) {
            next_layer->update(o);
        }
    }
};

}
