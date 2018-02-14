namespace cc {

class pooling_layer : public layer_t {
public:
    size_t window;
    size_t in_width;
    size_t in_height;
    size_t hw;
    size_t hh;

    pooling_layer(const size_t in_width,
                  const size_t in_height,
                  const size_t window,
                  const size_t channel)
        : layer_t(channel, channel),
          window(window),
          in_width(in_width),
          in_height(in_height) {
        if (window != 2) {
            throw std::runtime_error("not implemented");
        }
        hw = in_width / window;
        hh = in_height / window;
    }

    vec_t fp(const vec_t &input, size_t batch_size) override {
        this->input = input;

        const size_t N = batch_size;
        const size_t C = output_channel;
        const size_t H = in_height;
        const size_t W = in_width;

        output.resize(N * C * hh * hw);
        std::fill(output.begin(), output.end(), -1e10);

        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < H; h++) {
                    for (size_t w = 0; w < W; w++) {
                        const size_t i_i = w +
                                           h * W +
                                           n * W * H +
                                           c * W * H * N;
                        const size_t o_i = (w / window) +
                                           (h / window) * hw +
                                           n            * hw * hh +
                                           c            * hw * hh * N;
                        output[o_i] = std::max(output[o_i], input[i_i]);
                    }
                }
            }
        }

        if (next_layer != nullptr) {
            return next_layer->fp(output, batch_size);
        }
        return output;
    }

    void bp(const vec_t &delta, size_t batch_size) override {
        const size_t N = batch_size;
        const size_t C = output_channel;
        const size_t H = in_height;
        const size_t W = in_width;

        this->delta.clear();
        this->delta.resize(N * C * H * W);

        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t h = 0; h < hh; h++) {
                    for (size_t w = 0; w < hw; w++) {
                        for (size_t s = 0; s < window; s++) {
                            for (size_t r = 0; r < window; r++) {
                                const size_t x_i = (r + w * window) +
                                                   (s + h * window) * W +
                                                   n                * W * H +
                                                   c                * W * H * N;
                                const size_t y_i = w +
                                                   h * hw +
                                                   n * hw * hh +
                                                   c * hw * hh * N;
                                if (input[x_i] == output[y_i]) {
                                    this->delta[x_i] = delta[y_i];
                                    goto next;
                                }
                            }
                        }
next:
                        nop();
                    }
                }
            }
        }

        if (prev_layer.lock() != nullptr) {
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
