namespace cc {

class softmax_layer : public layer_t {
public:
    softmax_layer(const size_t channel)
        : layer_t(channel, channel) {
    }

    vec_t fp(const vec_t &input, size_t batch_size) override {
        this->input = input;
        output.clear();
        output.resize(batch_size * output_channel);

        float_t save_exp_overflow = *std::max_element(input.begin(), input.end());

        for (size_t n = 0; n < batch_size; n++) {
            float_t denom = 0.0;
            for (size_t c = 0; c < input_channel; c++) {
                denom += std::exp(input[c + n * input_channel] - save_exp_overflow);
            }
            for (size_t c = 0; c < input_channel; c++) {
                output[c + n * input_channel] = std::exp(input[c + n * input_channel] - save_exp_overflow) / denom;
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
            this->delta.resize(batch_size * input_channel);

            for (size_t n = 0; n < batch_size; n++) {
                for (size_t c = 0; c < input_channel; c++) {
                    for (size_t k = 0; k < output_channel; k++) {
                        size_t  c_i   = c + n * input_channel;
                        size_t  k_i   = k + n * output_channel;
                        float_t dy_da = (c_i == k_i) ? (output[k_i] * (1.0 - input[c_i])) : (-output[k_i] * input[c_i]);
                        this->delta[c_i] += delta[k_i] * dy_da;
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
