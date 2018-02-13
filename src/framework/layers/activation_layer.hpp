namespace di {

template<typename Activation>
class activation_layer : public layer_t {
public:
    Activation activ;

    activation_layer(const size_t channel)
        : layer_t(channel, channel) {
    }

    vec_t fp(const vec_t &input, size_t batch_size) override {
        this->input = input;
        output.clear();
        output.resize(batch_size * output_channel);

        for (size_t i = 0; i < output.size(); i++) {
            output[i] = activ.f(input[i]);
        }

        if (next_layer != nullptr) {
            return next_layer->fp(output, batch_size);
        }
        return output;
    }

    void bp(const vec_t &delta, size_t batch_size) override {
        if (prev_layer.lock() != nullptr) {
            this->delta.clear();
            this->delta.resize(batch_size * input_channel);

            for (size_t i = 0; i < delta.size(); i++) {
                this->delta[i] = delta[i] * activ.df(output[i]);
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

