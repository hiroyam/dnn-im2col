
namespace cc {
class layer_t {
public:
    std::shared_ptr<layer_t> next_layer;
    std::weak_ptr<layer_t>   prev_layer;

    size_t nth_layer;
    size_t input_channel;
    size_t output_channel;

    vec_t input;
    vec_t output;
    vec_t delta;

    layer_t(const size_t input_channel,
            const size_t output_channel)
        : input_channel(input_channel),
          output_channel(output_channel) {
    }

    virtual vec_t fp(const vec_t &input, size_t batch_size) = 0;
    virtual void  bp(const vec_t &delta, size_t batch_size) = 0;
    virtual void  update(optimizer &o)                      = 0;
};
}
