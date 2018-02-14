namespace cc {
class fully_connected_layer : public layer_t {
public:
    vec_t weight;
    vec_t bias;
    vec_t grad_weight;
    vec_t grad_bias;

    fully_connected_layer(const size_t input_channel,
                          const size_t output_channel)
        : layer_t(input_channel, output_channel) {
        weight.resize(input_channel * output_channel);
        grad_weight.resize(input_channel * output_channel);

        bias.resize(output_channel);
        grad_bias.resize(output_channel);

        const float xavier_weight = std::sqrt(3.0 / (input_channel * output_channel));
        uniform_rand(weight.begin(), weight.end(), -xavier_weight, xavier_weight);
        uniform_rand(bias.begin(), bias.end(), -xavier_weight, xavier_weight);
    }

    vec_t fp(const vec_t &input, size_t batch_size) override {
        this->input = input;
        output.clear();
        output.resize(batch_size * output_channel);

        //
        // multiply weight
        //
        gemm('C',
             'T',
             'N',
             output_channel,        // m A(m,k) B(k,n) C(m,n)
             batch_size,            // n
             input_channel,         // k
             1.0,                   // alpha
             &weight[0],            // matrix A
             input_channel,         // lda
             &input[0],             // matrix B
             input_channel,         // ldb
             0.0,                   // beta
             &output[0],            // matrix C
             output_channel);       // ldc

        //
        // add bias
        //
        vec_t onevec(batch_size);
        std::fill(onevec.begin(), onevec.end(), 1.0);

        gemm('C',
             'N',
             'N',
             output_channel,        // m A(m,k) B(k,n) C(m,n)
             batch_size,            // n
             1,                     // k
             1.0,                   // alpha
             &bias[0],              // matrix A
             output_channel,        // lda
             &onevec[0],            // matrix B
             1,                     // ldb
             1.0,                   // beta
             &output[0],            // matrix C
             output_channel);       // ldc

        if (next_layer != nullptr) {
            return next_layer->fp(output, batch_size);
        }
        return output;
    }

    void bp(const vec_t &delta, size_t batch_size) override {
        //
        // calc grad_weight
        //
        gemm('C',
             'N',
             'T',
             input_channel,          // m A(m,k) B(k,n) C(m,n)
             output_channel,         // n
             batch_size,             // k
             1.0 / batch_size,       // alpha
             &input[0],              // matrix A
             input_channel,          // lda
             &delta[0],              // matrix B
             output_channel,         // ldb
             0.0,                    // beta
             &grad_weight[0],        // matrix C
             input_channel);         // ldc

        //
        // calc grad_bias
        //
        vec_t onevec(batch_size);
        std::fill(onevec.begin(), onevec.end(), 1.0);

        gemv('N',
             output_channel,
             batch_size,
             1.0 / batch_size,
             &delta[0],
             output_channel,
             &onevec[0],
             1,
             0.0,
             &grad_bias[0],
             1);

        //
        // propagate delta
        //
        if (prev_layer.lock() != nullptr) {
            this->delta.clear();
            this->delta.resize(batch_size * input_channel);

            gemm('C',
                 'N',
                 'N',
                 input_channel,         // m A(m,k) B(k,n) C(m,n)
                 batch_size,            // n
                 output_channel,        // k
                 1.0,                   // alpha
                 &weight[0],            // matrix A
                 input_channel,         // lda
                 &delta[0],             // matrix B
                 output_channel,        // ldb
                 0.0,                   // beta
                 &this->delta[0],       // matrix C
                 input_channel);        // ldc
            prev_layer.lock()->bp(this->delta, batch_size);
        }
    }

    void update(optimizer &o) override {
        o.update(grad_weight, weight);
        o.update(grad_bias, bias);

        if (next_layer != nullptr) {
            next_layer->update(o);
        }
    }

};
}
