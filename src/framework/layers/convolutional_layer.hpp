namespace cc {
class convolutional_layer : public layer_t {
public:
    size_t in_width;
    size_t in_height;
    size_t window_size;
    size_t pad;
    size_t stride;
    size_t out_width;
    size_t out_height;

    vec_t weight;
    vec_t bias;
    vec_t grad_weight;
    vec_t grad_bias;

    convolutional_layer(const size_t in_width,
                        const size_t in_height,
                        const size_t window_size,
                        const size_t input_channel,
                        const size_t output_channel)
        : layer_t(input_channel, output_channel),
        in_width(in_width),
        in_height(in_height),
        window_size(window_size) {
        pad        = 0;
        stride     = 1;
        out_width  = 1 + (in_width + 2 * pad - window_size) / stride;
        out_height = 1 + (in_height + 2 * pad - window_size) / stride;

        weight.resize(window_size * window_size * input_channel * output_channel);
        grad_weight.resize(window_size * window_size * input_channel * output_channel);

        bias.resize(output_channel);
        grad_bias.resize(output_channel);

        const float xavier_weight = sqrt(3.0f / (window_size * window_size * input_channel));
        uniform_rand(weight.begin(), weight.end(), -xavier_weight, xavier_weight);
        uniform_rand(bias.begin(), bias.end(), -xavier_weight, xavier_weight);
    }

    vec_t fp(const vec_t &input, size_t batch_size) override {
        const size_t N = batch_size;
        const size_t W = in_width;
        const size_t H = in_height;
        const size_t C = input_channel;
        const size_t K = output_channel;
        const size_t R = window_size;
        const size_t S = window_size;
        const size_t Q = out_height;
        const size_t P = out_width;

        const size_t k   = K;
        const size_t crs = C * R * S;
        const size_t npq = N * P * Q;

        //
        // convolution
        //
        this->input.resize(crs * npq);
        output.clear();
        output.resize(batch_size * output_channel * out_height * out_width);

        // im2col
        for (size_t n = 0; n < N; n++) {
            for (size_t c = 0; c < C; c++) {
                for (size_t s = 0; s < S; s++) {
                    for (size_t r = 0; r < R; r++) {
                        for (size_t q = 0; q < Q; q++) {
                            for (size_t p = 0; p < P; p++) {
                                const size_t i_i = (r + p) +
                                                   (s + q) * W +
                                                   n       * W * H +
                                                   c       * W * H * N;
                                const size_t o_i = p +
                                                   q * P +
                                                   n * P * Q +
                                                   r * P * Q * N +
                                                   s * P * Q * N * R +
                                                   c * P * Q * N * R * S;
                                this->input[o_i] = input[i_i];
                            }
                        }
                    }
                }
            }
        }

        gemm('R',
             'N',
             'N',
             k,                     // m A(m,k) B(k,n) C(m,n)
             npq,                   // n
             crs,                   // k
             1.0,                   // alpha
             &weight[0],            // matrix A
             crs,                   // lda
             &this->input[0],       // matrix B
             npq,                   // ldb
             0.0,                   // beta
             &output[0],            // matrix C
             npq);                  // ldc

        //
        // add bias
        //
        vec_t onevec(N * P * Q);
        std::fill(onevec.begin(), onevec.end(), 1.0);

        gemm('R',
             'N',
             'N',
             k,                // m A(m,k) B(k,n) C(m,n)
             npq,              // n
             1,                // k
             1.0,              // alpha
             &bias[0],         // matrix A
             1,                // lda
             &onevec[0],       // matrix B
             npq,              // ldb
             1.0,              // beta
             &output[0],       // matrix C
             npq);             // ldc

        if (next_layer != nullptr) {
            return next_layer->fp(output, batch_size);
        }
        return output;
    }

    void bp(const vec_t &delta, size_t batch_size) override {
        const size_t N   = batch_size;
        const size_t W   = in_width;
        const size_t H   = in_height;
        const size_t C   = input_channel;
        const size_t K   = output_channel;
        const size_t R   = window_size;
        const size_t S   = window_size;
        const size_t Q   = out_height;
        const size_t P   = out_width;
        const size_t npq = N * P * Q;
        const size_t crs = C * R * S;

        //
        // calc grad_weight
        //
        vec_t mF(K * npq);

        grad_weight.clear();
        grad_weight.resize(C * K * R * S);

        gemm('R',
             'N',
             'T',
             K,                     // m A(m,k) B(k,n) C(m,n)
             crs,                   // n
             npq,                   // k
             1.0 / N,               // alpha
             &delta[0],             // matrix A
             npq,                   // lda
             &this->input[0],       // matrix B
             npq,                   // ldb
             0.0,                   // beta
             &grad_weight[0],       // matrix C
             crs);                  // ldc

        //
        // calc grad_bias
        //
        grad_bias.clear();
        grad_bias.resize(output_channel);

        vec_t onevec(npq);
        std::fill(onevec.begin(), onevec.end(), 1.0);

        gemm('R',
             'N',
             'N',
             K,
             1,
             npq,
             1.0 / N,
             &delta[0],
             npq,
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
            this->delta.resize(N * C * H * W);

            vec_t mO(crs * npq);

            gemm('R',
                 'T',
                 'N',
                 crs,              // m A(m,k) B(k,n) C(m,n)
                 npq,              // n
                 K,                // k
                 1.0,              // alpha
                 &weight[0],       // matrix A
                 crs,              // lda
                 &delta[0],        // matrix B
                 npq,              // ldb
                 0.0,              // beta
                 &mO[0],           // matrix C
                 npq);             // ldc

            // col2im
            for (size_t n = 0; n < N; n++) {
                for (size_t c = 0; c < C; c++) {
                    for (size_t s = 0; s < S; s++) {
                        for (size_t r = 0; r < R; r++) {
                            for (size_t q = 0; q < Q; q++) {
                                for (size_t p = 0; p < P; p++) {
                                    const size_t o_i = (r + p) +
                                                       (s + q) * W +
                                                       n       * W * H +
                                                       c       * W * H * N;
                                    const size_t i_i = p +
                                                       q * P +
                                                       n * P * Q +
                                                       r * P * Q * N +
                                                       s * P * Q * N * R +
                                                       c * P * Q * N * R * S;
                                    this->delta[o_i] += mO[i_i];
                                }
                            }
                        }
                    }
                }
            }
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
