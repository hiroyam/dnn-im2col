#pragma once

namespace cc {
class network_t {
public:
    std::shared_ptr<layer_t> head_layer;
    std::shared_ptr<layer_t> tail_layer;

    size_t num_layers;
    network_t() : num_layers(0) {}

    template<typename Error, typename Optimizer, typename OnMinibatch, typename OnEpoch>
    void train(Optimizer                  &optimizer,
               const std::vector<vec_t>   &images,
               const std::vector<label_t> &labels,
               size_t                      minibatch_size,
               size_t                      num_epoches,
               OnMinibatch                 on_minibatch_enumerate,
               OnEpoch                     on_epoch_enumerate) {
        for (size_t epoch = 0; epoch < num_epoches; epoch++) {
            std::cout << format_str("epoch %zu", epoch) << std::endl;
            int error_count = 0;
            for (size_t itr = 0; itr < images.size() / minibatch_size; itr++) {
                progress_display(itr, images.size() / minibatch_size);

                // tensor2vec
                vec_t input;
                for (size_t n = 0; n < minibatch_size; n++) {
                    size_t index = n + itr * minibatch_size;
                    input.insert(input.end(), images[index].begin(), images[index].end());
                }

                // label2vec
                vec_t onehot(minibatch_size * tail_layer->output_channel);
                for (size_t n = 0; n < minibatch_size; n++) {
                    size_t index = n + itr * minibatch_size;
                    onehot[labels[index] + n * tail_layer->output_channel] = 1.0;
                }

                // forward propagation
                vec_t fp = head_layer->fp(input, minibatch_size);

                // calculate delta
                vec_t df = Error::df(fp, onehot);

                // back propagation
                tail_layer->bp(df, minibatch_size);

                // update parameters
                head_layer->update(optimizer);

                // calculate error count
                for (size_t n = 0; n < minibatch_size; n++) {
                    auto   begin   = std::begin(fp) + (n + 0) * tail_layer->output_channel;
                    auto   end     = std::begin(fp) + (n + 1) * tail_layer->output_channel;
                    size_t predict = std::max_element(begin, end) - begin;
                    size_t index   = n + itr * minibatch_size;
                    if (labels[index] != predict) {
                        error_count++;
                    }
                }
                on_minibatch_enumerate();
            }
            std::cout << format_str("%.1f%%\n", 100.0 * (images.size() - error_count) / images.size());
            on_epoch_enumerate();
        }
    }

    void inference(const std::vector<vec_t>   &images,
                   const std::vector<label_t> &labels,
                   size_t                      minibatch_size) {
        int error_count = 0;
        for (size_t itr = 0; itr < images.size() / minibatch_size; itr++) {
            progress_display(itr, images.size() / minibatch_size);

            // tensor2vec
            vec_t input;
            for (size_t n = 0; n < minibatch_size; n++) {
                size_t index = n + itr * minibatch_size;
                input.insert(input.end(), images[index].begin(), images[index].end());
            }

            // label2vec
            vec_t onehot(minibatch_size * tail_layer->output_channel);
            for (size_t n = 0; n < minibatch_size; n++) {
                size_t index = n + itr * minibatch_size;
                onehot[labels[index] + n * tail_layer->output_channel] = 1.0;
            }

            // forward propagation
            vec_t fp = head_layer->fp(input, minibatch_size);

            // calculate error count .
            for (size_t n = 0; n < minibatch_size; n++) {
                auto   begin   = std::begin(fp) + (n + 0) * tail_layer->output_channel;
                auto   end     = std::begin(fp) + (n + 1) * tail_layer->output_channel;
                size_t predict = std::max_element(begin, end) - begin;
                size_t index   = n + itr * minibatch_size;
                if (labels[index] != predict) {
                    error_count++;
                }
            }
        }
        std::cout << format_str("%.1f%%\n", 100.0 * (images.size() - error_count) / images.size());
    }
};

template<typename Layer>
network_t &operator<<(network_t &nn, Layer && l) {
    std::shared_ptr<Layer> b(new Layer(l));

    b->nth_layer = nn.num_layers++;

    if (nn.head_layer == nullptr) {
        nn.tail_layer = b;
        nn.head_layer = b;
    } else {
        b->prev_layer = nn.tail_layer;
        b->prev_layer.lock()->next_layer = b;
        nn.tail_layer = b;
    }
    return nn;
}
}

