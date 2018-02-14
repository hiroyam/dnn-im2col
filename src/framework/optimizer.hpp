#pragma once

namespace cc {
class optimizer {
public:
    virtual void update(const vec_t &dw, vec_t &w) = 0;
};

class gradient_descent : public optimizer {
public:
    gradient_descent() : alpha(0.01), lambda(0) {}

    void update(const vec_t &dw, vec_t &w) {
        for (size_t i = 0; i < w.size(); i++) {
            w[i] = w[i] - alpha * (dw[i] + lambda * w[i]);
        }
    }

    float_t alpha;
    float_t lambda;
};
} // namespace cc
