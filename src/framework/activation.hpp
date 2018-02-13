#pragma once

namespace di {
namespace activation {

class function {
public:
    virtual float_t f(float_t v)  const = 0;
    virtual float_t df(float_t v) const = 0;
};

class tan_h : public function {
public:
    inline float_t f(float_t v) const override {
        return std::tanh(v);
    }
    inline float_t df(float_t v) const override {
        return 1.0 - sqr(v);
    }
};

class sigmoid : public function {
public:
    inline float_t f(float_t v) const override {
        return 1.0 / (1.0 + std::exp(-v));
    }
    inline float_t df(float_t v) const override {
        return v * (1.0 - v);
    }
};

class relu : public function {
public:
    inline float_t f(float_t v) const override {
        return std::max(v, float_t(0));
    }
    inline float_t df(float_t v) const override {
        return v > 0 ? 1.0 : 0.0;
    }
};

class leaky_relu : public function {
public:
    inline float_t f(float_t v) const override {
        return v > 0 ? v : 0.01 * v;
    }
    inline float_t df(float_t v) const override {
        return v > 0 ? 1.0 : 0.01;
    }
};

#define K_PRELU_GRAD 0.2
class parametric_relu : public function {
public:
    inline float_t f(float_t v) const override {
        return v > 0 ? v : K_PRELU_GRAD * v;
    }
    inline float_t df(float_t v) const override {
        return v > 0 ? 1.0 : K_PRELU_GRAD;
    }
};

class elu : public function {
public:
    inline float_t f(float_t v) const override {
        return v > 0 ? v : std::exp(v) - 1.0;
    }
    inline float_t df(float_t v) const override {
        return v > 0 ? 1.0 : v + 1.0;
    }
};

} // namespace activation
} // namespace di
