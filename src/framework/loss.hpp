
namespace di {

class cross_entropy_multiclass {
 public:
  static float_t f(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    float_t d{0.0};

    for (size_t i = 0; i < y.size(); ++i) d += -t[i] * std::log(y[i]);

    return d;
  }

  static vec_t df(const vec_t &y, const vec_t &t) {
    assert(y.size() == t.size());
    vec_t d(t.size());

    for (size_t i = 0; i < y.size(); ++i) d[i] = -t[i] / y[i];

    return d;
  }
};

}
