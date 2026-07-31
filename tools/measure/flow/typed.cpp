#include <rund/compute.hpp>

#include <cstdint>

struct Value final {};
struct Weight final {};
struct Score final {};
struct Total final {};

auto BuildTyped() {
  using namespace rund::compute;
  return on(Target::cpu(1u))
      .input<std::int32_t>(4096u)
      .zip_input<std::uint32_t>(4096u)
      .map("score",
           [](auto value, auto weight) {
             return record(field<Value>(value), field<Weight>(weight),
                           field<Score>(value * weight));
           })
      .branch([](auto rows) {
        const auto score = rows.template get<Score>();
        const auto total = score.reduce(Reduce::Sum);
        const auto normalized =
            score.combine("normalize", total, [](auto value, auto sum) {
              return value * 1000 / sum;
            });
        return outputs(rows, normalized, record(field<Total>(total)));
      });
}
