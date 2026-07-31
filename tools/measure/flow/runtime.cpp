#include <rund/compute.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>

namespace {
using Clock = std::chrono::steady_clock;

struct Value final {};
struct Weight final {};
struct Score final {};
struct Total final {};

template <class Build> void Measure(const char *name, Build build) {
  std::array<double, 101u> samples{};
  build();
  for (std::size_t index = 0; index < samples.size(); ++index) {
    const auto begin = Clock::now();
    build();
    const auto end = Clock::now();
    samples[index] =
        std::chrono::duration<double, std::micro>(end - begin).count();
    std::printf("%s\t%zu\tus\t%.3f\n", name, index + 1u, samples[index]);
  }
  std::sort(samples.begin(), samples.end());
  std::printf("%s\tmedian\tus\t%.3f\n", name, samples[samples.size() / 2u]);
}
} // namespace

int main() {
  Measure("graph_minimal", [] {
    auto flow = rund::compute::on(rund::compute::Target::cpu(1u))
                    .map<std::int32_t>("minimal", 4096u, [](auto value) {
                      return value * 3 + 1;
                    });
    (void)flow;
  });
  Measure("graph_typed", [] {
    using namespace rund::compute;
    auto flow =
        on(Target::cpu(1u))
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
    (void)flow;
  });
}
