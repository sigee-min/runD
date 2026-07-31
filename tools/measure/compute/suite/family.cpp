#include "core.hpp"

namespace rund::measure::compute {

#if !defined(RUND_COMPUTE_FOCUS)
bool Families(const Backend backend) {
  constexpr std::size_t count = 4096u;
  constexpr std::size_t iterations = 7u;
  std::vector<std::uint32_t> values(count);
  std::vector<std::int32_t> signed_values(count);
  std::vector<std::uint32_t> indices(count);
  std::vector<std::uint32_t> flags(count);
  std::vector<std::uint32_t> heads(count);
  for (std::size_t i = 0; i < count; ++i) {
    values[i] = static_cast<std::uint32_t>((i * 37u + 11u) & 255u);
    signed_values[i] = static_cast<std::int32_t>(i % 257u) - 128;
    indices[i] = static_cast<std::uint32_t>(count - 1u - i);
    flags[i] = static_cast<std::uint32_t>(i & 1u);
    heads[i] = static_cast<std::uint32_t>((i % 256u) == 0u);
  }
  bool ok = true;

  auto map = rund::compute::on(TargetFor(backend))
                 .map<std::uint32_t>("family-map", count,
                                     [](auto value) { return value * 3u + 1u; })
                 .compile();
  ok = Bench("map", backend, map, iterations,
             ReferenceKey{"family", "map", count}, values) &&
       ok;
  auto select =
      rund::compute::on(TargetFor(backend))
          .map<std::uint32_t>("family-select", count,
                              [](auto value) {
                                return rund::compute::select<std::uint32_t>(
                                    value > 127u, value, 127u);
                              })
          .compile();
  ok = Bench("select", backend, select, iterations,
             ReferenceKey{"family", "select", count}, values) &&
       ok;
  auto scan = rund::compute::on(TargetFor(backend))
                  .map<std::uint32_t>("family-scan", count,
                                      [](auto value) { return value; })
                  .scan(rund::compute::Scan::InclusiveSum)
                  .compile();
  ok = Bench("scan", backend, scan, iterations,
             ReferenceKey{"family", "scan", count}, values) &&
       ok;
  auto reduce = rund::compute::on(TargetFor(backend))
                    .map<std::uint32_t>("family-reduce", count,
                                        [](auto value) { return value; })
                    .reduce()
                    .compile();
  ok = Bench("reduce", backend, reduce, iterations,
             ReferenceKey{"family", "reduce", count}, values) &&
       ok;
  auto sort = rund::compute::on(TargetFor(backend))
                  .map<std::uint32_t>("family-sort", count,
                                      [](auto value) { return value; })
                  .sort()
                  .compile();
  ok = Bench("sort", backend, sort, iterations,
             ReferenceKey{"family", "sort", count}, values) &&
       ok;
  auto argsort = rund::compute::on(TargetFor(backend))
                     .map<std::uint32_t>("family-argsort", count,
                                         [](auto value) { return value; })
                     .argsort()
                     .compile();
  ok = Bench("argsort", backend, argsort, iterations,
             ReferenceKey{"family", "argsort", count}, values) &&
       ok;
  auto compact = rund::compute::on(TargetFor(backend))
                     .map<std::uint32_t>("family-compact", count,
                                         [](auto value) { return value & 1u; })
                     .compact({.capacity = count})
                     .compile();
  ok = Bench("compact", backend, compact, iterations,
             ReferenceKey{"family", "compact", count}, values) &&
       ok;
  auto filter = rund::compute::on(TargetFor(backend))
                    .map<std::int32_t>("family-filter", count,
                                       [](auto value) { return value; })
                    .filter([](auto value) { return value > 0; })
                    .compile();
  ok = Bench("filter", backend, filter, iterations,
             ReferenceKey{"family", "filter", count}, signed_values) &&
       ok;
  auto histogram = rund::compute::on(TargetFor(backend))
                       .map<std::uint32_t>("family-histogram", count,
                                           [](auto value) { return value; })
                       .histogram({.bins = 256u})
                       .compile();
  ok = Bench("histogram", backend, histogram, iterations,
             ReferenceKey{"family", "histogram", count}, values) &&
       ok;
  auto gather = rund::compute::on(TargetFor(backend))
                    .map<std::uint32_t>("family-gather", count,
                                        [](auto value) { return value; })
                    .gather(count)
                    .compile();
  ok = Bench("gather", backend, gather, iterations,
             ReferenceKey{"family", "gather", count}, values, indices) &&
       ok;
  auto scatter = rund::compute::on(TargetFor(backend))
                     .map<std::uint32_t>("family-scatter", count,
                                         [](auto value) { return value; })
                     .scatter(count, {.count = count})
                     .compile();
  ok = Bench("scatter", backend, scatter, iterations,
             ReferenceKey{"family", "scatter", count}, values, indices) &&
       ok;
  auto partition = rund::compute::on(TargetFor(backend))
                       .map<std::uint32_t>("family-partition", count,
                                           [](auto value) { return value; })
                       .partition(count)
                       .compile();
  ok = Bench("partition", backend, partition, iterations,
             ReferenceKey{"family", "partition", count}, values, flags) &&
       ok;
  auto segmented_scan =
      rund::compute::on(TargetFor(backend))
          .map<std::uint32_t>("family-segmented-scan", count,
                              [](auto value) { return value; })
          .segmented_scan(count, rund::compute::Scan::InclusiveSum)
          .compile();
  ok = Bench("segmented_scan", backend, segmented_scan, iterations,
             ReferenceKey{"family", "segmented_scan", count}, values, heads) &&
       ok;
  auto segmented_reduce =
      rund::compute::on(TargetFor(backend))
          .map<std::uint32_t>("family-segmented-reduce", count,
                              [](auto value) { return value; })
          .segmented_reduce(count)
          .compile();
  ok =
      Bench("segmented_reduce", backend, segmented_reduce, iterations,
            ReferenceKey{"family", "segmented_reduce", count}, values, heads) &&
      ok;
  auto stencil = rund::compute::on(TargetFor(backend))
                     .map<std::uint32_t>("family-stencil", count,
                                         [](auto value) { return value; })
                     .window({.op = rund::compute::Window::Sum, .radius = 2u})
                     .compile();
  ok = Bench("stencil", backend, stencil, iterations,
             ReferenceKey{"family", "stencil", count}, values) &&
       ok;

  constexpr std::size_t transform_count = 64u;
  std::vector<Fixed<1, 31>> real(transform_count,
                                 Fixed<1, 31>::from_raw(1 << 20));
  std::vector<Fixed<1, 31>> imag(transform_count, Fixed<1, 31>::zero());
  auto transform =
      rund::compute::on(TargetFor(backend))
          .map<Fixed<1, 31>>("family-transform", transform_count,
                             [](auto value) {
                               return rund::compute::quantize<Fixed<1, 31>>(
                                   value);
                             })
          .complex()
          .fourier()
          .compile();
  ok =
      Bench("transform", backend, transform, iterations,
            ReferenceKey{"family", "transform", transform_count}, real, imag) &&
      ok;

  constexpr std::size_t side = 4u;
  std::vector<std::int32_t> left(side * side, 0);
  std::vector<std::int32_t> right(side * side, 0);
  std::vector<Fixed<1, 31>> fixed(side * side, Fixed<1, 31>::zero());
  std::vector<Fixed<1, 31>> rhs(side, Fixed<1, 31>::from_raw(1 << 24));
  for (std::size_t i = 0; i < side; ++i) {
    left[i * side + i] = 2;
    right[i * side + i] = 3;
    fixed[i * side + i] = Fixed<1, 31>::from_raw(1 << 29);
  }
  auto matrix = rund::compute::on(TargetFor(backend))
                    .map<std::int32_t>("family-matrix", side * side,
                                       [](auto value) { return value; })
                    .matrix<side, side>()
                    .matmul<side, side>()
                    .compile();
  ok = Bench("matrix", backend, matrix, iterations,
             ReferenceKey{"family", "matrix", side * side}, left, right) &&
       ok;
  auto factor = rund::compute::on(TargetFor(backend))
                    .map<Fixed<1, 31>>(
                        "family-factor", side * side,
                        [](auto value) {
                          return rund::compute::quantize<Fixed<1, 31>>(value);
                        })
                    .matrix<side, side>()
                    .lu()
                    .compile();
  ok = Bench("factor", backend, factor, iterations,
             ReferenceKey{"family", "factor", side * side}, fixed) &&
       ok;
  auto solve = rund::compute::on(TargetFor(backend))
                   .map<Fixed<1, 31>>(
                       "family-solve", side * side,
                       [](auto value) {
                         return rund::compute::quantize<Fixed<1, 31>>(value);
                       })
                   .matrix<side, side>()
                   .lu()
                   .solve<1u>()
                   .compile();
  ok = Bench("solve", backend, solve, iterations,
             ReferenceKey{"family", "solve", side * side}, fixed, rhs) &&
       ok;
  auto spectrum = rund::compute::on(TargetFor(backend))
                      .map<Fixed<1, 31>>(
                          "family-spectrum", side * side,
                          [](auto value) {
                            return rund::compute::quantize<Fixed<1, 31>>(value);
                          })
                      .matrix<side, side>()
                      .svd<rund::compute::SpectrumVectors::Values>(12u)
                      .compile();
  ok = Bench("spectrum", backend, spectrum, iterations,
             ReferenceKey{"family", "spectrum", side * side}, fixed) &&
       ok;
  return ok;
}

#endif

} // namespace rund::measure::compute
