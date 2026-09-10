#include "local.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace rund_node_test_flow_primitives {

[[nodiscard]] int CheckScatterReduce() {
  using namespace rund::compute;
  const std::array<std::int32_t, 4u> values{5, 7, 11, 13};
  const std::array<std::uint32_t, 4u> indices{0u, 1u, 0u, 1u};
  auto exact = on(Target::cpu(2u))
                   .input<std::int32_t>(values.size())
                   .zip_input<std::uint32_t>(indices.size())
                   .branch([](auto input, auto targets) {
                     return input.scatter_reduce(targets, 2u, Reduce::Sum);
                   })
                   .compile();
  if (!exact)
    return 1;
  const auto exact_backend = exact->backend();
  if (!exact_backend || *exact_backend != Backend::Cpu)
    return 1;
  auto reduced = exact->run(values, indices);
  if (!reduced || *reduced != std::vector<std::int32_t>{16, 20})
    return 2;

  const std::array<std::uint32_t, 4u> invalid_indices{0u, 2u, 0u, 1u};
  auto invalid = exact->run(values, invalid_indices);
  if (invalid ||
      invalid.error() != "compute_scatter_reduce_index_out_of_range" ||
      *exact_backend != Backend::Cpu) {
    return 3;
  }

  auto bounded = on(Target::cpu(2u))
                     .input<Bounded<std::int32_t>>(values.size())
                     .branch([](auto input) {
                       auto targets = input.indices().map(
                           "scatter-reduce-target", [](auto ordinal) {
                             return ordinal & std::uint32_t{1};
                           });
                       return input.scatter_reduce(targets, 2u, Reduce::Sum);
                     })
                     .compile();
  if (!bounded)
    return 4;
  const std::array<std::uint32_t, 1u> overflow_count{5u};
  auto overflow = bounded->run(values, overflow_count);
  const auto bounded_backend = bounded->backend();
  if (overflow || overflow.error() != "compute_workset_overflow" ||
      !bounded_backend || *bounded_backend != Backend::Cpu) {
    return 5;
  }

  auto bounded_sort = on(Target::cpu(2u))
                          .input<Bounded<std::int32_t>>(values.size())
                          .branch([](auto input) { return input.sort(); })
                          .compile();
  if (!bounded_sort)
    return 6;
  const auto fingerprint = bounded_sort->graph().fingerprint;
  const std::size_t nodes = bounded_sort->graph().nodes.size();
  const std::size_t resources = bounded_sort->graph().resources.size();
  auto sort_overflow = bounded_sort->run(values, overflow_count);
  if (sort_overflow || sort_overflow.error() != "compute_workset_overflow" ||
      bounded_sort->graph().fingerprint != fingerprint ||
      bounded_sort->graph().nodes.size() != nodes ||
      bounded_sort->graph().resources.size() != resources) {
    return 7;
  }
  constexpr std::array<std::uint32_t, 1u> full_count{values.size()};
  auto sorted = bounded_sort->run(values, full_count);
  return sorted && *sorted == std::vector<std::int32_t>{5, 7, 11, 13} ? 0 : 8;
}

} // namespace rund_node_test_flow_primitives
