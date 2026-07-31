#include "model.hpp"

namespace package_device_program {

[[nodiscard]] int CheckFailures(Device &device, const Backend backend) {
  // The installed SDK must preserve the selected backend and its typed
  // failure contract. A successful retry after every rejection proves the
  // Program remains usable; lower-level contracts additionally pin the
  // failure-atomic output bytes.
  if (backend != Backend::Cpu)
    return 0;

  constexpr std::array<std::int32_t, 4u> values{5, 7, 11, 13};
  constexpr std::array<std::uint32_t, 4u> valid_indices{0u, 1u, 0u, 1u};

  auto gather = on(device)
                    .input<std::int32_t>(values.size())
                    .zip_input<std::uint32_t>(valid_indices.size())
                    .branch([](auto source, auto requested) {
                      auto active = requested.filter([](auto index) {
                        return index != std::uint32_t{99};
                      });
                      return source.gather(active);
                    })
                    .compile();
  if (!gather)
    return gather.exit_code();
  const auto gather_backend = gather->backend();
  if (!gather_backend || *gather_backend != backend)
    return 2;
  constexpr std::array<std::uint32_t, 4u> invalid_gather{0u, 9u, 0u, 1u};
  auto rejected_gather = gather->run(values, invalid_gather);
  if (rejected_gather ||
      rejected_gather.error() != "compute_gather_index_out_of_range" ||
      *gather_backend != backend) {
    return 2;
  }
  auto recovered_gather = gather->run(values, valid_indices);
  if (!recovered_gather ||
      *recovered_gather != std::vector<std::int32_t>{5, 7, 5, 7}) {
    return 2;
  }

  auto scatter_reduce =
      on(device)
          .input<std::int32_t>(values.size())
          .zip_input<std::uint32_t>(valid_indices.size())
          .branch([](auto input, auto targets) {
            return input.scatter_reduce(targets, 2u, Reduce::Sum);
          })
          .compile();
  if (!scatter_reduce)
    return scatter_reduce.exit_code();
  const auto scatter_backend = scatter_reduce->backend();
  if (!scatter_backend || *scatter_backend != backend)
    return 2;
  constexpr std::array<std::uint32_t, 4u> invalid_scatter{0u, 2u, 0u, 1u};
  auto rejected_scatter = scatter_reduce->run(values, invalid_scatter);
  if (rejected_scatter ||
      rejected_scatter.error() != "compute_scatter_reduce_index_out_of_range" ||
      *scatter_backend != backend) {
    return 2;
  }
  auto recovered_scatter = scatter_reduce->run(values, valid_indices);
  if (!recovered_scatter ||
      *recovered_scatter != std::vector<std::int32_t>{16, 20}) {
    return 2;
  }

  auto bounded_scatter_reduce =
      on(device)
          .input<Bounded<std::int32_t>>(values.size())
          .branch([](auto input) {
            auto targets =
                input.indices().map("scatter-reduce-target", [](auto ordinal) {
                  return ordinal & std::uint32_t{1};
                });
            return input.scatter_reduce(targets, 2u, Reduce::Sum);
          })
          .compile();
  if (!bounded_scatter_reduce)
    return bounded_scatter_reduce.exit_code();
  const auto bounded_backend = bounded_scatter_reduce->backend();
  if (!bounded_backend || *bounded_backend != backend)
    return 2;
  constexpr std::array<std::uint32_t, 1u> overflowing_count{5u};
  auto rejected_count = bounded_scatter_reduce->run(values, overflowing_count);
  if (rejected_count || rejected_count.error() != "compute_workset_overflow" ||
      *bounded_backend != backend) {
    return 2;
  }
  constexpr std::array<std::uint32_t, 1u> full_count{values.size()};
  auto recovered_count = bounded_scatter_reduce->run(values, full_count);
  return recovered_count &&
                 *recovered_count == std::vector<std::int32_t>{16, 20}
             ? 0
             : 2;
}

} // namespace package_device_program
