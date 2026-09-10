#include "local.hpp"

#include <array>
#include <cstdio>

namespace rund_node_test_pipeline::view {

[[nodiscard]] int CheckSort(rund::compute::Device &device,
                            const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 8u> unsorted_values{9, 4, 7, 2,
                                                         5, 8, 1, 6};
  auto unsorted = Upload(device, unsorted_values);
  auto sort_target = device.buffer<std::int32_t>(unsorted_values.size());
  auto sort = on(device)
                  .input<std::int32_t>(4u)
                  .branch([](auto values) { return values.sort(); })
                  .compile();
  if (!unsorted || !sort_target || !sort) {
    return 14;
  }
  auto sort_input = unsorted->view(1u, 4u, 2u);
  auto sort_output = sort_target->view(0u, 4u, 2u);
  if (!sort_input || !sort_output) {
    return 15;
  }
  Stats dense_sort_stats{};
  {
    auto dense_sort_target = device.buffer<std::int32_t>(4u);
    auto dense_sort_input = unsorted->view(0u, 4u);
    auto dense_sort_output = dense_sort_target->view(0u, 4u);
    if (!dense_sort_target || !dense_sort_input || !dense_sort_output) {
      return 15;
    }
    auto dense_sorted =
        pipeline(device)
            .then(*sort, read(*dense_sort_input), write(*dense_sort_output))
            .prepare();
    std::array<std::int32_t, 4u> dense_sorted_values{};
    if (!dense_sorted || !dense_sorted->run() ||
        !ReadExact(*dense_sorted, *dense_sort_target, dense_sorted_values) ||
        dense_sorted_values != std::array<std::int32_t, 4u>{2, 4, 7, 9}) {
      return 15;
    }
    dense_sort_stats = dense_sorted->stats();
  }
  auto sorted = pipeline(device)
                    .then(*sort, read(*sort_input), write(*sort_output))
                    .prepare();
  std::array<std::int32_t, 8u> sorted_values{};
  if (!sorted || !sorted->run() ||
      !ReadExact(*sorted, *sort_target, sorted_values) ||
      sorted_values != std::array<std::int32_t, 8u>{2, 0, 4, 0, 6, 0, 8, 0} ||
      sorted->stats().internal_roundtrip_bytes != 32u ||
      (backend != Backend::Cpu &&
       (sorted->stats().dispatches != sorted->stats().final_dispatches ||
        sorted->stats().original_dispatches !=
            dense_sort_stats.original_dispatches ||
        sorted->stats().final_dispatches !=
            dense_sort_stats.final_dispatches + 2u))) {
    if (sorted) {
      std::fprintf(
          stderr,
          "pipeline view sort backend=%u roundtrip=%llu original=%llu "
          "dispatches=%llu final=%llu first=%d second=%d\n",
          static_cast<unsigned>(backend),
          static_cast<unsigned long long>(
              sorted->stats().internal_roundtrip_bytes),
          static_cast<unsigned long long>(sorted->stats().original_dispatches),
          static_cast<unsigned long long>(sorted->stats().dispatches),
          static_cast<unsigned long long>(sorted->stats().final_dispatches),
          sorted_values[0u], sorted_values[2u]);
    } else {
      std::fprintf(stderr, "pipeline view sort backend=%u prepare=%u\n",
                   static_cast<unsigned>(backend),
                   static_cast<unsigned>(sorted.reason()));
    }
    return 16;
  }

  return 0;
}

} // namespace rund_node_test_pipeline::view
