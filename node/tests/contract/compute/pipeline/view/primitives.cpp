#include "local.hpp"

#include <array>

namespace rund_node_test_pipeline::view {

[[nodiscard]] int CheckPrimitives(rund::compute::Device &device,
                                  const rund::compute::Backend backend) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 8u> source_values{0, 1, 2, 3, 4, 5, 6, 7};
  auto source = Upload(device, source_values);
  if (!source) {
    return 23;
  }
  auto scan =
      on(device)
          .input<std::int32_t>(4u)
          .branch([](auto values) { return values.scan(Scan::InclusiveSum); })
          .compile();
  auto scan_target = device.buffer<std::int32_t>(source_values.size());
  auto scan_input = source->view(1u, 4u, 2u);
  if (!scan || !scan_target || !scan_input) {
    return 23;
  }
  auto scan_output = scan_target->view(1u, 4u, 2u);
  if (!scan_output) {
    return 23;
  }
  auto scanned = pipeline(device)
                     .then(*scan, read(*scan_input), write(*scan_output))
                     .prepare();
  std::array<std::int32_t, 8u> scanned_values{};
  if (!scanned || !scanned->run() ||
      !ReadExact(*scanned, *scan_target, scanned_values) ||
      scanned_values != std::array<std::int32_t, 8u>{0, 1, 0, 4, 0, 9, 0, 16} ||
      scanned->stats().internal_roundtrip_bytes != 32u ||
      (backend != Backend::Cpu &&
       (scanned->stats().dispatches != scanned->stats().final_dispatches ||
        scanned->stats().final_dispatches !=
            scanned->stats().original_dispatches + 2u))) {
    return 24;
  }

  constexpr std::array<std::int32_t, 8u> left_values{1, 0, 2, 0, 3, 0, 4, 0};
  constexpr std::array<std::int32_t, 8u> right_values{5, 0, 6, 0, 7, 0, 8, 0};
  auto left = Upload(device, left_values);
  auto right = Upload(device, right_values);
  auto matrix_target = device.buffer<std::int32_t>(left_values.size());
  auto matrix = on(device)
                    .map<std::int32_t>("pipeline-view-matrix", 4u,
                                       [](auto value) { return value; })
                    .matrix<2u, 2u>()
                    .matmul<2u, 2u>()
                    .compile();
  if (!left || !right || !matrix_target || !matrix) {
    return 25;
  }
  auto left_view = left->view(0u, 4u, 2u);
  auto right_view = right->view(0u, 4u, 2u);
  auto matrix_output = matrix_target->view(0u, 4u, 2u);
  if (!left_view || !right_view || !matrix_output) {
    return 26;
  }
  auto multiplied =
      pipeline(device)
          .then(*matrix, read(*left_view, *right_view), write(*matrix_output))
          .prepare();
  std::array<std::int32_t, 8u> matrix_values{};
  if (!multiplied || !multiplied->run() ||
      !ReadExact(*multiplied, *matrix_target, matrix_values) ||
      matrix_values !=
          std::array<std::int32_t, 8u>{19, 0, 22, 0, 43, 0, 50, 0} ||
      multiplied->stats().internal_roundtrip_bytes != 48u ||
      (backend != Backend::Cpu &&
       (multiplied->stats().dispatches !=
            multiplied->stats().final_dispatches ||
        multiplied->stats().final_dispatches !=
            multiplied->stats().original_dispatches + 3u))) {
    return 27;
  }

  return 0;
}

} // namespace rund_node_test_pipeline::view
