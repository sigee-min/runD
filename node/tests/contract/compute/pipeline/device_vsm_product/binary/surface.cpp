#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"
#include "../../persistent_product/fixture.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <cstdio>
#include <cstring>
#include <span>
#include <vector>

namespace rund_node_test_device_vsm_product::binary_test {

bool RunPublicBinaryProduct(const rund::compute::Backend backend,
                            bool &unavailable) noexcept {
  using namespace rund::compute;
  constexpr std::uint64_t Pages = 9u;
  const std::uint64_t elements = Pages * PageElements - 3u;
  unavailable = false;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto program =
      on(*opened)
          .input<std::uint32_t>(PageElements)
          .zip_input<std::uint32_t>(PageElements)
          .map("device-vsm-public-binary",
               [](auto first, auto second) { return first + second; })
          .compile();
  auto first_backing = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint32_t)));
  auto second_backing = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint32_t)));
  auto output_backing = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint32_t)));
  std::vector<std::uint32_t> first(static_cast<std::size_t>(elements));
  std::vector<std::uint32_t> second(first.size());
  std::vector<std::uint32_t> expected(first.size());
  for (std::size_t index = 0u; index < first.size(); ++index) {
    first[index] = static_cast<std::uint32_t>(index * 11u + 5u);
    second[index] = static_cast<std::uint32_t>(index * 13u + 3u);
    expected[index] = first[index] + second[index];
  }
  if (!program || first_backing == nullptr || second_backing == nullptr ||
      output_backing == nullptr ||
      !first_backing->seed(std::as_bytes(std::span{first})) ||
      !second_backing->seed(std::as_bytes(std::span{second}))) {
    return false;
  }
  auto first_buffer = virtual_buffer<std::uint32_t>(elements, first_backing);
  auto second_buffer = virtual_buffer<std::uint32_t>(elements, second_backing);
  auto output_buffer = virtual_buffer<std::uint32_t>(elements, output_backing);
  if (!first_buffer || !second_buffer || !output_buffer) {
    return false;
  }
  auto pipeline = virtual_pipeline(*program, *first_buffer, *second_buffer,
                                   *output_buffer, ResidencyConfig{});
  if (!pipeline) {
    std::fprintf(stderr, "DeviceVsm public binary prepare failed reason=%u\n",
                 static_cast<unsigned>(pipeline.reason()));
    return false;
  }
  const std::uint64_t version_before =
      rund_node_test_persistent_product::BackingVersion(*output_backing);
  const Status status = pipeline->run();
  if (!status) {
    std::fprintf(stderr, "DeviceVsm public binary run failed reason=%u\n",
                 static_cast<unsigned>(status.reason()));
    return false;
  }
  std::vector<std::uint32_t> observed(expected.size());
  const Stats stats = pipeline->stats();
  const ResidencyStats &residency = stats.pipeline.residency;
  const std::uint64_t bytes = elements * sizeof(std::uint32_t);
  const bool output =
      output_backing->observe(std::as_writable_bytes(std::span{observed})) &&
      observed == expected;
  const std::uint64_t version =
      rund_node_test_persistent_product::BackingVersion(*output_backing);
  const std::uint64_t recovery =
      rund_node_test_persistent_product::BackingRecovery(*output_backing);
  const bool valid = output && stats.command_submits == 1u &&
                     stats.dispatches == 1u &&
                     residency.window_handoff_count == 1u &&
                     residency.window_queue_call_count == 1u &&
                     residency.page_in_count == 2u * Pages &&
                     residency.page_out_count == Pages &&
                     residency.backing_read_bytes == 2u * bytes &&
                     residency.backing_write_bytes == bytes &&
                     version == version_before + 1u && recovery == 0u;
  std::fprintf(
      stderr,
      "DeviceVsm public binary backend=%u valid=%u output=%u "
      "submit=%llu dispatch=%llu handoff=%llu queue=%llu "
      "page=%llu/%llu bytes=%llu/%llu version=%llu recovery=%llu\n",
      static_cast<unsigned>(backend), static_cast<unsigned>(valid),
      static_cast<unsigned>(output),
      static_cast<unsigned long long>(stats.command_submits),
      static_cast<unsigned long long>(stats.dispatches),
      static_cast<unsigned long long>(residency.window_handoff_count),
      static_cast<unsigned long long>(residency.window_queue_call_count),
      static_cast<unsigned long long>(residency.page_in_count),
      static_cast<unsigned long long>(residency.page_out_count),
      static_cast<unsigned long long>(residency.backing_read_bytes),
      static_cast<unsigned long long>(residency.backing_write_bytes),
      static_cast<unsigned long long>(version),
      static_cast<unsigned long long>(recovery));
  return valid;
}

} // namespace rund_node_test_device_vsm_product::binary_test

#endif
