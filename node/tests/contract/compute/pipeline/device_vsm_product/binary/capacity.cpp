#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"
#include "../../persistent_product/fixture.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstdio>
#include <memory>
#include <span>
#include <vector>

namespace rund_node_test_device_vsm_product::binary_test {
namespace {

using Backing = rund_node_test_persistent_product::PersistentProductBacking;

constexpr std::uint64_t InputCount = 7u;
constexpr std::uint64_t Pages = 5u;

} // namespace

bool RunPublicMaximumInputProduct(const rund::compute::Backend backend,
                                  bool &unavailable) noexcept {
  using namespace rund::compute;
  const std::uint64_t elements = Pages * PageElements - 3u;
  const std::size_t bytes =
      static_cast<std::size_t>(elements * sizeof(std::uint32_t));
  unavailable = false;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto program = on(*opened)
                     .input<std::uint32_t>(PageElements)
                     .zip_input<std::uint32_t>(PageElements)
                     .zip_input<std::uint32_t>(PageElements)
                     .zip_input<std::uint32_t>(PageElements)
                     .zip_input<std::uint32_t>(PageElements)
                     .zip_input<std::uint32_t>(PageElements)
                     .zip_input<std::uint32_t>(PageElements)
                     .map("device-vsm-public-seven-input",
                          [](auto a, auto b, auto c, auto d, auto e, auto f,
                             auto g) { return a + b + c + d + e + f + g; })
                     .compile();
  std::array<std::shared_ptr<Backing>, InputCount> input_backings{};
  std::array<std::vector<std::uint32_t>, InputCount> inputs{};
  std::vector<std::uint32_t> expected(static_cast<std::size_t>(elements), 0u);
  for (std::size_t input = 0u; input < input_backings.size(); ++input) {
    input_backings[input] = std::make_shared<Backing>(bytes);
    inputs[input].resize(expected.size());
    for (std::size_t index = 0u; index < expected.size(); ++index) {
      inputs[input][index] =
          static_cast<std::uint32_t>(1u + input * 17u + index * (input + 3u));
      expected[index] += inputs[input][index];
    }
    if (input_backings[input] == nullptr ||
        !input_backings[input]->seed(std::as_bytes(std::span{inputs[input]}))) {
      return false;
    }
  }
  auto output_backing = std::make_shared<Backing>(bytes);
  if (!program || output_backing == nullptr) {
    return false;
  }
  auto a = virtual_buffer<std::uint32_t>(elements, input_backings[0u]);
  auto b = virtual_buffer<std::uint32_t>(elements, input_backings[1u]);
  auto c = virtual_buffer<std::uint32_t>(elements, input_backings[2u]);
  auto d = virtual_buffer<std::uint32_t>(elements, input_backings[3u]);
  auto e = virtual_buffer<std::uint32_t>(elements, input_backings[4u]);
  auto f = virtual_buffer<std::uint32_t>(elements, input_backings[5u]);
  auto g = virtual_buffer<std::uint32_t>(elements, input_backings[6u]);
  auto output = virtual_buffer<std::uint32_t>(elements, output_backing);
  if (!a || !b || !c || !d || !e || !f || !g || !output) {
    return false;
  }
  auto pipeline = virtual_pipeline(*program, *a, *b, *c, *d, *e, *f, *g,
                                   *output, ResidencyConfig{});
  if (!pipeline) {
    std::fprintf(stderr,
                 "DeviceVsm public seven-input prepare failed reason=%u\n",
                 static_cast<unsigned>(pipeline.reason()));
    return false;
  }
  const std::uint64_t version_before =
      rund_node_test_persistent_product::BackingVersion(*output_backing);
  const Status status = pipeline->run();
  std::vector<std::uint32_t> observed(expected.size());
  const Stats stats = pipeline->stats();
  const ResidencyStats &residency = stats.pipeline.residency;
  const std::uint64_t version =
      rund_node_test_persistent_product::BackingVersion(*output_backing);
  const std::uint64_t recovery =
      rund_node_test_persistent_product::BackingRecovery(*output_backing);
  const bool output_exact =
      output_backing->observe(std::as_writable_bytes(std::span{observed})) &&
      observed == expected;
  const bool valid = status && output_exact && stats.command_submits == 1u &&
                     stats.dispatches == 1u &&
                     residency.window_handoff_count == 1u &&
                     residency.window_queue_call_count == 1u &&
                     residency.page_in_count == InputCount * Pages &&
                     residency.page_out_count == Pages &&
                     residency.backing_read_bytes == InputCount * bytes &&
                     residency.backing_write_bytes == bytes &&
                     version == version_before + 1u && recovery == 0u;
  std::fprintf(stderr,
               "DeviceVsm public seven-input backend=%u valid=%u output=%u "
               "submit=%llu dispatch=%llu page=%llu/%llu bytes=%llu/%llu "
               "version=%llu recovery=%llu\n",
               static_cast<unsigned>(backend), static_cast<unsigned>(valid),
               static_cast<unsigned>(output_exact),
               static_cast<unsigned long long>(stats.command_submits),
               static_cast<unsigned long long>(stats.dispatches),
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
