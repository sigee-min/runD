#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>

#include <array>
#include <cstdio>

namespace rund_node_test_device_vsm_product::multi_scan_test {

bool RejectWideMultiScan(const rund::compute::Backend backend,
                         bool &unavailable) noexcept {
  using namespace rund::compute;
  unavailable = false;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return unavailable;
  }
  auto program =
      on(*opened)
          .input<std::uint64_t>(PageElements)
          .zip_input<std::uint64_t>(PageElements)
          .zip_input<std::uint64_t>(PageElements)
          .zip_input<std::uint64_t>(PageElements)
          .zip_input<std::uint64_t>(PageElements)
          .zip_input<std::uint64_t>(PageElements)
          .zip_input<std::uint64_t>(PageElements)
          .zip_input<std::uint64_t>(PageElements)
          .map("device-vsm-wide-map-scan",
               [](auto a, auto b, auto c, auto d, auto e, auto f, auto g,
                  auto h) { return a + b + c + d + e + f + g + h; })
          .scan(Scan::InclusiveSum)
          .compile();
  if (!program) {
    std::fprintf(stderr, "DeviceVsm wide Scan compile backend=%u reason=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(program.reason()));
    return false;
  }
  constexpr std::uint64_t elements = PageElements * 5u - 3u;
  const auto backing = [] {
    return std::make_shared<
        rund_node_test_persistent_product::PersistentProductBacking>(
        static_cast<std::size_t>(elements * sizeof(std::uint64_t)));
  };
  std::array<std::shared_ptr<detail::VirtualBufferState>, 8u> inputs{};
  for (std::shared_ptr<detail::VirtualBufferState> &input : inputs) {
    auto buffer = detail::make_virtual_buffer(elements, sizeof(std::uint64_t),
                                              detail::Type::U64, {}, backing());
    if (!buffer) {
      return false;
    }
    input = std::move(buffer).value();
  }
  auto output = detail::make_virtual_buffer(elements, sizeof(std::uint64_t),
                                            detail::Type::U64, {}, backing());
  const std::shared_ptr<detail::ProgramState> state =
      program ? detail::ProgramAccess::state(*program) : nullptr;
  if (state == nullptr || !output) {
    return false;
  }
  const auto prepared = detail::prepare_virtual_pipeline(
      state, inputs, std::move(output).value(), ResidencyConfig{});
  const bool rejected =
      !prepared && prepared.reason() == Reason::PipelineInvalid;
  if (!rejected) {
    std::fprintf(stderr, "DeviceVsm wide Scan prepare backend=%u reason=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned>(prepared.reason()));
  }
  return rejected;
}

} // namespace rund_node_test_device_vsm_product::multi_scan_test

#endif
