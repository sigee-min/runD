#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstdio>
#include <cstring>
#include <span>

namespace rund_node_test_device_vsm_product::multi_scan_test {

bool PrepareMultiScanProduct(const rund::compute::Backend backend,
                             const std::uint64_t pages,
                             const rund::kernel::ScanOp operation,
                             PreparedMultiScan &prepared, bool &unavailable) {
  using namespace rund::compute;
  prepared = {};
  unavailable = false;
  const bool exclusive = operation == rund::kernel::ScanOp::ExclusiveSum;
  const std::uint64_t payload = PageElements - (exclusive ? 1u : 0u);
  const std::uint64_t elements = pages * payload - 3u;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto program =
      on(*opened)
          .input<std::uint64_t>(PageElements)
          .zip_input<std::uint64_t>(PageElements)
          .map("device-vsm-multi-map-scan",
               [](auto first, auto second) { return first + second; })
          .scan(exclusive ? Scan::ExclusiveSum : Scan::InclusiveSum)
          .compile();
  prepared.first = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint64_t)));
  prepared.second = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint64_t)));
  prepared.output = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint64_t)));
  prepared.element_count = elements;
  prepared.operation = operation;
  std::vector<std::uint64_t> first(static_cast<std::size_t>(elements));
  std::vector<std::uint64_t> second(first.size());
  prepared.expected.resize(first.size());
  std::uint64_t prefix = 0u;
  for (std::size_t index = 0u; index < first.size(); ++index) {
    first[index] = (index * 7u + pages) % 31u;
    second[index] = (index * 11u + 5u) % 37u;
    if (!exclusive) {
      prefix += first[index] + second[index];
    }
    prepared.expected[index] = prefix;
    if (exclusive) {
      prefix += first[index] + second[index];
    }
  }
  if (prepared.first == nullptr || prepared.second == nullptr ||
      prepared.output == nullptr ||
      !prepared.first->seed(std::as_bytes(std::span{first})) ||
      !prepared.second->seed(std::as_bytes(std::span{second}))) {
    return false;
  }
  auto first_buffer = detail::make_virtual_buffer(
      elements, sizeof(std::uint64_t), detail::Type::U64, {}, prepared.first);
  auto second_buffer = detail::make_virtual_buffer(
      elements, sizeof(std::uint64_t), detail::Type::U64, {}, prepared.second);
  auto output_buffer = detail::make_virtual_buffer(
      elements, sizeof(std::uint64_t), detail::Type::U64, {}, prepared.output);
  const auto program_state =
      program ? detail::ProgramAccess::state(*program) : nullptr;
  if (program_state == nullptr || !first_buffer || !second_buffer ||
      !output_buffer) {
    return false;
  }
  const std::array inputs{std::move(first_buffer).value(),
                          std::move(second_buffer).value()};
  auto state = detail::prepare_virtual_pipeline(
      program_state, inputs, std::move(output_buffer).value(),
      ResidencyConfig{});
  if (!state || state.value() == nullptr) {
    std::fprintf(stderr,
                 "DeviceVsm multi Scan prepare backend=%u Q=%llu op=%u "
                 "reason=%u\n",
                 static_cast<unsigned>(backend),
                 static_cast<unsigned long long>(pages),
                 static_cast<unsigned>(operation),
                 static_cast<unsigned>(state.reason()));
    return false;
  }
  prepared.state = std::move(state).value();
  const bool valid =
      prepared.state->geometry.route == detail::VirtualRoute::Scan &&
      prepared.state->geometry.device_vsm_required &&
      prepared.state->input_count == inputs.size() &&
      prepared.state->pipeline != nullptr &&
      prepared.state->alternate_pipeline != nullptr &&
      prepared.state->pipeline->residency == nullptr &&
      prepared.state->alternate_pipeline->residency == nullptr &&
      prepared.state->pipeline->steps.size() == 1u &&
      prepared.state->alternate_pipeline->steps.size() == 1u;
  if (!valid) {
    std::fprintf(
        stderr,
        "DeviceVsm multi Scan shape backend=%u Q=%llu route=%u "
        "inputs=%llu steps=%llu/%llu residency=%u/%u\n",
        static_cast<unsigned>(backend), static_cast<unsigned long long>(pages),
        static_cast<unsigned>(prepared.state->geometry.route),
        static_cast<unsigned long long>(prepared.state->input_count),
        static_cast<unsigned long long>(prepared.state->pipeline->steps.size()),
        static_cast<unsigned long long>(
            prepared.state->alternate_pipeline->steps.size()),
        static_cast<unsigned>(prepared.state->pipeline->residency != nullptr),
        static_cast<unsigned>(prepared.state->alternate_pipeline->residency !=
                              nullptr));
  }
  return valid;
}

bool ExactMultiScanOutput(const PreparedMultiScan &prepared) noexcept {
  if (prepared.output == nullptr || prepared.expected.empty()) {
    return false;
  }
  std::vector<std::byte> bytes(prepared.expected.size() *
                               sizeof(std::uint64_t));
  return prepared.output->observe(bytes) &&
         std::memcmp(bytes.data(), prepared.expected.data(), bytes.size()) == 0;
}

} // namespace rund_node_test_device_vsm_product::multi_scan_test

#endif
