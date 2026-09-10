#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>

#include <array>
#include <cstring>
#include <span>
#include <vector>

namespace rund_node_test_device_vsm_product::binary_test {

bool PrepareBinaryProduct(const rund::compute::Backend backend,
                          const std::uint64_t pages, PreparedBinary &prepared,
                          bool &unavailable) {
  using namespace rund::compute;
  prepared = {};
  unavailable = false;
  const std::uint64_t elements = pages * PageElements - 3u;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto program =
      on(*opened)
          .input<std::uint32_t>(PageElements)
          .zip_input<std::uint32_t>(PageElements)
          .map("device-vsm-binary-product",
               [](auto first, auto second) { return first + second; })
          .compile();
  prepared.first = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint32_t)));
  prepared.second = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint32_t)));
  prepared.output = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(std::uint32_t)));
  prepared.element_count = elements;
  std::vector<std::uint32_t> first(static_cast<std::size_t>(elements));
  std::vector<std::uint32_t> second(first.size());
  std::vector<std::uint32_t> expected(first.size());
  for (std::size_t index = 0u; index < first.size(); ++index) {
    first[index] = static_cast<std::uint32_t>(index * 3u + pages);
    second[index] = static_cast<std::uint32_t>(index * 5u + 7u);
    expected[index] = first[index] + second[index];
  }
  const std::span<const std::byte> expected_bytes =
      std::as_bytes(std::span{expected});
  prepared.expected.assign(expected_bytes.begin(), expected_bytes.end());
  if (prepared.first == nullptr || prepared.second == nullptr ||
      prepared.output == nullptr ||
      !prepared.first->seed(std::as_bytes(std::span{first})) ||
      !prepared.second->seed(std::as_bytes(std::span{second}))) {
    return false;
  }
  auto first_buffer = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {}, prepared.first);
  auto second_buffer = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {}, prepared.second);
  auto output_buffer = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {}, prepared.output);
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
    return false;
  }
  prepared.state = std::move(state).value();
  return prepared.state->geometry.route ==
             detail::VirtualRoute::MultiPointwise &&
         prepared.state->input_count == inputs.size() &&
         prepared.state->pipeline != nullptr &&
         prepared.state->alternate_pipeline != nullptr &&
         prepared.state->pipeline->residency == nullptr &&
         prepared.state->alternate_pipeline->residency == nullptr;
}

bool ExactBinaryOutput(const PreparedBinary &prepared) noexcept {
  if (prepared.output == nullptr || prepared.expected.empty()) {
    return false;
  }
  std::vector<std::byte> bytes(prepared.expected.size());
  return prepared.output->observe(bytes) &&
         std::memcmp(bytes.data(), prepared.expected.data(), bytes.size()) == 0;
}

} // namespace rund_node_test_device_vsm_product::binary_test

#endif
