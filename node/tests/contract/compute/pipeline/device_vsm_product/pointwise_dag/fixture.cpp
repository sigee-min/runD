#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <cstring>
#include <span>
#include <type_traits>
#include <utility>

namespace rund_node_test_device_vsm_product::pointwise_dag_test {
namespace {

template <class T>
[[nodiscard]] bool
PrepareTyped(const rund::compute::Backend backend, const std::uint64_t pages,
             PreparedPointwiseDag &prepared, bool &unavailable) {
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
          .input<T>(PageElements)
          .branch([](auto values) {
            const auto left =
                values.map("device-vsm-pointwise-dag-left",
                           [](auto value) { return value + T{1u}; });
            const auto right =
                values.map("device-vsm-pointwise-dag-right",
                           [](auto value) { return value * T{2u}; });
            return zip(left, right)
                .map("device-vsm-pointwise-dag-join",
                     [](auto first, auto second) { return first + second; });
          })
          .compile();
  prepared.input = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(T)));
  prepared.output = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(T)));
  prepared.element_count = elements;
  prepared.element_bytes = sizeof(T);
  prepared.type = std::is_same_v<T, std::uint32_t> ? PointwiseDagType::U32
                                                   : PointwiseDagType::U64;
  std::vector<T> input(static_cast<std::size_t>(elements));
  std::vector<T> expected(input.size());
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<T>(index * 17u + 3u);
    expected[index] = static_cast<T>(static_cast<T>(input[index] + T{1u}) +
                                     static_cast<T>(input[index] * T{2u}));
  }
  const std::span<const std::byte> expected_bytes =
      std::as_bytes(std::span{expected});
  prepared.expected.assign(expected_bytes.begin(), expected_bytes.end());
  if (prepared.input == nullptr || prepared.output == nullptr ||
      !prepared.input->seed(std::as_bytes(std::span{input}))) {
    return false;
  }
  auto input_buffer = detail::make_virtual_buffer(
      elements, sizeof(T),
      std::is_same_v<T, std::uint32_t> ? detail::Type::U32 : detail::Type::U64,
      {}, prepared.input);
  auto output_buffer = detail::make_virtual_buffer(
      elements, sizeof(T),
      std::is_same_v<T, std::uint32_t> ? detail::Type::U32 : detail::Type::U64,
      {}, prepared.output);
  const auto program_state =
      program ? detail::ProgramAccess::state(*program) : nullptr;
  auto state =
      program_state != nullptr && input_buffer && output_buffer
          ? detail::prepare_virtual_pipeline(
                program_state, std::move(input_buffer).value(),
                std::move(output_buffer).value(), ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!state || state.value() == nullptr) {
    return false;
  }
  prepared.state = std::move(state).value();
  return prepared.state->geometry.route == detail::VirtualRoute::Pointwise &&
         prepared.state->pipeline != nullptr &&
         prepared.state->alternate_pipeline != nullptr &&
         prepared.state->pipeline->residency->stream().page_count() == pages &&
         prepared.state->pipeline->residency->stream().frame_capacity() == 2u;
}

} // namespace

bool PreparePointwiseDagProduct(const rund::compute::Backend backend,
                                const std::uint64_t pages,
                                const PointwiseDagType type,
                                PreparedPointwiseDag &prepared,
                                bool &unavailable) {
  return type == PointwiseDagType::U32
             ? PrepareTyped<std::uint32_t>(backend, pages, prepared,
                                           unavailable)
             : PrepareTyped<std::uint64_t>(backend, pages, prepared,
                                           unavailable);
}

bool ExactPointwiseDagOutput(const PreparedPointwiseDag &prepared) noexcept {
  if (prepared.output == nullptr || prepared.expected.empty()) {
    return false;
  }
  std::vector<std::byte> bytes(prepared.expected.size());
  return prepared.output->observe(bytes) &&
         std::memcmp(bytes.data(), prepared.expected.data(), bytes.size()) == 0;
}

} // namespace rund_node_test_device_vsm_product::pointwise_dag_test

#endif
