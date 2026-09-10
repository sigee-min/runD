#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace rund_node_test_device_vsm_product::reduce_ops_test {
namespace {

template <class T> [[nodiscard]] bool seed(PreparedReduce &prepared) noexcept {
  std::vector<T> input(static_cast<std::size_t>(prepared.element_count));
  T minimum = std::numeric_limits<T>::max();
  T maximum = 0u;
  std::uint64_t nonzero = 0u;
  for (std::size_t index = 0u; index < input.size(); ++index) {
    const T value =
        index % 11u == 0u ? T{} : static_cast<T>((index * 17u + 3u) % 101u);
    input[index] = value;
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
    nonzero += value != 0u ? 1u : 0u;
  }
  prepared.expected = prepared.operation == rund::kernel::ReduceOp::CountNonzero
                          ? nonzero
                      : prepared.operation == rund::kernel::ReduceOp::Min
                          ? static_cast<std::uint64_t>(minimum)
                          : static_cast<std::uint64_t>(maximum);
  return prepared.input != nullptr &&
         prepared.input->seed(std::as_bytes(std::span{input}));
}

template <class T, class Program>
[[nodiscard]] bool finish_prepare(const std::uint64_t pages, Program &&program,
                                  PreparedReduce &prepared) {
  using namespace rund::compute;
  prepared.input = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(prepared.element_count * sizeof(T)));
  prepared.output = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(sizeof(T));
  prepared.element_bytes = sizeof(T);
  if (!seed<T>(prepared)) {
    return false;
  }
  constexpr detail::Type Type =
      std::is_same_v<T, std::uint32_t> ? detail::Type::U32 : detail::Type::U64;
  auto input = detail::make_virtual_buffer(prepared.element_count, sizeof(T),
                                           Type, {}, prepared.input);
  auto output =
      detail::make_virtual_buffer(1u, sizeof(T), Type, {}, prepared.output);
  const auto program_state =
      program ? detail::ProgramAccess::state(*program) : nullptr;
  auto state =
      program_state != nullptr && input && output
          ? detail::prepare_virtual_pipeline(
                program_state, std::move(input).value(),
                std::move(output).value(), ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!state || state.value() == nullptr) {
    return false;
  }
  prepared.state = std::move(state).value();
  return prepared.state->geometry.route == detail::VirtualRoute::Reduction &&
         prepared.state->pipeline != nullptr &&
         prepared.state->alternate_pipeline != nullptr &&
         prepared.state->pipeline->residency->stream().page_count() == pages &&
         prepared.state->pipeline->residency->stream().frame_capacity() == 2u;
}

template <class T>
[[nodiscard]] bool prepare(const rund::compute::Backend backend,
                           const std::uint64_t pages,
                           const rund::kernel::ReduceOp operation,
                           PreparedReduce &prepared, bool &unavailable) {
  using namespace rund::compute;
  prepared = {};
  prepared.operation = operation;
  prepared.element_count = pages * PageElements - 3u;
  unavailable = false;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto flow = on(*opened).input<T>(PageElements);
  if (operation == rund::kernel::ReduceOp::CountNonzero) {
    auto program = std::move(flow)
                       .branch([](auto values) { return values.count(); })
                       .compile();
    return finish_prepare<T>(pages, std::move(program), prepared);
  }
  if (operation == rund::kernel::ReduceOp::Min) {
    auto program =
        std::move(flow)
            .branch([](auto values) { return values.reduce(Reduce::Min); })
            .compile();
    return finish_prepare<T>(pages, std::move(program), prepared);
  }
  if (operation == rund::kernel::ReduceOp::Max) {
    auto program =
        std::move(flow)
            .branch([](auto values) { return values.reduce(Reduce::Max); })
            .compile();
    return finish_prepare<T>(pages, std::move(program), prepared);
  }
  return false;
}

} // namespace

bool PrepareReduceProduct(const rund::compute::Backend backend,
                          const std::uint64_t pages,
                          const std::uint32_t element_bytes,
                          const rund::kernel::ReduceOp operation,
                          PreparedReduce &prepared, bool &unavailable) {
  if (element_bytes == sizeof(std::uint32_t)) {
    return prepare<std::uint32_t>(backend, pages, operation, prepared,
                                  unavailable);
  }
  return element_bytes == sizeof(std::uint64_t)
             ? prepare<std::uint64_t>(backend, pages, operation, prepared,
                                      unavailable)
             : false;
}

bool ExactReduceOutput(const PreparedReduce &prepared) noexcept {
  if (prepared.output == nullptr) {
    return false;
  }
  std::uint64_t observed = 0u;
  std::span<std::byte> bytes{reinterpret_cast<std::byte *>(&observed),
                             prepared.element_bytes};
  return prepared.output->observe(bytes) && observed == prepared.expected;
}

} // namespace rund_node_test_device_vsm_product::reduce_ops_test

#endif
