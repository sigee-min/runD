#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <cstring>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace rund_node_test_device_vsm_product::reduce_sum_test {
namespace {

template <class T>
[[nodiscard]] bool seed(PreparedReduce &prepared,
                        const bool overflow) noexcept {
  std::vector<T> input(static_cast<std::size_t>(prepared.element_count));
  unsigned __int128 total = 0u;
  for (std::size_t index = 0u; index < input.size(); ++index) {
    input[index] = static_cast<T>((index * 17u + 3u) % 101u);
    total += input[index];
  }
  if (overflow) {
    std::fill(input.begin(), input.end(), T{});
    const std::size_t first = static_cast<std::size_t>(2u * PageElements);
    if (first + 1u >= input.size()) {
      return false;
    }
    input[first] = std::numeric_limits<T>::max();
    input[first + 1u] = 1u;
    total = static_cast<unsigned __int128>(std::numeric_limits<T>::max()) + 1u;
  }
  prepared.expected = static_cast<std::uint64_t>(total);
  return prepared.input != nullptr &&
         prepared.input->seed(std::as_bytes(std::span{input}));
}

template <class T>
[[nodiscard]] bool prepare(const rund::compute::Backend backend,
                           const std::uint64_t pages, const bool overflow,
                           PreparedReduce &prepared, bool &unavailable) {
  using namespace rund::compute;
  prepared = {};
  unavailable = false;
  const std::uint64_t elements = pages * PageElements - 3u;
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto flow = on(*opened).input<T>(PageElements);
  auto program =
      std::move(flow)
          .branch([](auto values) { return values.reduce(Reduce::Sum); })
          .compile();
  prepared.input = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      static_cast<std::size_t>(elements * sizeof(T)));
  prepared.output = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(sizeof(T));
  prepared.element_count = elements;
  prepared.element_bytes = sizeof(T);
  if (!seed<T>(prepared, overflow)) {
    return false;
  }
  constexpr detail::Type Type =
      std::is_same_v<T, std::uint32_t> ? detail::Type::U32 : detail::Type::U64;
  auto input = detail::make_virtual_buffer(elements, sizeof(T), Type, {},
                                           prepared.input);
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

} // namespace

bool PrepareReduceProduct(const rund::compute::Backend backend,
                          const std::uint64_t pages,
                          const std::uint32_t element_bytes,
                          const bool overflow, PreparedReduce &prepared,
                          bool &unavailable) {
  if (element_bytes == sizeof(std::uint32_t)) {
    return prepare<std::uint32_t>(backend, pages, overflow, prepared,
                                  unavailable);
  }
  return element_bytes == sizeof(std::uint64_t)
             ? prepare<std::uint64_t>(backend, pages, overflow, prepared,
                                      unavailable)
             : false;
}

bool SeedSafe(PreparedReduce &prepared) noexcept {
  return prepared.element_bytes == sizeof(std::uint32_t)
             ? seed<std::uint32_t>(prepared, false)
         : prepared.element_bytes == sizeof(std::uint64_t)
             ? seed<std::uint64_t>(prepared, false)
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

} // namespace rund_node_test_device_vsm_product::reduce_sum_test

#endif
