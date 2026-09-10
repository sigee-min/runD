#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "../../../../target/selection.hpp"
#include "../../persistent_product/fixture.hpp"

#include "src/compute/virtual/backing.hpp"

#include <node/runtime/compute/access.hpp>
#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <algorithm>
#include <bit>
#include <cstdio>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <utility>

namespace rund_node_test_device_vsm_product::window_test {
namespace {

template <std::uint32_t First, std::size_t Count, class Expression>
[[nodiscard]] constexpr auto add_stage_u32(Expression value) {
  static_assert(Count != 0u);
  if constexpr (Count == 1u) {
    return value + First;
  } else {
    constexpr std::size_t Left = Count / 2u;
    return add_stage_u32<First, Left>(value) +
           add_stage_u32<First + static_cast<std::uint32_t>(Left),
                         Count - Left>(value);
  }
}

template <std::uint32_t First, std::size_t Count>
[[nodiscard]] constexpr std::uint32_t
stage_value_u32(const std::uint32_t value) noexcept {
  static_assert(Count != 0u);
  constexpr std::uint64_t LiteralSum =
      static_cast<std::uint64_t>(Count) * (Count + 1u) / 2u;
  const std::uint64_t result = static_cast<std::uint64_t>(value) * Count +
                               LiteralSum +
                               static_cast<std::uint64_t>(First - 1u) * Count;
  return static_cast<std::uint32_t>(result);
}

[[nodiscard]] std::uint32_t input_value(const std::size_t index) noexcept {
  return static_cast<std::uint32_t>((index * 13u + 5u) % 101u);
}

[[nodiscard]] std::uint32_t combine(const Operation operation,
                                    const std::uint32_t value,
                                    const std::uint32_t sample) noexcept {
  if (operation == Operation::Minimum) {
    return std::min(value, sample);
  }
  if (operation == Operation::Maximum) {
    return std::max(value, sample);
  }
  return value + sample;
}

[[nodiscard]] std::uint32_t identity(const Operation operation) noexcept {
  return operation == Operation::Minimum
             ? std::numeric_limits<std::uint32_t>::max()
             : 0u;
}

[[nodiscard]] rund::compute::Window
public_operation(const Operation operation) noexcept {
  if (operation == Operation::Minimum) {
    return rund::compute::Window::Min;
  }
  if (operation == Operation::Maximum) {
    return rund::compute::Window::Max;
  }
  return rund::compute::Window::Sum;
}

[[nodiscard]] std::uint32_t output_value(const std::size_t index,
                                         const std::size_t count,
                                         const Operation operation,
                                         const Boundary boundary,
                                         const PipelineShape shape) noexcept {
  std::uint32_t value = identity(operation);
  for (std::size_t slot = 0u; slot < Radius * 2u + 1u; ++slot) {
    const std::size_t shifted = index + slot;
    if (boundary == Boundary::Clip &&
        (shifted < Radius || shifted - Radius >= count)) {
      continue;
    }
    const std::size_t selected =
        shifted < Radius ? 0u : std::min(shifted - Radius, count - 1u);
    std::uint32_t sample = input_value(selected);
    if (shape == PipelineShape::MapTripleChainWindowMapTripleChain) {
      sample = stage_value_u32<1u, 260u>(sample);
      sample = stage_value_u32<261u, 260u>(sample);
      sample = stage_value_u32<521u, 260u>(sample);
    } else if (shape == PipelineShape::MapChainWindowMapChain) {
      sample = stage_value_u32<1u, 260u>(sample);
      sample = stage_value_u32<261u, 260u>(sample);
    } else if (shape == PipelineShape::MapDagWindowMapDag) {
      sample = static_cast<std::uint32_t>(sample + 1u) +
               static_cast<std::uint32_t>(sample * 2u);
    } else if (shape == PipelineShape::MapWindowMap) {
      sample += 3u;
    }
    value = combine(operation, value, sample);
  }
  if (shape == PipelineShape::MapTripleChainWindowMapTripleChain) {
    value = stage_value_u32<781u, 260u>(value);
    value = stage_value_u32<1041u, 260u>(value);
    return stage_value_u32<1301u, 260u>(value);
  }
  if (shape == PipelineShape::MapChainWindowMapChain) {
    value = stage_value_u32<521u, 260u>(value);
    return stage_value_u32<781u, 260u>(value);
  }
  if (shape == PipelineShape::MapDagWindowMapDag) {
    return static_cast<std::uint32_t>(value + 5u) +
           static_cast<std::uint32_t>(value * 3u);
  }
  return shape == PipelineShape::MapWindowMap ? value * 2u : value;
}

} // namespace

bool PrepareWindowProduct(
    const rund::compute::Backend backend, const std::uint64_t pages,
    const Operation operation, const Boundary boundary,
    const PipelineShape shape,
    rund_node_test_persistent_product::PreparedProduct &prepared,
    bool &unavailable) {
  using namespace rund::compute;
  unavailable = false;
  const std::size_t elements =
      static_cast<std::size_t>(pages * PayloadElements - 3u);
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto flow = on(*opened).input<std::uint32_t>(FrameElements);
  auto program = [&]() {
    const WindowSpec window{
        .op = public_operation(operation),
        .radius = Radius,
        .edge =
            boundary == Boundary::Clamp ? WindowEdge::Clamp : WindowEdge::Clip,
    };
    if (shape == PipelineShape::MapWindowMap) {
      return std::move(flow)
          .map("device-vsm-window-prefix",
               [](auto value) { return value + 3u; })
          .window(window)
          .map("device-vsm-window-suffix",
               [](auto value) { return value * 2u; })
          .compile();
    }
    if (shape == PipelineShape::MapDagWindowMapDag) {
      return std::move(flow)
          .map("device-vsm-window-prefix-dag",
               [](auto value) { return (value + 1u) + (value * 2u); })
          .window(window)
          .map("device-vsm-window-suffix-dag",
               [](auto value) { return (value + 5u) + (value * 3u); })
          .compile();
    }
    if (shape == PipelineShape::MapChainWindowMapChain) {
      return std::move(flow)
          .map("device-vsm-window-prefix-chain-a",
               [](auto value) { return add_stage_u32<1u, 260u>(value); })
          .map("device-vsm-window-prefix-chain-b",
               [](auto value) { return add_stage_u32<261u, 260u>(value); })
          .window(window)
          .map("device-vsm-window-suffix-chain-a",
               [](auto value) { return add_stage_u32<521u, 260u>(value); })
          .map("device-vsm-window-suffix-chain-b",
               [](auto value) { return add_stage_u32<781u, 260u>(value); })
          .compile();
    }
    if (shape == PipelineShape::MapTripleChainWindowMapTripleChain) {
      return std::move(flow)
          .map("device-vsm-window-prefix-chain-a",
               [](auto value) { return add_stage_u32<1u, 260u>(value); })
          .map("device-vsm-window-prefix-chain-b",
               [](auto value) { return add_stage_u32<261u, 260u>(value); })
          .map("device-vsm-window-prefix-chain-c",
               [](auto value) { return add_stage_u32<521u, 260u>(value); })
          .window(window)
          .map("device-vsm-window-suffix-chain-a",
               [](auto value) { return add_stage_u32<781u, 260u>(value); })
          .map("device-vsm-window-suffix-chain-b",
               [](auto value) { return add_stage_u32<1041u, 260u>(value); })
          .map("device-vsm-window-suffix-chain-c",
               [](auto value) { return add_stage_u32<1301u, 260u>(value); })
          .compile();
    }
    return std::move(flow)
        .branch([window](auto values) { return values.window(window); })
        .compile();
  }();
  prepared.input = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      elements * sizeof(std::uint32_t));
  prepared.output = std::make_shared<
      rund_node_test_persistent_product::PersistentProductBacking>(
      elements * sizeof(std::uint32_t));
  std::vector<std::uint32_t> seeded(elements);
  prepared.expected.resize(elements);
  for (std::size_t index = 0u; index < elements; ++index) {
    seeded[index] = input_value(index);
    prepared.expected[index] =
        output_value(index, elements, operation, boundary, shape);
  }
  const bool seeded_ok = prepared.input->seed(std::as_bytes(std::span{seeded}));
  auto input = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {}, prepared.input);
  auto output = detail::make_virtual_buffer(
      elements, sizeof(std::uint32_t), detail::Type::U32, {}, prepared.output);
  auto state =
      program && input && output
          ? detail::prepare_virtual_pipeline(
                detail::ProgramAccess::state(*program),
                std::move(input).value(), std::move(output).value(),
                ResidencyConfig{})
          : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                Reason::PipelineInvalid);
  if (!seeded_ok || !state || state.value() == nullptr) {
    std::fprintf(stderr,
                 "DeviceVsm Window prepare failed seeded=%u state=%u reason=%u "
                 "op=%u boundary=%u shape=%u Q=%llu\n",
                 static_cast<unsigned>(seeded_ok),
                 static_cast<unsigned>(static_cast<bool>(state)),
                 static_cast<unsigned>(state.reason()),
                 static_cast<unsigned>(operation),
                 static_cast<unsigned>(boundary), static_cast<unsigned>(shape),
                 static_cast<unsigned long long>(pages));
    return false;
  }
  prepared.state = std::move(state).value();
  const auto &stream = prepared.state->pipeline->residency->stream();
  const bool valid =
      prepared.state->geometry.route == detail::VirtualRoute::Window &&
      stream.page_count() == pages && stream.frame_capacity() == 2u;
  if (!valid) {
    std::fprintf(stderr,
                 "DeviceVsm Window geometry failed route=%u pages=%llu/%llu "
                 "frames=%llu op=%u boundary=%u shape=%u\n",
                 static_cast<unsigned>(prepared.state->geometry.route),
                 static_cast<unsigned long long>(stream.page_count()),
                 static_cast<unsigned long long>(pages),
                 static_cast<unsigned long long>(stream.frame_capacity()),
                 static_cast<unsigned>(operation),
                 static_cast<unsigned>(boundary), static_cast<unsigned>(shape));
  }
  return valid;
}

bool PrepareWindowRingProduct(
    const rund::compute::Backend backend,
    rund_node_test_persistent_product::PreparedProduct &prepared,
    bool &unavailable) {
  if (!PrepareWindowProduct(backend, WindowRingPages, Operation::Sum,
                            Boundary::Clamp, PipelineShape::Window, prepared,
                            unavailable)) {
    return unavailable;
  }
  return prepared.state != nullptr &&
         !prepared.state->geometry.device_vsm_required;
}

namespace {

[[nodiscard]] std::int32_t i32_seed(const std::size_t index) noexcept {
  switch (index % 11u) {
  case 0u:
    return std::numeric_limits<std::int32_t>::max();
  case 1u:
    return std::numeric_limits<std::int32_t>::min();
  case 2u:
    return 0;
  default:
    return static_cast<std::int32_t>((index % 401u) * 37u % 401u) - 200;
  }
}

[[nodiscard]] std::int32_t wrap_add(const std::int32_t left,
                                    const std::int32_t right) noexcept {
  return std::bit_cast<std::int32_t>(
      std::bit_cast<std::uint32_t>(left) +
      std::bit_cast<std::uint32_t>(right));
}

[[nodiscard]] std::int32_t i32_window(const std::span<const std::int32_t> input,
                                      const std::size_t index) noexcept {
  std::int32_t result = 0;
  for (std::size_t delta = 0u; delta <= Radius * 2u; ++delta) {
    const std::size_t raw = index + delta;
    const std::size_t selected =
        raw < Radius ? 0u : std::min(raw - Radius, input.size() - 1u);
    result = wrap_add(result, input[selected]);
  }
  return result;
}

} // namespace

bool PrepareI32WindowProduct(const rund::compute::Backend backend,
                             const std::uint64_t pages, const bool resident,
                             I32WindowProduct &prepared, bool &unavailable) {
  using namespace rund::compute;
  unavailable = false;
  if (pages < 2u || pages > std::numeric_limits<std::size_t>::max() /
                              PayloadElements) {
    return false;
  }
  const std::size_t elements =
      static_cast<std::size_t>(pages * PayloadElements - 3u);
  auto opened = open(rund::node::test_contract::target_for(backend));
  if (!opened) {
    unavailable = opened.reason() == Reason::AdapterUnavailable;
    return false;
  }
  auto flow = on(*opened).input<std::int32_t>(FrameElements);
  auto program = std::move(flow)
                     .branch([](auto values) {
                       return values.window(WindowSpec{
                           .op = Window::Sum,
                           .radius = Radius,
                           .edge = WindowEdge::Clamp});
                     })
                     .compile();
  if (!program) {
    return false;
  }

  try {
    if (elements > std::numeric_limits<std::size_t>::max() /
                       sizeof(std::int32_t)) {
      return false;
    }
    const std::size_t bytes = elements * sizeof(std::int32_t);
    if (resident) {
      auto input = resident_virtual_backing<std::int32_t>(*opened, elements);
      auto output = resident_virtual_backing<std::int32_t>(*opened, elements);
      if (!input || !output) {
        return false;
      }
      prepared.input = std::move(input).value();
      prepared.output = std::move(output).value();
    } else {
      prepared.input = std::make_shared<
          rund_node_test_persistent_product::PersistentProductBacking>(bytes);
      prepared.output = std::make_shared<
          rund_node_test_persistent_product::PersistentProductBacking>(bytes);
    }
    std::vector<std::int32_t> seeded(elements);
    std::vector<std::int32_t> cleared(elements, 0);
    prepared.expected.resize(elements);
    for (std::size_t index = 0u; index < elements; ++index) {
      seeded[index] = i32_seed(index);
    }
    for (std::size_t index = 0u; index < elements; ++index) {
      prepared.expected[index] =
          i32_window(std::span<const std::int32_t>{seeded}, index);
    }
    const bool seeded_ok =
        prepared.input != nullptr && prepared.output != nullptr &&
        static_cast<bool>(prepared.input->write(
            0u, std::as_bytes(std::span{seeded}))) &&
        static_cast<bool>(prepared.output->write(
            0u, std::as_bytes(std::span{cleared})));
    auto input = detail::make_virtual_buffer(
        elements, sizeof(std::int32_t), detail::Type::I32, {}, prepared.input);
    auto output = detail::make_virtual_buffer(
        elements, sizeof(std::int32_t), detail::Type::I32, {}, prepared.output);
    auto state =
        seeded_ok && input && output
            ? detail::prepare_virtual_pipeline(
                  detail::ProgramAccess::state(*program),
                  std::move(input).value(), std::move(output).value(),
                  ResidencyConfig{})
            : Result<std::shared_ptr<detail::VirtualPipelineState>>::fail(
                  Reason::PipelineInvalid);
    if (!state || state.value() == nullptr) {
      return false;
    }
    prepared.state = std::move(state).value();
    prepared.resident = resident;
    if (prepared.state->pipeline == nullptr ||
        prepared.state->pipeline->residency == nullptr) {
      return false;
    }
    const auto &stream = prepared.state->pipeline->residency->stream();
    return prepared.state->geometry.route == detail::VirtualRoute::Window &&
           stream.page_count() == pages && stream.frame_capacity() == 2u;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool ExactI32WindowOutput(const I32WindowProduct &prepared) noexcept {
  if (prepared.output == nullptr || prepared.expected.empty()) {
    return false;
  }
  try {
    std::vector<std::int32_t> observed(prepared.expected.size());
    if (!static_cast<bool>(prepared.output->read(
            0u, std::as_writable_bytes(std::span{observed})))) {
      return false;
    }
    return observed == prepared.expected;
  } catch (const std::bad_alloc &) {
    return false;
  }
}

bool ExactWindowOutput(const rund_node_test_persistent_product::PreparedProduct
                           &prepared) noexcept {
  std::vector<std::uint32_t> observed(prepared.expected.size());
  if (!prepared.output->observe(std::as_writable_bytes(std::span{observed}))) {
    return false;
  }
  for (std::size_t index = 0u; index < observed.size(); ++index) {
    if (observed[index] != prepared.expected[index]) {
      std::fprintf(stderr,
                   "DeviceVsm Window mismatch index=%zu observed=%u "
                   "expected=%u\n",
                   index, observed[index], prepared.expected[index]);
      return false;
    }
  }
  return true;
}

} // namespace rund_node_test_device_vsm_product::window_test

#endif
