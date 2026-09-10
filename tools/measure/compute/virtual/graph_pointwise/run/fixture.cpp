#include "local.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

namespace rund::measure::compute::virtual_graph_pointwise::run_detail {
namespace {

[[nodiscard]] std::shared_ptr<::rund::compute::VirtualBacking>
make_nonresident_backing() {
  try {
    return std::make_shared<virtual_residency::MemoryBacking>(
        Spec::ElementCount * sizeof(std::uint64_t));
  } catch (...) {
    return {};
  }
}

[[nodiscard]] bool
seed_input(const std::shared_ptr<::rund::compute::VirtualBacking> &backing,
           const std::size_t input) noexcept {
  if (backing == nullptr) {
    return false;
  }
  std::array<std::uint64_t, Spec::ElementCount> values{};
  for (std::size_t index = 0u; index < values.size(); ++index) {
    values[index] = Spec::input_value(input, index);
  }
  return static_cast<bool>(
      backing->write(0u, std::as_bytes(std::span{values})));
}

[[nodiscard]] bool clear_output(
    const std::shared_ptr<::rund::compute::VirtualBacking> &backing) noexcept {
  if (backing == nullptr) {
    return false;
  }
  std::array<std::uint64_t, Spec::ElementCount> values{};
  return static_cast<bool>(
      backing->write(0u, std::as_bytes(std::span{values})));
}

} // namespace

bool prepare(::rund::compute::Device &device, Case &result) {
  auto program = Spec::build(device);
  if (!program) {
    return false;
  }
  const auto fingerprint = program->fingerprint();
  const auto entries = Spec::page_map_entries();
  const ::rund::compute::GraphPageMap page_map{
      .graph_hi = fingerprint.hi,
      .graph_lo = fingerprint.lo,
      .entries = std::span{entries},
  };
  for (std::size_t input = 0u; input < Spec::InputCount; ++input) {
    result.inputs[input] = make_nonresident_backing();
    if (!seed_input(result.inputs[input], input)) {
      return false;
    }
  }
  result.output = make_nonresident_backing();
  if (!clear_output(result.output)) {
    return false;
  }
  auto first = ::rund::compute::virtual_buffer<std::uint64_t>(
      Spec::ElementCount, result.inputs[0u]);
  auto second = ::rund::compute::virtual_buffer<std::uint64_t>(
      Spec::ElementCount, result.inputs[1u]);
  auto third = ::rund::compute::virtual_buffer<std::uint64_t>(
      Spec::ElementCount, result.inputs[2u]);
  auto output = ::rund::compute::virtual_buffer<std::uint64_t>(
      Spec::ElementCount, result.output);
  if (!first || !second || !third || !output) {
    return false;
  }
  auto pipeline = ::rund::compute::virtual_pipeline(*program, *first, *second,
                                                    *third, *output, page_map);
  if (!pipeline) {
    return false;
  }
  result.graph_hi = fingerprint.hi;
  result.graph_lo = fingerprint.lo;
  result.pipeline = std::make_unique<Pipeline>(std::move(pipeline).value());
  return true;
}

} // namespace rund::measure::compute::virtual_graph_pointwise::run_detail
