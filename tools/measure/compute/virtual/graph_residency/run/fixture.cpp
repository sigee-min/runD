#include "local.hpp"

#include "../../../suite/core.hpp"

#include <chrono>
#include <span>
#include <utility>
#include <vector>

namespace rund::measure::compute::virtual_graph_residency::run_detail {

[[nodiscard]] bool input_digest(const Case &test_case,
                                std::uint64_t &digest) noexcept {
  std::uint64_t value = 1469598103934665603ull;
  std::array<std::uint64_t, Spec::ElementCount> values{};
  for (std::size_t input = 0u; input < Spec::InputCount; ++input) {
    if (test_case.inputs[input] == nullptr ||
        !test_case.inputs[input]->read(
            0u, std::as_writable_bytes(std::span{values}))) {
      return false;
    }
    for (const std::uint64_t item : values) {
      value ^= item;
      value *= 1099511628211ull;
    }
  }
  digest = value == 0u ? 1u : value;
  return true;
}

[[nodiscard]] std::shared_ptr<::rund::compute::VirtualBacking>
make_backing(::rund::compute::Device &device, const Backend backend) {
  if (backend == Backend::Cpu) {
    try {
      return std::make_shared<virtual_residency::MemoryBacking>(
          Spec::ElementCount * sizeof(std::uint64_t));
    } catch (...) {
      return {};
    }
  }
  auto backing = ::rund::compute::resident_virtual_backing<std::uint64_t>(
      device, Spec::ElementCount);
  return backing ? std::move(backing).value()
                 : std::shared_ptr<::rund::compute::VirtualBacking>{};
}

[[nodiscard]] bool
seed_backing(const std::shared_ptr<::rund::compute::VirtualBacking> &backing,
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

[[nodiscard]] bool
read_output(const std::shared_ptr<::rund::compute::VirtualBacking> &backing,
            std::vector<std::uint64_t> &values) noexcept {
  if (backing == nullptr || values.size() != Spec::ElementCount) {
    return false;
  }
  return static_cast<bool>(
      backing->read(0u, std::as_writable_bytes(std::span{values})));
}

[[nodiscard]] bool expected_output(const std::vector<std::uint64_t> &values) {
  if (values.size() != Spec::ElementCount) {
    return false;
  }
  for (std::size_t index = 0u; index < values.size(); ++index) {
    if (values[index] != Spec::expected_value(index)) {
      return false;
    }
  }
  return Spec::hash(std::span{values}) == Spec::expected_hash();
}

[[nodiscard]] bool prepare(::rund::compute::Device &device,
                           const Backend backend, Case &result) {
  auto program = Spec::build(device);
  if (!program || !Spec::validate(*program)) {
    return false;
  }
  for (std::size_t input = 0u; input < Spec::InputCount; ++input) {
    result.inputs[input] = make_backing(device, backend);
    if (!seed_backing(result.inputs[input], input)) {
      return false;
    }
  }
  result.output = make_backing(device, backend);
  if (result.output == nullptr) {
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
  auto pipeline = ::rund::compute::virtual_pipeline(
      *program, *first, *second, *third, *output,
      ::rund::compute::ResidencyConfig{});
  if (!pipeline) {
    return false;
  }
  result.pipeline = std::make_unique<Pipeline>(std::move(pipeline).value());
  return result.pipeline->plan().residency.frame_capacity == 2u;
}

} // namespace rund::measure::compute::virtual_graph_residency::run_detail
