#include "local.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rund::measure::compute::virtual_graph_pointwise::run_detail {

bool input_digest(const Case &test_case, std::uint64_t &digest) noexcept {
  std::uint64_t value = 1469598103934665603ull;
  std::array<std::uint64_t, Spec::ElementCount> values{};
  for (const auto &backing : test_case.inputs) {
    if (backing == nullptr ||
        !backing->read(0u, std::as_writable_bytes(std::span{values}))) {
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

bool read_output(
    const std::shared_ptr<::rund::compute::VirtualBacking> &backing,
    std::vector<std::uint64_t> &values) noexcept {
  return backing != nullptr && values.size() == Spec::ElementCount &&
         static_cast<bool>(
             backing->read(0u, std::as_writable_bytes(std::span{values})));
}

bool expected_output(const std::vector<std::uint64_t> &values) noexcept {
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

} // namespace rund::measure::compute::virtual_graph_pointwise::run_detail
