#pragma once

#include "../type.hpp"

#include <kernel/program/compute/scan/model.hpp>

#include <optional>

namespace rund::compute::detail::graph_detail {

[[nodiscard]] inline constexpr std::optional<kernel::ScanElement>
scan_element(const Type type) noexcept {
  switch (type) {
  case Type::I32:
  case Type::U32:
  case Type::FixedLane32:
    return kernel::ScanElement::U32;
  case Type::I64:
  case Type::U64:
  case Type::FixedLane64:
    return kernel::ScanElement::U64;
  }
  return std::nullopt;
}

} // namespace rund::compute::detail::graph_detail
