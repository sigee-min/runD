#pragma once

#include <kernel/program/compute/model.hpp>
#include <rund/compute/fixed.hpp>

#include <cstddef>

namespace rund::compute::detail {

[[nodiscard]] inline constexpr std::size_t
type_bytes(const Type type) noexcept {
  switch (type) {
  case Type::I32:
  case Type::U32:
  case Type::FixedLane32:
    return 4u;
  case Type::I64:
  case Type::U64:
  case Type::FixedLane64:
    return 8u;
  }
  return 0u;
}

[[nodiscard]] inline constexpr bool valid_type(const Type type) noexcept {
  switch (type) {
  case Type::I32:
  case Type::U32:
  case Type::I64:
  case Type::U64:
  case Type::FixedLane32:
  case Type::FixedLane64:
    return true;
  }
  return false;
}

[[nodiscard]] inline constexpr kernel::ComputeScalar
type_scalar(const Type type) noexcept {
  switch (type) {
  case Type::I32:
  case Type::U32:
  case Type::FixedLane32:
    return kernel::ComputeScalar::Lane32;
  case Type::I64:
  case Type::U64:
  case Type::FixedLane64:
    return kernel::ComputeScalar::Lane64;
  }
  return static_cast<kernel::ComputeScalar>(0u);
}

[[nodiscard]] inline constexpr kernel::ComputeDomain
type_domain(const Type type) noexcept {
  switch (type) {
  case Type::I32:
    return kernel::ComputeDomain::I32;
  case Type::U32:
    return kernel::ComputeDomain::U32;
  case Type::I64:
    return kernel::ComputeDomain::I64;
  case Type::U64:
    return kernel::ComputeDomain::U64;
  case Type::FixedLane32:
  case Type::FixedLane64:
    return kernel::ComputeDomain::Fixed;
  }
  return static_cast<kernel::ComputeDomain>(0u);
}

} // namespace rund::compute::detail
