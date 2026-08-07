#pragma once

#include <kernel/program/compute/model.hpp>
#include <rund/compute/fixed.hpp>

#include <cstddef>

namespace rund::compute::detail {

struct TypeProjection final {
  std::size_t bytes{};
  kernel::ComputeScalar scalar{};
  kernel::ComputeDomain domain{};
};

[[nodiscard]] inline constexpr TypeProjection
project_type(const Type type) noexcept {
  switch (type) {
  case Type::I32:
    return {4u, kernel::ComputeScalar::Lane32, kernel::ComputeDomain::I32};
  case Type::U32:
    return {4u, kernel::ComputeScalar::Lane32, kernel::ComputeDomain::U32};
  case Type::I64:
    return {8u, kernel::ComputeScalar::Lane64, kernel::ComputeDomain::I64};
  case Type::U64:
    return {8u, kernel::ComputeScalar::Lane64, kernel::ComputeDomain::U64};
  case Type::FixedLane32:
    return {4u, kernel::ComputeScalar::Lane32, kernel::ComputeDomain::Fixed};
  case Type::FixedLane64:
    return {8u, kernel::ComputeScalar::Lane64, kernel::ComputeDomain::Fixed};
  }
  return {};
}

[[nodiscard]] inline constexpr std::size_t
type_bytes(const Type type) noexcept {
  return project_type(type).bytes;
}

[[nodiscard]] inline constexpr bool valid_type(const Type type) noexcept {
  return project_type(type).bytes != 0u;
}

[[nodiscard]] inline constexpr kernel::ComputeScalar
type_scalar(const Type type) noexcept {
  return project_type(type).scalar;
}

[[nodiscard]] inline constexpr kernel::ComputeDomain
type_domain(const Type type) noexcept {
  return project_type(type).domain;
}

[[nodiscard]] inline constexpr bool type_fixed(const Type type) noexcept {
  return type_domain(type) == kernel::ComputeDomain::Fixed;
}

} // namespace rund::compute::detail
