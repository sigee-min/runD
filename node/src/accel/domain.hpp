#pragma once

#include <kernel/program/compute/model.hpp>

namespace rund::node::accel::detail {

[[nodiscard]] constexpr bool
IsSignedDomain(const rund::kernel::ComputeDomain domain) noexcept {
  return domain == rund::kernel::ComputeDomain::I32 ||
         domain == rund::kernel::ComputeDomain::I64 ||
         domain == rund::kernel::ComputeDomain::Fixed;
}

static_assert(IsSignedDomain(rund::kernel::ComputeDomain::I32));
static_assert(!IsSignedDomain(rund::kernel::ComputeDomain::U32));
static_assert(IsSignedDomain(rund::kernel::ComputeDomain::I64));
static_assert(!IsSignedDomain(rund::kernel::ComputeDomain::U64));
static_assert(IsSignedDomain(rund::kernel::ComputeDomain::Fixed));

} // namespace rund::node::accel::detail
