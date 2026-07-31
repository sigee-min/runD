#pragma once

#include <accel/api.hpp>
#include <accel/check.hpp>
#include <accel/kernel/check.hpp>

#include <node/accel/context.hpp>

#include "../../backend/token.hpp"

#include <kernel/program/compute/model.hpp>

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

struct ContextAdmission {
  rund::AccelCheck check{};
  std::uint64_t context_id = 0u;
  rund::AccelApi api = rund::AccelApi::Auto;
  rund::kernel::ComputeCaps caps{};
  std::shared_ptr<void> owner{};
  std::shared_ptr<PickToken> pick{};
};

struct KernelAdmission {
  rund::AccelKernelCheck check{};
  std::uint64_t kernel_id = 0u;
  std::uint64_t context_id = 0u;
  std::uint64_t graph_id_hi = 0u;
  std::uint64_t graph_id_lo = 0u;
  std::uint64_t node_count = 0u;
  rund::AccelApi api = rund::AccelApi::Auto;
  rund::kernel::ComputeScalar scalar = rund::kernel::ComputeScalar::Lane32;
  rund::kernel::ComputeDomain domain = rund::kernel::ComputeDomain::Fixed;
  rund::kernel::ComputeCaps frozen_caps{};
  std::shared_ptr<void> owner{};
};

} // namespace rund::node::accel::detail
