#pragma once

#include <accel/api.hpp>
#include <accel/runtime.hpp>

#include <cstdint>
#include <type_traits>

namespace rund {

struct AccelIdentity final {
  std::uint64_t graph_id_hi = 0u;
  std::uint64_t graph_id_lo = 0u;
  std::uint64_t kernel_id = 0u;
  AccelApi backend = AccelApi::Auto;
};

struct AccelEvidence final {
  AccelIdentity identity{};
  AccelRunFacts run{};
  AccelOutcome outcome{.reason = "accel_kernel_run_invalid"};
};

static_assert(std::is_standard_layout_v<AccelIdentity>);
static_assert(std::is_standard_layout_v<AccelEvidence>);
static_assert(std::is_trivially_copyable_v<AccelEvidence>);
static_assert(sizeof(void *) != 8u || sizeof(AccelIdentity) == 32u);
static_assert(sizeof(void *) != 8u || sizeof(AccelEvidence) == 408u);

} // namespace rund
