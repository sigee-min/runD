#include "local.hpp"

#include <array>
#include <cstddef>
#include <cstdio>

namespace rund_node_collective_modes {

namespace {

struct DomainRow final {
  Domain domain;
  const char *name;
  std::size_t width;
  std::size_t evidence;
};

constexpr std::array kDomains{
    DomainRow{Domain::I32, "i32", 4u, 0u},
    DomainRow{Domain::U32, "u32", 4u, 1u},
    DomainRow{Domain::I64, "i64", 8u, 2u},
    DomainRow{Domain::U64, "u64", 8u, 3u},
    DomainRow{Domain::Fixed16x16, "fixed-16-16", 4u, 4u},
    DomainRow{Domain::Fixed20x44, "fixed-20-44", 8u, 5u},
};

[[nodiscard]] bool CheckFamily(const bool result,
                               const rund::compute::Backend backend,
                               const std::size_t width,
                               const char *const family) {
  if (!result) {
    std::fprintf(stderr,
                 "compute modes domain failed backend=%u width=%zu family=%s\n",
                 static_cast<unsigned>(backend), width, family);
  }
  return result;
}

[[nodiscard]] bool CheckDomain(const rund::compute::Backend backend,
                               DomainEvidence &evidence, const DomainRow row) {
  return CheckFamily(CheckCrossBlock(backend, evidence, row.domain), backend,
                     row.width, "cross-block-cancellation") &&
         CheckFamily(CheckCore(backend, evidence, row.domain), backend,
                     row.width, "all/clip/extrema") &&
         CheckFamily(CheckBounded(backend, evidence, row.domain), backend,
                     row.width, "bounded/bounded-window") &&
         CheckFamily(CheckEmpty(backend, evidence, row.domain), backend,
                     row.width, "empty") &&
         CheckFamily(CheckReductionCancellation(backend, evidence, row.domain),
                     backend, row.width, "reduction-cancellation") &&
         CheckFamily(CheckScale(backend, evidence, row.domain), backend,
                     row.width, "reduction-scale") &&
         CheckFamily(CheckReductionOverflow(backend, evidence, row.domain),
                     backend, row.width, "reduction-overflow") &&
         CheckFamily(CheckExclusive(backend, evidence, row.domain), backend,
                     row.width, "exclusive-overflow/segmented-carry-overflow");
}

void BeginMode(const char *const label, const rund::compute::Backend backend) {
  std::fprintf(stderr, "compute modes begin backend=%u family=%s\n",
               static_cast<unsigned>(backend), label);
}

[[nodiscard]] bool CheckMode(const char *const label, const bool result,
                             const rund::compute::Backend backend) {
  if (!result) {
    std::fprintf(stderr, "compute modes failed backend=%u family=%s\n",
                 static_cast<unsigned>(backend), label);
  }
  return result;
}

} // namespace

[[nodiscard]] bool CheckBackend(const rund::compute::Backend backend,
                                Evidence &evidence) {
  for (const DomainRow row : kDomains) {
    BeginMode(row.name, backend);
    if (!CheckMode(row.name,
                   CheckDomain(backend, evidence.domain[row.evidence], row),
                   backend)) {
      return false;
    }
  }
  constexpr std::array overflow{
      DomainRow{Domain::I32, "inclusive-overflow-i32", 4u, 0u},
      DomainRow{Domain::U32, "inclusive-overflow-u32", 4u, 1u},
      DomainRow{Domain::I64, "inclusive-overflow-i64", 8u, 2u},
      DomainRow{Domain::U64, "inclusive-overflow-u64", 8u, 3u},
      DomainRow{Domain::Lane32, "inclusive-overflow-fixed-lane32", 4u, 4u},
      DomainRow{Domain::Lane64, "inclusive-overflow-fixed-lane64", 8u, 5u},
  };
  for (const DomainRow row : overflow) {
    BeginMode(row.name, backend);
    if (!CheckMode(row.name, CheckInclusive(backend, evidence, row.domain),
                   backend)) {
      return false;
    }
  }

  BeginMode("segmented-reset-overflow-u32", backend);
  if (!CheckMode("segmented-reset-overflow-u32",
                 CheckReset(backend, evidence, Domain::U32), backend)) {
    return false;
  }
  BeginMode("segmented-reset-overflow-u64", backend);
  if (!CheckMode("segmented-reset-overflow-u64",
                 CheckReset(backend, evidence, Domain::U64), backend)) {
    return false;
  }
  BeginMode("segmented-heads", backend);
  if (!CheckMode("segmented-heads", CheckHeads(backend, evidence), backend)) {
    return false;
  }
  BeginMode("reduce-overflow-fixed-lane32", backend);
  if (!CheckMode("reduce-overflow-fixed-lane32",
                 CheckReductionOverflow(backend, evidence.lane_overflow[0u],
                                        Domain::Lane32),
                 backend)) {
    return false;
  }
  BeginMode("reduce-overflow-fixed-lane64", backend);
  if (!CheckMode("reduce-overflow-fixed-lane64",
                 CheckReductionOverflow(backend, evidence.lane_overflow[1u],
                                        Domain::Lane64),
                 backend)) {
    return false;
  }
  return true;
}

} // namespace rund_node_collective_modes
