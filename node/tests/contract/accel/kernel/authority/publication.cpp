#include "src/accel/kernel/publication.hpp"

#include <kernel/program/compute/binding/model.hpp>

#include <cstdint>

#include "hooks.hpp"
#include "publication.hpp"

namespace node_accel_contract {

[[nodiscard]] bool PreparedPublicationFingerprintIsSemantic() {
  using namespace rund::node::accel::detail;
  PreparedKernelPublicationIdentity publication{};
  publication.sources[0] = PreparedKernelPublicationViewIdentity{
      .backing_bytes = 4096u,
      .offset_bytes = 64u,
      .count = 16u,
      .stride_bytes = 8u,
      .element_bytes = 4u,
      .resource_ordinal = 3u,
      .usage = rund::kernel::kResidentUsageRead,
  };
  publication.sources[1] = publication.sources[0];
  publication.sources[2] = publication.sources[0];
  publication.target = PreparedKernelPublicationViewIdentity{
      .backing_bytes = 8192u,
      .offset_bytes = 128u,
      .count = 16u,
      .stride_bytes = 4u,
      .element_bytes = 4u,
      .resource_ordinal = 7u,
      .usage = rund::kernel::kResidentUsageWrite,
  };
  publication.state = 5u;
  publication.final = 2u;

  std::uint64_t expected_hi = 0u;
  std::uint64_t expected_lo = 0u;
  SeedPreparedKernelPublicationFingerprint(expected_hi, expected_lo);
  MixPreparedKernelPublicationFingerprint(expected_hi, expected_lo,
                                          publication);

  PreparedKernelPublicationIdentity tampered = publication;
  ++tampered.sources[1].offset_bytes;
  std::uint64_t tampered_hi = 0u;
  std::uint64_t tampered_lo = 0u;
  SeedPreparedKernelPublicationFingerprint(tampered_hi, tampered_lo);
  MixPreparedKernelPublicationFingerprint(tampered_hi, tampered_lo, tampered);
  if (tampered_hi == expected_hi && tampered_lo == expected_lo) {
    return false;
  }

  tampered = publication;
  ++tampered.target.resource_ordinal;
  tampered_hi = 0u;
  tampered_lo = 0u;
  SeedPreparedKernelPublicationFingerprint(tampered_hi, tampered_lo);
  MixPreparedKernelPublicationFingerprint(tampered_hi, tampered_lo, tampered);
  if (tampered_hi == expected_hi && tampered_lo == expected_lo) {
    return false;
  }

  tampered = publication;
  ++tampered.outer_bound;
  tampered_hi = 0u;
  tampered_lo = 0u;
  SeedPreparedKernelPublicationFingerprint(tampered_hi, tampered_lo);
  MixPreparedKernelPublicationFingerprint(tampered_hi, tampered_lo, tampered);
  return tampered_hi != expected_hi || tampered_lo != expected_lo;
}

[[nodiscard]] bool PreparedPublicationResolvedIdentityIsExact() {
  using namespace rund::node::accel::detail;
  constexpr PreparedKernelPublicationViewIdentity identity{
      .resident_id = 17u,
      .backing_bytes = 4096u,
      .offset_bytes = 64u,
      .count = 16u,
      .stride_bytes = 8u,
      .element_bytes = 4u,
      .resource_ordinal = 3u,
      .usage = rund::kernel::kResidentUsageRead,
  };
  constexpr rund::kernel::ResidentBufferRef exact{
      .id = 17u,
      .bytes = 4096u,
      .offset_bytes = 64u,
      .element_bytes = 4u,
      .stride_bytes = 8u,
      .count = 16u,
      .usage = rund::kernel::kResidentUsageRead,
  };
  auto swapped = exact;
  swapped.id = 19u;
  auto zero = exact;
  zero.id = 0u;
  return PreparedPublicationViewMatchesForContract(exact, identity) &&
         !PreparedPublicationViewMatchesForContract(swapped, identity) &&
         !PreparedPublicationViewMatchesForContract(zero, identity);
}

} // namespace node_accel_contract
