#include "internal.hpp"

#include "../../../../../../hash/fnv.hpp"

namespace rund::node::accel::detail::device_vsm_window_source {

rund::kernel::ComputeDomain
executable_domain(const RangeExec &execution) noexcept {
  if (execution.domain() == rund::kernel::ComputeDomain::I32 &&
      !execution.wide_elements()) {
    return rund::kernel::ComputeDomain::I32;
  }
  if (execution.domain() == rund::kernel::ComputeDomain::I64 &&
      execution.wide_elements()) {
    return rund::kernel::ComputeDomain::I64;
  }
  if (execution.signed_values()) {
    return execution.wide_elements() ? rund::kernel::ComputeDomain::I64
                                     : rund::kernel::ComputeDomain::I32;
  }
  return execution.wide_elements() ? rund::kernel::ComputeDomain::U64
                                   : rund::kernel::ComputeDomain::U32;
}

rund::kernel::ComputeApi api_for(const RangeSource source) noexcept {
  return source == RangeSource::Metal ? rund::kernel::ComputeApi::Metal
                                      : rund::kernel::ComputeApi::Vulkan;
}

rund::kernel::ArtifactKey
artifact_key(const RangeIdentity source, const RangeIdentity physical,
             const rund::kernel::ComputeApi api,
             const rund::kernel::ComputeScalar scalar,
             const rund::kernel::ComputeDomain domain,
             const DeviceVsmWindowFusion &fusion,
             const DeviceVsmWindowRingPlan &ring) noexcept {
  ::rund::node::hash_detail::Fnv hi{
      ::rund::node::hash_detail::kFnvStandardOffset};
  ::rund::node::hash_detail::Fnv lo{};
  const auto mix = [&](const std::uint64_t value) noexcept {
    hi.Number(value);
    lo.Number(value ^ 0x9e3779b97f4a7c15ull);
  };
  mix(source.hi);
  mix(source.lo);
  mix(physical.hi);
  mix(physical.lo);
  mix(static_cast<std::uint64_t>(scalar));
  mix(static_cast<std::uint64_t>(domain));
  mix(static_cast<std::uint64_t>(fusion.before.kind));
  mix(fusion.before.immediate);
  mix(fusion.before.source_hi);
  mix(fusion.before.source_lo);
  mix(fusion.before.canonical_hi);
  mix(fusion.before.canonical_lo);
  mix(static_cast<std::uint64_t>(fusion.before_second.kind));
  mix(fusion.before_second.immediate);
  mix(fusion.before_second.source_hi);
  mix(fusion.before_second.source_lo);
  mix(fusion.before_second.canonical_hi);
  mix(fusion.before_second.canonical_lo);
  mix(static_cast<std::uint64_t>(fusion.before_third.kind));
  mix(fusion.before_third.immediate);
  mix(fusion.before_third.source_hi);
  mix(fusion.before_third.source_lo);
  mix(fusion.before_third.canonical_hi);
  mix(fusion.before_third.canonical_lo);
  mix(static_cast<std::uint64_t>(fusion.after.kind));
  mix(fusion.after.immediate);
  mix(fusion.after.source_hi);
  mix(fusion.after.source_lo);
  mix(fusion.after.canonical_hi);
  mix(fusion.after.canonical_lo);
  mix(static_cast<std::uint64_t>(fusion.after_second.kind));
  mix(fusion.after_second.immediate);
  mix(fusion.after_second.source_hi);
  mix(fusion.after_second.source_lo);
  mix(fusion.after_second.canonical_hi);
  mix(fusion.after_second.canonical_lo);
  mix(static_cast<std::uint64_t>(fusion.after_third.kind));
  mix(fusion.after_third.immediate);
  mix(fusion.after_third.source_hi);
  mix(fusion.after_third.source_lo);
  mix(fusion.after_third.canonical_hi);
  mix(fusion.after_third.canonical_lo);
  mix(fusion.stage_count);
  mix(ring.gpu_owned ? 1u : 0u);
  mix(ring.slot_count);
  mix(ring.frame_elements);
  mix(ring.payload_elements);
  mix(ring.halo_elements);
  mix(ring.schedule_checksum);
  mix(ring.config_stride);
  mix(ring.frame_bytes);
  mix(ring.state_bytes);
  mix(ring.scratch_bytes);
  mix(ring.config_bytes);
  return rund::kernel::ArtifactKey{
      .api = api,
      .scalar = scalar,
      .domain = domain,
      .variant = rund::kernel::LoweringArtifactVariant::DeviceVsm,
      .op_hash_hi = hi.Finish(),
      .op_hash_lo = lo.Finish(),
      .canonical_ir_hash_hi = physical.hi,
      .canonical_ir_hash_lo = physical.lo,
  };
}

std::string entry_name(const rund::kernel::ArtifactKey &key) noexcept {
  constexpr char Hex[] = "0123456789abcdef";
  try {
    std::string result{"rund_compute_map_"};
    const auto append = [&](const std::uint64_t value) {
      for (int shift = 60; shift >= 0; shift -= 4) {
        result.push_back(Hex[(value >> static_cast<unsigned>(shift)) & 0x0fu]);
      }
    };
    append(key.op_hash_hi);
    result.push_back('_');
    append(key.op_hash_lo);
    result += "_device_vsm";
    return result;
  } catch (...) {
    return {};
  }
}

} // namespace rund::node::accel::detail::device_vsm_window_source
