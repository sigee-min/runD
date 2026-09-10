#pragma once

#include "../model/capability.hpp"
#include "../model/shape.hpp"
#include "evaluation.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rund::node::accel::detail {
namespace range_plan_detail {

class IdentityBuilder final {
public:
  constexpr IdentityBuilder() noexcept = default;

  template <typename T>
    requires(std::is_integral_v<T> && sizeof(T) <= sizeof(std::uint64_t))
  constexpr void add(const T value) noexcept {
    add64(static_cast<std::uint64_t>(value));
  }

  constexpr void add(const rund::kernel::u128 value) noexcept {
    add64(static_cast<std::uint64_t>(value >> 64u));
    add64(static_cast<std::uint64_t>(value));
  }

  [[nodiscard]] constexpr RangeIdentity finish() const noexcept {
    return RangeIdentity{.hi = hi_, .lo = lo_};
  }

private:
  constexpr void add64(const std::uint64_t value) noexcept {
    hi_ ^= value + 0x9e3779b97f4a7c15ull + (hi_ << 6u) + (hi_ >> 2u);
    hi_ *= 0xbf58476d1ce4e5b9ull;
    lo_ ^= value + 0x94d049bb133111ebull + (lo_ << 7u) + (lo_ >> 3u);
    lo_ *= 0x9e3779b185ebca87ull;
  }
  std::uint64_t hi_{0x6a09e667f3bcc909ull};
  std::uint64_t lo_{0xbb67ae8584caa73bull};
};

[[nodiscard]] constexpr RangeIdentity
SourceIdentity(const RangeShape &shape, const RangeCaps &capabilities,
               const RangeCandidate candidate) noexcept {
  IdentityBuilder identity{};
  identity.add(0x72616e67652d7372ull); // "range-sr"
  identity.add(
      capabilities.source_variant() == RangeSource::Metal &&
              candidate.disposition() == RangePath::BlockPrefixSuffix
          ? 4u
      : capabilities.source_variant() == RangeSource::Metal &&
              (candidate.disposition() == RangePath::PrefixDifference ||
               candidate.disposition() == RangePath::TiledDifference)
          ? 6u
      : candidate.disposition() == RangePath::PrefixDifference &&
              !capabilities.cpu_only()
          ? 4u
          : 3u);
  identity.add(static_cast<std::uint8_t>(capabilities.source_variant()));
  identity.add(static_cast<std::uint8_t>(candidate.disposition()));
  identity.add(candidate.width());
  identity.add(candidate.radius_capacity());
  identity.add(static_cast<std::uint8_t>(shape.traits().operation()));
  identity.add(static_cast<std::uint8_t>(shape.traits().domain()));
  identity.add(static_cast<std::uint8_t>(shape.traits().arithmetic_law()));
  identity.add(static_cast<std::uint8_t>(shape.boundary()));
  identity.add(shape.element_bytes());
  identity.add(static_cast<std::uint8_t>(shape.count()));
  if (capabilities.source_variant() == RangeSource::Metal &&
      candidate.disposition() == RangePath::BlockPrefixSuffix) {
    identity.add(shape.stride() == 1u);
  }
  return identity.finish();
}

[[nodiscard]] constexpr RangeIdentity
ExecutionIdentity(const RangeShape &shape,
                  const CandidateEvaluation &evaluation,
                  const RangeIdentity source_identity) noexcept {
  IdentityBuilder identity{};
  identity.add(0x72616e67652d6578ull); // "range-ex"
  identity.add(source_identity.hi);
  identity.add(source_identity.lo);
  identity.add(shape.input_count());
  identity.add(shape.output_count());
  identity.add(shape.window_size());
  identity.add(shape.stride());
  identity.add(shape.padding());
  identity.add(evaluation.stage_count);
  for (std::size_t index = 0u; index < evaluation.stage_count; ++index) {
    const RangeStagePlan &stage = evaluation.stages[index];
    identity.add(static_cast<std::uint8_t>(stage.disposition));
    identity.add(stage.level);
    identity.add(stage.element_count);
    identity.add(stage.groups);
    identity.add(stage.width);
  }
  identity.add(evaluation.temporary_count);
  for (std::size_t index = 0u; index < evaluation.temporary_count; ++index) {
    const RangeTempReq &temporary = evaluation.temporaries[index];
    identity.add(static_cast<std::uint8_t>(temporary.role));
    identity.add(temporary.ordinal);
    identity.add(temporary.bytes);
    identity.add(temporary.alignment);
    identity.add(temporary.first_stage);
    identity.add(temporary.last_stage);
  }
  identity.add(evaluation.cost.global_read_bytes);
  identity.add(evaluation.cost.global_write_bytes);
  identity.add(evaluation.cost.workgroup_count);
  identity.add(evaluation.cost.combine_ops);
  identity.add(evaluation.cost.inverse_ops);
  identity.add(evaluation.cost.scale_ops);
  identity.add(evaluation.cost.shared_bytes);
  identity.add(evaluation.cost.scratch_bytes);
  identity.add(evaluation.cost.dispatch_count);
  identity.add(evaluation.cost.launched_lanes);
  return identity.finish();
}

} // namespace range_plan_detail
} // namespace rund::node::accel::detail
