#pragma once

#include "../model/plan.hpp"

#include <cassert>
#include <cstdint>

namespace rund::node::accel::detail {

// Scratch binding values are immutable projections of the selected plan's
// temporary roles. Resource owners resolve the slots to native buffers.
class RangeTempSlot final {
public:
  RangeTempSlot() = delete;

  [[nodiscard]] static constexpr RangeTempSlot
  of(const RangeTempRole role, const std::uint8_t ordinal) noexcept {
    return RangeTempSlot{role, ordinal};
  }

  [[nodiscard]] constexpr RangeTempRole role() const noexcept { return role_; }

  [[nodiscard]] constexpr std::uint8_t ordinal() const noexcept {
    return ordinal_;
  }

  [[nodiscard]] friend constexpr bool
  operator==(const RangeTempSlot &, const RangeTempSlot &) = default;

private:
  constexpr RangeTempSlot(const RangeTempRole role,
                          const std::uint8_t ordinal) noexcept
      : role_(role), ordinal_(ordinal) {}

  RangeTempRole role_;
  std::uint8_t ordinal_;
};

enum class RangeScratchKind : std::uint8_t {
  None,
  Pair,
};

class RangeScratch final {
public:
  RangeScratch() = delete;

  [[nodiscard]] static constexpr RangeScratch none() noexcept {
    return RangeScratch{RangeScratchKind::None,
                        RangeTempSlot::of(RangeTempRole::PrefixValues, 0u),
                        RangeTempSlot::of(RangeTempRole::PrefixValues, 0u)};
  }

  [[nodiscard]] static constexpr RangeScratch
  pair(const RangeTempSlot first, const RangeTempSlot second) noexcept {
    return RangeScratch{RangeScratchKind::Pair, first, second};
  }

  [[nodiscard]] constexpr bool uses_global_scratch() const noexcept {
    return disposition_ == RangeScratchKind::Pair;
  }

  [[nodiscard]] constexpr RangeTempSlot first() const noexcept {
    assert(uses_global_scratch());
    return first_;
  }

  [[nodiscard]] constexpr RangeTempSlot second() const noexcept {
    assert(uses_global_scratch());
    return second_;
  }

private:
  constexpr RangeScratch(const RangeScratchKind disposition,
                         const RangeTempSlot first,
                         const RangeTempSlot second) noexcept
      : disposition_(disposition), first_(first), second_(second) {}

  RangeScratchKind disposition_;
  RangeTempSlot first_;
  RangeTempSlot second_;
};

} // namespace rund::node::accel::detail
