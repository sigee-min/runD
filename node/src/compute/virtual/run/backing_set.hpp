#pragma once

#include "../state.hpp"

#include <rund/compute/stats.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace rund::compute::detail {

struct VirtualRunProjection;

struct VirtualBackingLockSet final {
  std::array<std::unique_lock<std::mutex>,
             VirtualPipelineState::InputCapacity + 1u>
      locks{};
  std::size_t count{};

  [[nodiscard]] explicit operator bool() const noexcept { return count >= 2u; }
};

// Invocation-owned resources stay alive through the terminal owner. Backing
// gates are acquired before projection/recovery; the pool gate is acquired
// only after those checks and is held until finalization returns.
struct VirtualRunResources final {
  VirtualBackingLockSet backings{};
  std::unique_lock<std::mutex> pool{};
  std::unique_ptr<residency::Authority::ViewCommitReceipt> view_commit{};
  std::uint64_t page_count{};
};

// Locks every public input plus the output in one global gate-address order.
// Program-port order remains in VirtualPipelineState; this type owns only the
// mutation-serialization lifetime for the invocation.
[[nodiscard]] bool lock_backings(VirtualPipelineState &,
                                 VirtualRunResources &) noexcept;

// Acquires the Pool execution gate. Poolless routes never call this owner and
// therefore never dereference a residency pool.
[[nodiscard]] Status lock_pool(VirtualPipelineState &,
                               VirtualRunResources &) noexcept;

// Stages the pooled residency scope under the already-held Pool gate.
[[nodiscard]] Status stage_pool(VirtualPipelineState &,
                                const VirtualRunProjection &, Stats &,
                                VirtualRunResources &, bool &poison) noexcept;

// These helpers assume the Pool execution gate is held by the caller. The
// Authority gate is acquired by the receipt disposition itself; an active
// receipt is never reset without an explicit close, abort, or quarantine.
[[nodiscard]] Status close_pool_stage(VirtualRunResources &) noexcept;
[[nodiscard]] Status quarantine_pool_stage(VirtualRunResources &) noexcept;

} // namespace rund::compute::detail
