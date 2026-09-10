#include "../../state/assembly.hpp"
#include "local.hpp"

#include "../arena.hpp"
#include "../compare.hpp"
#include "../prepare.hpp"
#include "../resource.hpp"

#include "../../../../accel/kernel/recurrence.hpp"
#include "../../../backend.hpp"
#include "../../../buffer/local.hpp"
#include "../../../cpu/run/state.hpp"
#include "../../../job/local.hpp"
#include "../../../memory/arena.hpp"
#include "../../../status.hpp"
#include "../../../type.hpp"

#include <kernel/core/checked.hpp>
#include <rund/compute/abi/observe.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rund::compute::detail {

namespace {
[[nodiscard]] Status
append_buffer_commitment(const DeviceState &device, const std::uint64_t logical,
                         PipelineBufferCommitment &total) noexcept {
  const auto committed = planned_buffer_storage_bytes(device, logical);
  return committed &&
                 kernel::checked::add(total.logical, logical, total.logical) &&
                 kernel::checked::add(total.committed, *committed,
                                      total.committed)
             ? Status::success()
             : Status::fail(committed ? Reason::PipelineCapacity
                                      : committed.reason());
}

} // namespace

[[nodiscard]] Result<PipelineBufferCommitment>
plan_buffer_commitment(const DeviceState &device,
                       const PipelineMemoryPlan &plan) noexcept {
  PipelineBufferCommitment total{};
  for (const PipelineResolvedResourcePlan &resource : plan.resources) {
    if (std::holds_alternative<PipelineExternalResourcePlan>(
            resource.locator)) {
      continue;
    }
    const auto *internal =
        std::get_if<PipelineInternalResourcePlan>(&resource.locator);
    if (internal != nullptr && internal->owner != nullptr) {
      continue;
    }
    if (resource.physical_bytes < resource.bytes ||
        !kernel::checked::add(total.logical, resource.bytes, total.logical) ||
        !kernel::checked::add(total.committed, resource.physical_bytes,
                              total.committed)) {
      return Result<PipelineBufferCommitment>::fail(Reason::PipelineCapacity);
    }
  }
  for (const std::size_t words : plan.chunks) {
    std::uint64_t logical = 0u;
    if (!kernel::checked::mul(static_cast<std::uint64_t>(words), memory::Word,
                              logical)) {
      return Result<PipelineBufferCommitment>::fail(Reason::PipelineCapacity);
    }
    const Status appended = append_buffer_commitment(device, logical, total);
    if (!appended) {
      return Result<PipelineBufferCommitment>::fail(appended.reason());
    }
  }
  for (const std::size_t words : plan.view_chunks) {
    std::uint64_t logical = 0u;
    if (!kernel::checked::mul(static_cast<std::uint64_t>(words), memory::Word,
                              logical)) {
      return Result<PipelineBufferCommitment>::fail(Reason::PipelineCapacity);
    }
    const Status appended = append_buffer_commitment(device, logical, total);
    if (!appended) {
      return Result<PipelineBufferCommitment>::fail(appended.reason());
    }
  }
  for (const node::accel::detail::KernelScratchPage &scratch : plan.scratch) {
    const Status appended =
        append_buffer_commitment(device, scratch.bytes, total);
    if (!appended) {
      return Result<PipelineBufferCommitment>::fail(appended.reason());
    }
  }
  return Result<PipelineBufferCommitment>::success(total);
}

[[nodiscard]] Status
validate_buffer_commitment(const DeviceState &device,
                           const BufferState &buffer) noexcept {
  const auto committed = planned_buffer_storage_bytes(device, buffer.bytes);
  return committed && buffer.physical_bytes == *committed
             ? Status::success()
             : Status::fail(committed ? Reason::PipelineInvalid
                                      : committed.reason());
}
} // namespace rund::compute::detail
