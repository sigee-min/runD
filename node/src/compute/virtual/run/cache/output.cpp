#include "../../../backend.hpp"
#include "internal.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace rund::compute::detail::virtual_cache_detail {

bool output_bank_frame(const PipelineState &pipeline,
                       const VirtualRunProjection &run,
                       const std::uint32_t frame, std::size_t &local) noexcept {
  if (pipeline.residency_bank >= residency::Pool::BankCount) {
    return false;
  }
  const residency::FrameRegion region =
      pipeline.residency_stage == PipelineResidencyStage::Graph
          ? run.output_regions[pipeline.residency_bank]
          : residency::FrameRegion{
                .tier = pipeline.device->backend == Backend::Cpu
                            ? residency::FrameTier::Host
                            : residency::FrameTier::Device,
                .role = residency::FrameRole::Output,
                .first = pipeline.residency_pool->first_output_frame +
                         pipeline.residency_bank *
                             static_cast<std::uint32_t>(run.frame_capacity),
                .count = static_cast<std::uint32_t>(run.frame_capacity),
            };
  if (region.count != run.frame_capacity || frame < region.first ||
      frame - region.first >= region.count) {
    return false;
  }
  local = static_cast<std::size_t>(frame - region.first);
  return true;
}

} // namespace rund::compute::detail::virtual_cache_detail

namespace rund::compute::detail {

BufferReadView residency_output_view(PipelineState &pipeline) noexcept {
  std::lock_guard pipeline_lock{pipeline.gate};
  if (pipeline.publication == nullptr) {
    return {};
  }
  std::lock_guard publication_lock{pipeline.publication->gate};
  if (!has_private_residency_authority(pipeline) ||
      pipeline.device == nullptr || pipeline.device->ops == nullptr ||
      pipeline.device->ops->host_read == nullptr ||
      pipeline.residency_output >= pipeline.resources.size()) {
    return {};
  }
  const PipelineResource &resource =
      pipeline.resources[pipeline.residency_output];
  return resource.buffer == nullptr ? BufferReadView{}
                                    : pipeline.device->ops->host_read(
                                          *pipeline.device, *resource.buffer);
}

Status reserve_residency_output(PipelineState &pipeline,
                                const VirtualEpochProjection &epoch,
                                const VirtualRunProjection &run,
                                VirtualOutputReservation &reservation,
                                bool &poison) noexcept {
  reservation = {};
  if (pipeline.device == nullptr || pipeline.residency_pool == nullptr ||
      epoch.page_count == 0u || epoch.page_count > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (pipeline.device->backend == Backend::Cpu) {
    return Status::success();
  }
  residency::Pool &pool = *pipeline.residency_pool;
  if (pipeline.residency_bank >= residency::Pool::BankCount ||
      pool.host_output_frame_count !=
          run.host_output_frame_capacity * residency::Pool::BankCount ||
      run.host_output_frame_capacity < run.frame_capacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::array<residency::CacheUse, PipelineLeafCapacity> uses{};
  for (std::size_t index = 0u;
       index < static_cast<std::size_t>(epoch.page_count); ++index) {
    const std::uint64_t page = epoch.failed_page + index;
    residency::DirtyExtent dirty{};
    if (!project_virtual_output_dirty(run, page, dirty)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    uses[index] = residency::CacheUse{
        .key = virtual_output_cache_key(run, run.output_backing,
                                        run.output_version, page),
        .access = residency::Access::Write,
        .next_use = residency::NeverUse,
        .dirty = dirty,
    };
  }
  const std::uint32_t first =
      pool.first_host_output_frame +
      pipeline.residency_bank * run.host_output_frame_capacity;
  const residency::AuthorityResult acquired = pool.authority().begin(
      std::span<const residency::CacheUse>{
          uses.data(), static_cast<std::size_t>(epoch.page_count)},
      residency::FrameTier::Host, residency::FrameRole::Output, first,
      run.host_output_frame_capacity);
  if (!acquired ||
      acquired.lease.bindings.size() > reservation.bindings.size() ||
      acquired.lease.transitions.size() > reservation.transitions.size()) {
    bool clean = true;
    if (acquired) {
      clean = pool.authority().complete(acquired.lease.token, false);
    }
    poison = !clean || poison;
    return Status::fail(
        acquired.failure == residency::AuthorityFailure::Busy
            ? Reason::PipelineBusy
            : (clean ? Reason::PipelineMemoryBudget : Reason::PipelineInvalid));
  }
  const bool dirty_conflict = std::any_of(
      acquired.lease.bindings.begin(), acquired.lease.bindings.end(),
      [](const residency::CacheBinding binding) {
        return !binding.prior_dirty.empty();
      });
  const bool invalid_transition = std::any_of(
      acquired.lease.transitions.begin(), acquired.lease.transitions.end(),
      [](const residency::CacheTransition transition) {
        return transition.kind == residency::TransitionKind::Fetch ||
               transition.kind == residency::TransitionKind::Writeback;
      });
  if (dirty_conflict || invalid_transition) {
    poison = !pool.authority().complete(acquired.lease.token, false) || poison;
    return Status::fail(dirty_conflict ? Reason::PipelineBusy
                                       : Reason::PipelineInvalid);
  }
  reservation.binding_count = acquired.lease.bindings.size();
  reservation.transition_count = acquired.lease.transitions.size();
  reservation.token = acquired.lease.token;
  std::copy(acquired.lease.bindings.begin(), acquired.lease.bindings.end(),
            reservation.bindings.begin());
  std::copy(acquired.lease.transitions.begin(),
            acquired.lease.transitions.end(), reservation.transitions.begin());
  for (std::size_t index = 0u; index < reservation.binding_count; ++index) {
    if (virtual_host_output_frame(run, reservation.bindings[index].frame) ==
        nullptr) {
      poison = !pool.authority().complete(reservation.token, false) || poison;
      reservation.token = 0u;
      return Status::fail(Reason::PipelineInvalid);
    }
  }
  if (!pool.authority().activate(reservation.token)) {
    poison = !pool.authority().complete(reservation.token, false) || poison;
    reservation.token = 0u;
    return Status::fail(Reason::PipelineInvalid);
  }
  return Status::success();
}

residency::EpochLease
residency_output_lease(const PipelineState &pipeline,
                       const residency::EpochLease execution,
                       VirtualOutputReservation &reservation) noexcept {
  if (pipeline.device != nullptr && pipeline.device->backend == Backend::Cpu) {
    return execution;
  }
  return residency::EpochLease{
      .bindings =
          std::span<const residency::CacheBinding>{reservation.bindings.data(),
                                                   reservation.binding_count},
      .transitions =
          std::span<const residency::CacheTransition>{
              reservation.transitions.data(), reservation.transition_count},
      .token = reservation.token,
  };
}

bool cancel_residency_output(PipelineState &pipeline,
                             VirtualOutputReservation &reservation) noexcept {
  if (reservation.token == 0u) {
    return true;
  }
  if (pipeline.residency_pool == nullptr) {
    return false;
  }
  const bool completed = pipeline.residency_pool->authority().complete(
      reservation.token, false, true);
  reservation.token = 0u;
  return completed;
}

} // namespace rund::compute::detail
