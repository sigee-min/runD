#include "../../../device/state.hpp"
#include "../model.hpp"

#include "../../../cpu/view.hpp"
#include "../../local.hpp"
#include "local.hpp"

#include <rund/counter.hpp>

#include <cstddef>
#include <cstring>
#include <memory>
#include <optional>

namespace rund::compute::detail {
namespace {

Status copy_cpu_view(
    const CpuViewTransfer &transfer,
    const std::span<const std::shared_ptr<BufferState>> staging_owners,
    const bool publish, std::size_t &bytes) noexcept {
  bytes = 0u;
  if (transfer.binding >= staging_owners.size()) {
    return Status::fail(Reason::CpuBufferInvalid);
  }
  const std::optional<CpuView> external =
      cpu_view(transfer.external.get(), transfer.view);
  const std::optional<CpuView> staging =
      cpu_view(staging_owners[transfer.binding].get(), 0u, transfer.view.count,
               1u, transfer.view.element_bytes);
  if (!external || !staging ||
      external->footprint.bytes != staging->footprint.bytes) {
    return Status::fail(Reason::CpuBufferInvalid);
  }
  bytes = external->footprint.bytes;
  if (bytes == 0u) {
    return Status::success();
  }
  if (external->data == nullptr || staging->data == nullptr) {
    return Status::fail(Reason::CpuBufferInvalid);
  }
  if (external->footprint.dense()) {
    if (publish) {
      std::memcpy(external->data, staging->data, bytes);
    } else {
      std::memcpy(staging->data, external->data, bytes);
    }
    return Status::success();
  }

  std::byte *external_data = external->data;
  std::byte *staging_data = staging->data;
  for (std::size_t remaining = external->footprint.count; remaining > 1u;
       --remaining) {
    if (publish) {
      std::memcpy(external_data, staging_data, external->footprint.width);
    } else {
      std::memcpy(staging_data, external_data, external->footprint.width);
    }
    external_data += external->footprint.stride;
    staging_data += staging->footprint.stride;
  }
  if (publish) {
    std::memcpy(external_data, staging_data, external->footprint.width);
  } else {
    std::memcpy(staging_data, external_data, external->footprint.width);
  }
  return Status::success();
}

} // namespace

Status
gather_cpu_pipeline_views(const std::shared_ptr<JobState> &state) noexcept {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr ||
      state->program->device->backend != Backend::Cpu) {
    return Status::fail(Reason::PipelineInvalid);
  }
  state->cpu_view_gather_bytes = 0u;
  for (const CpuViewTransfer &transfer : state->cpu_view_inputs) {
    std::size_t bytes = 0u;
    const Status copied = copy_cpu_view(transfer, state->inputs, false, bytes);
    if (!copied) {
      return copied;
    }
    state->cpu_view_gather_bytes = ::rund::detail::counter::SaturatingAdd(
        state->cpu_view_gather_bytes, bytes);
  }
  return Status::success();
}

Status
publish_cpu_pipeline_views(const std::shared_ptr<JobState> &state) noexcept {
  if (state == nullptr || state->program == nullptr ||
      state->program->device == nullptr ||
      state->program->device->backend != Backend::Cpu) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (state->cpu == nullptr) {
    return state->cpu_view_inputs.empty() && state->cpu_view_outputs.empty() &&
                   state->cpu_view_gather_bytes == 0u
               ? Status::success()
               : Status::fail(Reason::CpuRunInvalid);
  }
  std::uint64_t bytes = state->cpu_view_gather_bytes;
  state->cpu_view_gather_bytes = 0u;
  for (const CpuViewTransfer &transfer : state->cpu_view_outputs) {
    std::size_t copied_bytes = 0u;
    const Status copied =
        copy_cpu_view(transfer, state->outputs, true, copied_bytes);
    if (!copied) {
      return copied;
    }
    bytes = ::rund::detail::counter::SaturatingAdd(bytes, copied_bytes);
  }
  ::rund::detail::counter::Accumulate(
      state->cpu->stats.internal_roundtrip_bytes, bytes);
  return Status::success();
}

} // namespace rund::compute::detail
