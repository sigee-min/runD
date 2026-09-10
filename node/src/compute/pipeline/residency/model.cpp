#include "model.hpp"

#include "../../type.hpp"

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <limits>

namespace rund::compute::detail::residency {
bool StreamPlan::epoch(const std::uint64_t index, PageRun &run) const noexcept {
  std::uint64_t first_page = 0u;
  if (index >= epoch_count() || frame_capacity_ == 0u ||
      !kernel::checked::mul(index, frame_capacity_, first_page) ||
      first_page >= page_count_) {
    return false;
  }
  run = PageRun{
      .first_page = first_page,
      .page_count = std::min(frame_capacity_, page_count_ - first_page),
      .pin = PinInterval{.first_epoch = index, .last_epoch = index},
      .prefetch_epoch =
          index > prefetch_distance_ ? index - prefetch_distance_ : 0u,
      .ready_epoch = index,
  };
  return true;
}

bool StreamPlan::dirty_extent(const std::uint64_t page,
                              DirtyRange &dirty) const noexcept {
  std::uint64_t logical_offset = 0u;
  if (page >= page_count_ ||
      !kernel::checked::mul(page, dirty_.bytes, logical_offset) ||
      logical_offset >= dirty_bytes_) {
    return false;
  }
  dirty = DirtyRange{
      .offset = dirty_.offset,
      .bytes = std::min(dirty_.bytes, dirty_bytes_ - logical_offset),
  };
  return true;
}

bool StreamPlan::next_use(const std::uint64_t page,
                          std::uint64_t &next) const noexcept {
  if (page >= page_count_) {
    return false;
  }
  next = page > NeverUse - page_count_ ? NeverUse : page_count_ + page;
  return true;
}

bool StreamPlan::first_use(const std::uint64_t page,
                           std::uint64_t &use) const noexcept {
  if (page >= page_count_) {
    return false;
  }
  use = page;
  return true;
}

TiledGraphPlan::TiledGraphPlan(
    const std::uint64_t page_count, const std::uint64_t frame_capacity,
    const std::uint64_t prefetch_distance,
    std::vector<TiledGraphResource> resources,
    std::vector<TiledGraphPhysicalClass> physical_classes,
    std::vector<TiledGraphStage> stages,
    const std::uint64_t graph_fingerprint_hi,
    const std::uint64_t graph_fingerprint_lo) noexcept
    : page_count_(page_count), frame_capacity_(frame_capacity),
      prefetch_distance_(prefetch_distance),
      graph_fingerprint_hi_(graph_fingerprint_hi),
      graph_fingerprint_lo_(graph_fingerprint_lo),
      resources_(std::move(resources)),
      physical_classes_(std::move(physical_classes)),
      stages_(std::move(stages)) {}

std::uint64_t TiledGraphPlan::epoch_count() const noexcept {
  std::uint64_t count = 0u;
  return kernel::checked::mul(batch_count(), stages_.size(), count) ? count
                                                                    : 0u;
}

const TiledGraphResource *
TiledGraphPlan::resource(const std::uint32_t resource_id) const noexcept {
  const auto found = std::lower_bound(
      resources_.begin(), resources_.end(), resource_id,
      [](const TiledGraphResource &resource, const std::uint32_t id) {
        return resource.resource < id;
      });
  return found == resources_.end() || found->resource != resource_id ? nullptr
                                                                     : &*found;
}

bool TiledGraphPlan::active(const std::uint64_t page_count,
                            const std::span<const std::uint64_t> logical_bytes,
                            TiledGraphInvocation &projection) const noexcept {
  projection = {};
  if (page_count == 0u || page_count > page_count_ || frame_capacity_ == 0u ||
      resources_.empty() || resources_.size() > TiledGraphResourceCapacity ||
      logical_bytes.size() != resources_.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < resources_.size(); ++index) {
    const TiledGraphResource &resource = resources_[index];
    std::uint64_t capacity = 0u;
    std::uint64_t last_offset = 0u;
    if (logical_bytes[index] == 0u ||
        logical_bytes[index] > resource.logical_bytes ||
        logical_bytes[index] % type_bytes(resource.type) != 0u ||
        !kernel::checked::mul(page_count, resource.page_bytes, capacity) ||
        !kernel::checked::mul(page_count - 1u, resource.page_bytes,
                              last_offset) ||
        logical_bytes[index] <= last_offset ||
        logical_bytes[index] > capacity) {
      return false;
    }
    projection.logical_bytes_[index] = logical_bytes[index];
  }
  projection.plan_ = this;
  projection.page_count_ = page_count;
  projection.resource_count_ = resources_.size();
  return true;
}

std::uint64_t TiledGraphInvocation::page_count() const noexcept {
  return valid() ? page_count_ : 0u;
}

std::uint64_t TiledGraphInvocation::frame_capacity() const noexcept {
  return valid() ? plan_->frame_capacity() : 0u;
}

std::uint64_t TiledGraphInvocation::prefetch_distance() const noexcept {
  return valid() ? plan_->prefetch_distance() : 0u;
}

std::uint64_t TiledGraphInvocation::batch_count() const noexcept {
  return !valid() || frame_capacity() == 0u
             ? 0u
             : page_count_ / frame_capacity() +
                   static_cast<std::uint64_t>(page_count_ % frame_capacity() !=
                                              0u);
}

std::uint64_t TiledGraphInvocation::epoch_count() const noexcept {
  std::uint64_t count = 0u;
  return valid() && kernel::checked::mul(batch_count(), plan_->stage_count(),
                                         count)
             ? count
             : 0u;
}

bool TiledGraphInvocation::batch(const std::uint64_t index,
                                 PageRun &run) const noexcept {
  std::uint64_t first_page = 0u;
  std::uint64_t first_epoch = 0u;
  if (!valid() || index >= batch_count() || frame_capacity() == 0u ||
      !kernel::checked::mul(index, frame_capacity(), first_page) ||
      !kernel::checked::mul(index, plan_->stage_count(), first_epoch) ||
      first_page >= page_count_) {
    return false;
  }
  const std::uint64_t prefetch_batch =
      index > prefetch_distance() ? index - prefetch_distance() : 0u;
  std::uint64_t prefetch_epoch = 0u;
  std::uint64_t last_epoch = first_epoch;
  if (!kernel::checked::mul(prefetch_batch, plan_->stage_count(),
                            prefetch_epoch) ||
      !kernel::checked::add(first_epoch, plan_->stage_count() - 1u,
                            last_epoch)) {
    return false;
  }
  run = PageRun{
      .first_page = first_page,
      .page_count = std::min(frame_capacity(), page_count_ - first_page),
      .pin = PinInterval{.first_epoch = first_epoch, .last_epoch = last_epoch},
      .prefetch_epoch = prefetch_epoch,
      .ready_epoch = first_epoch,
  };
  return true;
}

bool TiledGraphInvocation::supply_stream(
    StreamPlan &projection) const noexcept {
  projection = {};
  if (!valid() || plan_->resources_.empty() || plan_->stages_.empty()) {
    return false;
  }
  const TiledGraphStage &terminal = plan_->stages_.back();
  const auto output_port = std::find_if(
      terminal.ports.begin(), terminal.ports.end(),
      [](const TiledGraphPort &port) { return writes(port.access); });
  if (output_port == terminal.ports.end() ||
      std::find_if(std::next(output_port), terminal.ports.end(),
                   [](const TiledGraphPort &port) {
                     return writes(port.access);
                   }) != terminal.ports.end()) {
    return false;
  }
  const TiledGraphResource *const output =
      plan_->resource(output_port->resource);
  if (output == nullptr) {
    return false;
  }
  const std::size_t output_index =
      static_cast<std::size_t>(output - plan_->resources_.data());
  projection = StreamPlan{page_count_, frame_capacity(),
                          DirtyRange{.bytes = output->page_bytes},
                          logical_bytes_[output_index], prefetch_distance()};
  return true;
}

bool TiledGraphInvocation::first_use(const std::uint32_t resource_id,
                                     const std::uint64_t page,
                                     std::uint64_t &use) const noexcept {
  const TiledGraphResource *const resource =
      valid() ? plan_->resource(resource_id) : nullptr;
  std::uint64_t batch = 0u;
  if (resource == nullptr || page >= page_count_ || frame_capacity() == 0u ||
      resource->kind != GraphResourceKind::ExternalInput ||
      resource->persistence != ResourcePersistence::Backing ||
      resource->first_stage == NoGraphStage ||
      resource->first_stage >= plan_->stage_count() ||
      !kernel::checked::mul(page / frame_capacity(), plan_->stage_count(),
                            batch)) {
    return false;
  }
  return kernel::checked::add(batch, resource->first_stage, use);
}

bool TiledGraphInvocation::page_bytes(const std::uint32_t resource_id,
                                      const std::uint64_t page,
                                      std::uint64_t &bytes) const noexcept {
  bytes = 0u;
  const TiledGraphResource *const resource =
      valid() ? plan_->resource(resource_id) : nullptr;
  if (resource == nullptr || page >= page_count_) {
    return false;
  }
  const std::size_t resource_index =
      static_cast<std::size_t>(resource - plan_->resources_.data());
  std::uint64_t offset = 0u;
  if (!kernel::checked::mul(page, resource->page_bytes, offset) ||
      offset >= logical_bytes_[resource_index]) {
    return false;
  }
  bytes =
      std::min(resource->page_bytes, logical_bytes_[resource_index] - offset);
  return bytes != 0u;
}

bool TiledGraphInvocation::owned_by(const TiledGraphPlan &plan) const noexcept {
  return valid() && plan_ == &plan;
}

ResidencyPlan::ResidencyPlan(const std::uint64_t page_bytes,
                             const StreamPlan stream,
                             const Identity identity) noexcept
    : kind_(Kind::Stream), page_bytes_(page_bytes),
      frame_capacity_(static_cast<std::uint32_t>(stream.frame_capacity())),
      stream_(stream), identity_(identity) {}

ResidencyPlan::ResidencyPlan(const std::uint64_t page_bytes,
                             TiledGraphPlan tiled_graph,
                             const Identity identity) noexcept
    : kind_(Kind::TiledGraph), page_bytes_(page_bytes),
      frame_capacity_(static_cast<std::uint32_t>(tiled_graph.frame_capacity())),
      tiled_graph_(std::move(tiled_graph)), identity_(identity) {}

} // namespace rund::compute::detail::residency
