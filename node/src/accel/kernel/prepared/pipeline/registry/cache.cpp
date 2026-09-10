#include "../registry.hpp"

#include "internal.hpp"

#include <memory>
#include <new>

namespace rund::node::accel::detail {

std::shared_ptr<void> FindPreparedKernelTemplate(
    const PreparedKernelTemplateRegistry &registry,
    const KernelExecutionStep *const authority, const std::uint64_t variant_hi,
    const std::uint64_t variant_lo, const PreparedKernelTemplateMatch match,
    const void *const probe) noexcept {
  PreparedKernelTemplateRegistryState *const state = registry_state(registry);
  if (state == nullptr || authority == nullptr || match == nullptr ||
      probe == nullptr) {
    return {};
  }
  std::lock_guard lock{state->mutex};
  for (const PreparedKernelTemplateEntry &entry : state->entries) {
    if (entry.authority == authority && entry.variant_hi == variant_hi &&
        entry.variant_lo == variant_lo && entry.prepared != nullptr &&
        match(entry.prepared.get(), probe)) {
      return entry.prepared;
    }
  }
  return {};
}

rund::AccelCheck PublishPreparedKernelTemplate(
    PreparedKernelTemplateRegistry &registry,
    const KernelExecutionStep *const authority, const std::uint64_t variant_hi,
    const std::uint64_t variant_lo, const BackendOps &ops,
    const PreparedKernelTemplateMatch match, const void *const probe,
    std::shared_ptr<void> &prepared) noexcept {
  PreparedKernelTemplateRegistryState *const state = registry_state(registry);
  if (state == nullptr || authority == nullptr || match == nullptr ||
      probe == nullptr || prepared == nullptr || ops.api != state->api ||
      ops.observe_pipeline_template == nullptr) {
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  std::lock_guard lock{state->mutex};
  for (const PreparedKernelTemplateEntry &entry : state->entries) {
    if (entry.authority == authority && entry.variant_hi == variant_hi &&
        entry.variant_lo == variant_lo && entry.ops == &ops &&
        entry.prepared != nullptr && match(entry.prepared.get(), probe)) {
      prepared = entry.prepared;
      return rund::AccelCheck{true, "ok"};
    }
  }
  try {
    state->entries.push_back(PreparedKernelTemplateEntry{
        .authority = authority,
        .variant_hi = variant_hi,
        .variant_lo = variant_lo,
        .ops = &ops,
        .prepared = prepared,
    });
  } catch (const std::bad_alloc &) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  return rund::AccelCheck{true, "ok"};
}

} // namespace rund::node::accel::detail
