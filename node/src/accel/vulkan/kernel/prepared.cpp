#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../kernel/reset/projection.hpp"
#include "../../kernel/reset/proof.hpp"
#include "../../kernel/reset/stats.hpp"
#include "../buffer/create/telemetry.hpp"
#include "../buffer/resident/find.hpp"
#include "../collective/pipeline.hpp"
#include "../descriptor.hpp"
#include "../resident/access.hpp"
#include "../runtime/timestamp.hpp"
#include "../scratch.hpp"
#include "lease.hpp"
#include "local.hpp"
#include "reset_source.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/model.hpp>

#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

[[nodiscard]] bool ResetBinding(const VulkanAdapter &adapter,
                                const VulkanReset &clear,
                                VulkanStorageBinding &binding,
                                VkDeviceSize &origin) noexcept {
  if (clear.resident.device_buffer == nullptr || adapter.storage_align == 0u ||
      clear.range.count() == 0u ||
      clear.range.stride() < clear.range.element()) {
    return false;
  }
  constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
  if (clear.range.count() - 1u >
      (max - clear.range.element()) / clear.range.stride()) {
    return false;
  }
  const std::uint64_t span =
      (clear.range.count() - 1u) * clear.range.stride() + clear.range.element();
  if (clear.range.offset() > max - span) {
    return false;
  }
  const std::uint64_t base =
      clear.range.offset() - clear.range.offset() % adapter.storage_align;
  const std::uint64_t bytes = clear.range.offset() + span - base;
  if (bytes == 0u || bytes > adapter.storage_limit ||
      base > std::numeric_limits<VkDeviceSize>::max() ||
      bytes > std::numeric_limits<VkDeviceSize>::max()) {
    return false;
  }
  binding = VulkanStorageBinding{
      .buffer = clear.resident.device_buffer,
      .offset = static_cast<VkDeviceSize>(base),
      .range = static_cast<VkDeviceSize>(bytes),
  };
  origin = static_cast<VkDeviceSize>(base);
  return true;
}

[[nodiscard]] rund::AccelCheck
PrepareResetCommands(VulkanAdapter &adapter, VulkanKernelResources &resources) {
  const bool captured = IsPipelinePrivatePreparation(resources.mode);
  bool needs_pipeline = captured && !resources.resets.empty();
  std::uint64_t descriptor_set_count = 0u;
  for (const VulkanReset &clear : resources.resets) {
    needs_pipeline = needs_pipeline || !clear.range.dense();
    if ((captured || !clear.range.dense()) &&
        !rund::kernel::checked::add(descriptor_set_count, 1u,
                                    descriptor_set_count)) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (!needs_pipeline) {
    return rund::AccelCheck{true, "ok"};
  }
  if (captured) {
    resources.reset_pipeline = resources.program == nullptr
                                   ? nullptr
                                   : resources.program->reset_pipeline;
  } else {
    const rund::kernel::LoweringArtifact artifact = VulkanResetArtifact();
    if (!artifact.ok) {
      return rund::AccelCheck{false, artifact.reason};
    }
    resources.reset_pipeline = AcquireVulkanCollectivePipeline(
        adapter, 1u, sizeof(reset::Params), VulkanResetPlan(), artifact);
  }
  if (resources.reset_pipeline == nullptr) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  if (!captured &&
      !ReserveVulkanCollectiveDescriptorDemand(
          adapter, *resources.reset_pipeline, 1u, descriptor_set_count)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  for (VulkanReset &clear : resources.resets) {
    if (clear.range.dense() && !captured) {
      continue;
    }
    if (!AcquireVulkanCollectiveDescriptorSet(
            adapter, *resources.reset_pipeline, 1u, clear.descriptor)) {
      return rund::AccelCheck{false, VulkanLastError(&adapter)};
    }
    std::array<VulkanStorageBinding, 1u> bindings{};
    if (!ResetBinding(adapter, clear, bindings[0], clear.binding_offset)) {
      return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
    }
    if (!WriteVulkanStorageDescriptorSet(adapter, clear.descriptor, bindings)) {
      return rund::AccelCheck{false, VulkanLastError(&adapter)};
    }
  }
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck PrepareVulkanDescriptorLeaseStorage(
    const BoundStep *const steps, const std::size_t step_count,
    std::uint32_t *const failed_node, VulkanKernelResources &resources) {
  if (steps == nullptr || step_count == 0u || resources.size() != step_count ||
      !resources.descriptor_leases.empty()) {
    return rund::AccelCheck{false, "accel_kernel_run_invalid"};
  }
  std::uint64_t lease_count = 0u;
  const bool captured = IsPipelinePrivatePreparation(resources.mode);
  for (const VulkanReset &clear : resources.resets) {
    if ((captured || !clear.range.dense()) &&
        !rund::kernel::checked::add(lease_count, 1u, lease_count)) {
      RecordNode(failed_node, steps[0]);
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  for (std::size_t index = 0u; index < resources.size(); ++index) {
    const VulkanKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr ||
        !rund::kernel::checked::add(
            lease_count, VulkanViewDispatchCount(entry->view), lease_count)) {
      RecordNode(failed_node, steps[index]);
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (captured && (resources.program == nullptr ||
                   resources.program->steps.size() != step_count)) {
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "accel_kernel_template_invalid"};
  }
  for (std::size_t index = 0u; index < step_count; ++index) {
    const VulkanKernelEntry *const entry = resources.entry(index);
    if (entry == nullptr || steps[index].step == nullptr) {
      RecordNode(failed_node, steps[index]);
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const BoundStep &prepared_step =
        entry->view == nullptr ? steps[index] : entry->view->step;
    if (prepared_step.step == nullptr || prepared_step.planned == nullptr) {
      RecordNode(failed_node, steps[index]);
      return rund::AccelCheck{false, "accel_kernel_run_invalid"};
    }
    const PreparedBackendManifest manifest =
        captured ? resources.program->steps[index].manifest
                 : BuildVulkanBackendManifest(
                       *prepared_step.step, prepared_step.planned->plan,
                       &prepared_step,
                       resources.adapter == nullptr
                           ? 0u
                           : resources.adapter->max_dispatch_groups);
    if (!manifest.ok) {
      RecordNode(failed_node, prepared_step);
      return rund::AccelCheck{false, manifest.reason};
    }
    if (!rund::kernel::checked::add(
            lease_count, manifest.descriptor_lease_count, lease_count)) {
      RecordNode(failed_node, prepared_step);
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
  }
  if (lease_count > std::numeric_limits<std::size_t>::max()) {
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  resources.descriptor_leases.reserve(static_cast<std::size_t>(lease_count));
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck
BeginVulkanSteps(VulkanAdapter &adapter, VulkanKernelResources &resources) {
  if (!EnsureVulkanCommandResources(adapter) || !BeginVulkanCommand(adapter)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  BeginVulkanTimestampSpan(adapter, adapter.command_buffer);
  const rund::AccelCheck encoded = ExecuteVulkanKernel(adapter, resources);
  if (!encoded.ok) {
    CancelVulkanCommand(adapter);
  } else {
    EndVulkanTimestampSpan(adapter, adapter.command_buffer);
  }
  return encoded;
}

[[nodiscard]] rund::AccelCheck
RunVulkanSteps(VulkanAdapter &adapter, VulkanKernelResources &resources) {
  const rund::AccelCheck encoded = BeginVulkanSteps(adapter, resources);
  if (!encoded.ok) {
    return encoded;
  }
  if (!SubmitVulkanCommand(adapter)) {
    return rund::AccelCheck{false, VulkanLastError(&adapter)};
  }
  return rund::AccelCheck{true, "ok"};
}

[[nodiscard]] rund::AccelCheck
SubmitVulkanSteps(VulkanAdapter &adapter, VulkanKernelResources &resources) {
  if (resources.size() == 0u) {
    return rund::AccelCheck{false, "accel_vulkan_command_unavailable"};
  }
  const rund::AccelCheck encoded = BeginVulkanSteps(adapter, resources);
  if (!encoded.ok) {
    return encoded;
  }
  return rund::AccelCheck{true, "ok"};
}

void CompleteVulkanPrepared(void *const raw, KernelResult submitted) noexcept {
  auto *const state =
      static_cast<submission::State<VulkanKernelResources> *>(raw);
  if (state == nullptr) {
    return;
  }
  const submission::Claim<VulkanKernelResources> claim =
      submission::Take(*state);
  if (!claim) {
    return;
  }
  VulkanKernelResources &resources = *claim.owner;
  if (submitted.check.ok && resources.adapter != nullptr) {
    std::lock_guard lock{resources.adapter->mutex};
    submitted.check =
        FinishVulkanSteps(*resources.adapter, resources, &submitted.stats);
  }
  submitted.stats.dispatch_count =
      submitted.check.ok ? resources.dispatch_count : 0u;
  SetResetStats(submitted.stats, submitted.check.ok, resources.reset_count,
                resources.reset_bytes);
  claim.completion(claim.user, submitted);
}

void DestroyPreparedVulkanKernelResources(void *const raw) {
  auto *const resources = static_cast<VulkanKernelResources *>(raw);
  if (resources == nullptr) {
    return;
  }
  VulkanAdapter *const adapter = resources->adapter;
  if (adapter != nullptr) {
    std::lock_guard<std::mutex> lock{adapter->mutex};
    DestroyVulkanKernelCommand(*adapter, *resources);
    resources->release();
    for (const VulkanCollectiveDescriptorLease &lease :
         resources->descriptor_leases) {
      if (lease.pipeline != nullptr &&
          lease.slot < lease.pipeline->descriptor_leased.size()) {
        lease.pipeline->descriptor_leased[lease.slot] = false;
      }
    }
  }
  delete resources;
}

} // namespace
#endif

rund::AccelCheck PrepareVulkanResources(
    const rund::AccelDevice &pick, const BoundStep *const steps,
    const std::size_t step_count, const std::uint64_t dispatch_count,
    const KernelPreparationMode mode, const BoundResets *const resets,
    const KernelViewLayout *const views, const RunBinds *const view_binds,
    const KernelScratchLayout *const scratch,
    const BackendRun *const template_probe,
    PreparedKernelTemplateRegistry *const templates,
    std::uint32_t *const failed_node, std::shared_ptr<void> &prepared,
    PreparedMemory &memory) {
  memory = {};
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  prepared.reset();
  if (steps == nullptr || step_count == 0u) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  try {
    VulkanKernelContext context{};
    const rund::AccelCheck valid = ValidateVulkanKernelContext(pick, context);
    if (!valid.ok) {
      return valid;
    }
    VulkanAdapter *const adapter = context.adapter;
    std::shared_ptr<VulkanKernelResources> resources{
        new VulkanKernelResources{}, DestroyPreparedVulkanKernelResources};
    resources->adapter = adapter;
    resources->dispatch_count = dispatch_count;
    resources->mode = mode;
    std::optional<VulkanScratch> scratch_arena;
    if (scratch != nullptr) {
      if (view_binds == nullptr) {
        return rund::AccelCheck{false, "accel_kernel_scratch_invalid"};
      }
      scratch_arena.emplace(pick, *scratch, *view_binds);
      if (!scratch_arena->valid()) {
        return rund::AccelCheck{false, "accel_kernel_scratch_invalid"};
      }
    }
    const rund::AccelCheck view_ready =
        PrepareVulkanStepViews(pick, steps, step_count, mode, views, view_binds,
                               failed_node, *resources);
    if (!view_ready.ok) {
      return view_ready;
    }
    for (std::size_t index = 0u; index < resources->size(); ++index) {
      const VulkanKernelEntry *const entry = resources->entry(index);
      const std::uint64_t auxiliary =
          entry == nullptr ? 0u : VulkanViewDispatchCount(entry->view);
      if (auxiliary > std::numeric_limits<std::uint64_t>::max() -
                          resources->dispatch_count) {
        return rund::AccelCheck{false, "accel_kernel_run_invalid"};
      }
      resources->dispatch_count += auxiliary;
    }
    std::unique_lock<std::mutex> lock{adapter->mutex};
    if (!EnsureVulkanCommandResources(*adapter)) {
      return rund::AccelCheck{false, VulkanLastError(adapter)};
    }
    const VulkanMemoryStats before = adapter->staging_memory;
    if (resets != nullptr) {
      if (resets->size() > std::numeric_limits<std::size_t>::max()) {
        RecordNode(failed_node, steps[0]);
        return rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
      resources->resets.reserve(static_cast<std::size_t>(resets->size()));
      VulkanResidentState &resident = VulkanResidents(*adapter);
      for (std::uint64_t index = 0u; index < resets->size(); ++index) {
        const BoundReset &route = (*resets)[static_cast<std::size_t>(index)];
        const rund::kernel::ResidentBufferRef &ref = route.ref;
        const VulkanViewTransfer *replacement = nullptr;
        if (route.external &&
            !reset::Find(*resources, route.binding, replacement)) {
          return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
        }
        VulkanResidentBufferResult resolved =
            replacement == nullptr
                ? ResolveVulkanResidentBuffer(resident, ref, route.handle,
                                              "compute_resident_id_invalid")
                : replacement->dense;
        if (!resolved.check.ok || resolved.device_buffer == nullptr) {
          return resolved.check.ok
                     ? rund::AccelCheck{false, "accel_kernel_reset_invalid"}
                     : resolved.check;
        }
        const reset::Replacement dense{
            .count = replacement == nullptr ? 0u : replacement->count,
            .element = replacement == nullptr ? 0u : replacement->element_bytes,
        };
        const reset::Spec projected =
            reset::Project(ref, replacement == nullptr ? nullptr : &dense);
        if (!projected.dense() && adapter->max_dispatch_groups == 0u) {
          return rund::AccelCheck{false, "accel_kernel_reset_invalid"};
        }
        const reset::Result proved = reset::Prove(
            projected, resolved.device_buffer->bytes,
            projected.dense() ? std::numeric_limits<std::uint64_t>::max()
                              : std::numeric_limits<std::uint32_t>::max());
        if (!proved.check.ok) {
          return proved.check;
        }
        resources->resets.push_back(VulkanReset{
            .resident = std::move(resolved),
            .range = proved.range,
        });
        const std::uint64_t reset_commands =
            proved.range.dense()
                ? 1u
                : reset::Commands(
                      proved.range.count(),
                      static_cast<std::uint64_t>(adapter->max_dispatch_groups) *
                          256u);
        resources->reset_count = ::rund::detail::counter::SaturatingAdd(
            resources->reset_count, reset_commands);
        resources->reset_bytes = ::rund::detail::counter::SaturatingAdd(
            resources->reset_bytes, reset::Payload(proved.range));
      }
    }
    const rund::AccelCheck template_ready = PrepareVulkanKernelProgramTemplate(
        pick, steps, step_count, mode, template_probe, templates, failed_node,
        *resources);
    if (!template_ready.ok) {
      return template_ready;
    }
    const rund::AccelCheck lease_storage = PrepareVulkanDescriptorLeaseStorage(
        steps, step_count, failed_node, *resources);
    if (!lease_storage.ok) {
      return lease_storage;
    }
    BeginVulkanCollectiveDescriptorEpoch(*adapter);
    rund::AccelCheck ready{};
    {
      VulkanLeaseScope lease_scope{*adapter, resources->descriptor_leases};
      try {
        VulkanScratchScope scratch_scope{
            scratch_arena.has_value() ? &scratch_arena.value() : nullptr};
        const rund::AccelCheck resets_ready =
            PrepareResetCommands(*adapter, *resources);
        ready = resets_ready.ok
                    ? PrepareVulkanSteps(pick, steps, step_count, mode,
                                         failed_node, *resources)
                    : resets_ready;
        for (const VulkanReset &clear : resources->resets) {
          if (ready.ok &&
              (!clear.range.dense() || IsPipelinePrivatePreparation(mode))) {
            const std::uint64_t window =
                static_cast<std::uint64_t>(adapter->max_dispatch_groups) * 256u;
            const std::uint64_t commands =
                reset::Commands(clear.range.count(), window);
            if (commands > std::numeric_limits<std::uint64_t>::max() -
                               resources->dispatch_count) {
              ready = rund::AccelCheck{false, "accel_kernel_run_invalid"};
              break;
            }
            resources->dispatch_count += commands;
          }
        }
        if (ready.ok && !IsPipelinePrivatePreparation(mode)) {
          ready = RecordVulkanKernel(*adapter, *resources);
        }
      } catch (const std::bad_alloc &) {
        RecordNode(failed_node, steps[0]);
        ready = rund::AccelCheck{false, "compute_pipeline_capacity"};
      } catch (const std::length_error &) {
        RecordNode(failed_node, steps[0]);
        ready = rund::AccelCheck{false, "compute_pipeline_capacity"};
      }
    }
    resources->shared_scratch =
        scratch_arena.has_value() && scratch_arena->used();
    if (ready.ok) {
      memory = VulkanPreparedMemory(before, adapter->staging_memory,
                                    adapter->caps.staging_bytes);
      for (std::size_t index = 0u; index < resources->size(); ++index) {
        const VulkanKernelEntry *const entry = resources->entry(index);
        if (entry != nullptr) {
          std::uint64_t traffic = 0u;
          accumulate_memory(
              memory, VulkanViewMemory(entry->view, adapter->caps.staging_bytes,
                                       traffic));
          resources->traffic = ::rund::detail::counter::SaturatingAdd(
              resources->traffic, traffic);
        }
      }
    }
    lock.unlock();
    if (!ready.ok) {
      return ready;
    }
    prepared = std::move(resources);
    return rund::AccelCheck{true, "ok"};
  } catch (const std::bad_alloc &) {
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  } catch (const std::length_error &) {
    RecordNode(failed_node, steps[0]);
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
#else
  (void)pick;
  (void)steps;
  (void)step_count;
  (void)dispatch_count;
  (void)mode;
  (void)resets;
  (void)views;
  (void)view_binds;
  (void)scratch;
  (void)template_probe;
  (void)templates;
  (void)failed_node;
  (void)prepared;
  (void)memory;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

std::uint64_t
VulkanKernelTraffic(const std::shared_ptr<void> &prepared) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const auto *const resources =
      static_cast<const VulkanKernelResources *>(prepared.get());
  return resources == nullptr ? 0u : resources->traffic;
#else
  (void)prepared;
  return 0u;
#endif
}

rund::AccelCheck RunVulkanResources(const rund::AccelDevice &pick,
                                    const std::shared_ptr<void> &prepared) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const resources = static_cast<VulkanKernelResources *>(prepared.get());
  if (resources == nullptr || resources->size() == 0u) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  VulkanKernelContext context{};
  const rund::AccelCheck valid = ValidateVulkanKernelContext(pick, context);
  if (!valid.ok) {
    return valid;
  }
  VulkanAdapter *const adapter = context.adapter;
  std::lock_guard<std::mutex> lock{adapter->mutex};
  const rund::AccelCheck executed = RunVulkanSteps(*adapter, *resources);
  if (!executed.ok) {
    return executed;
  }
  return FinishVulkanSteps(*adapter, *resources);
#else
  (void)pick;
  (void)prepared;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

rund::AccelCheck SubmitVulkanResources(const rund::AccelDevice &pick,
                                       const std::shared_ptr<void> &prepared,
                                       const KernelCompletion completion,
                                       void *const user) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  auto *const resources = static_cast<VulkanKernelResources *>(prepared.get());
  if (resources == nullptr || resources->size() == 0u ||
      completion == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  VulkanKernelContext context{};
  const rund::AccelCheck valid = ValidateVulkanKernelContext(pick, context);
  if (!valid.ok) {
    return valid;
  }
  submission::State<VulkanKernelResources> &state = resources->submission;
  if (!submission::Begin(state, *resources, completion, user)) {
    return rund::AccelCheck{false, "compute_job_busy"};
  }
  VulkanAdapter *const adapter = context.adapter;
  rund::AccelCheck submitted{};
  {
    std::lock_guard lock{adapter->mutex};
    submitted = SubmitVulkanSteps(*adapter, *resources);
    if (submitted.ok &&
        !SubmitVulkanCommand(*adapter, CompleteVulkanPrepared, &state)) {
      submitted = rund::AccelCheck{false, VulkanLastError(adapter)};
    }
  }
  if (!submitted.ok) {
    submission::Cancel(state);
    return submitted;
  }
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)prepared;
  (void)completion;
  (void)user;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace rund::node::accel::detail
