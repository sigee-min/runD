#include "../../adapter/error.hpp"
#include "../../adapter/access.hpp"

#include "../admission.hpp"
#include "../api.hpp"
#include "../control.hpp"
#include "../local.hpp"
#include "../source/control.hpp"
#include "../source/upper.hpp"
#include "../resources/admission.hpp"

#include "../../collective/pipeline.hpp"
#include "../../../kernel/backend/template/identity.hpp"
#include "../../../kernel/step/map/stride.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace {

[[nodiscard]] bool
SameVulkanMapTemplateBindings(const VulkanMapTemplateResources &prepared,
                              const rund::kernel::BindingSet &bindings,
                              const std::uint64_t alignment) noexcept {
  const auto same =
      [alignment](const std::vector<VulkanMapBindingLayout> &layouts,
                  const rund::kernel::ResidentBindingRange &refs) {
        if (alignment == 0u || layouts.size() != refs.count) {
          return false;
        }
        for (std::size_t index = 0u; index < layouts.size(); ++index) {
          const auto *const ref = refs.ref(index);
          if (ref == nullptr || ref->stride_bytes != layouts[index].stride ||
              ref->offset_bytes % alignment != layouts[index].base) {
            return false;
          }
        }
        return true;
      };
  return same(prepared.input_layouts, bindings.resident_inputs) &&
         same(prepared.output_layouts, bindings.resident_outputs);
}

} // namespace

#endif

bool VulkanMapTemplateMatches(
    const VulkanMapTemplateResources &prepared, const VulkanAdapter &adapter,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::BindingSet &bindings) noexcept {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  return prepared.adapter == &adapter &&
         backend_template_plan::same_plan(prepared.plan, plan) &&
         SameVulkanMapTemplateBindings(prepared, bindings,
                                       adapter.storage_align);
#else
  (void)prepared;
  (void)adapter;
  (void)plan;
  (void)bindings;
  return false;
#endif
}

namespace {

[[nodiscard]] static rund::AccelCheck PrepareVulkanMapTemplateImpl(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    rund::kernel::LoweringArtifact *const owned_artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const VulkanMapTemplateResources> &prepared) {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  prepared.reset();
  auto *const adapter = CheckedVulkanAdapter(pick);
  if (adapter == nullptr) {
    return rund::AccelCheck{false, "accel_vulkan_unavailable"};
  }
  SetVulkanLastError(*adapter, "ok");
  const rund::AccelCheck valid = ValidateVulkanMapPrepare(
      *adapter, plan, artifact, windows, window_count, bindings);
  if (!valid.ok) {
    return valid;
  }

  auto owned = std::make_shared<VulkanMapTemplateResources>();
  VulkanMapTemplateResources *const raw = owned.get();
  raw->adapter = adapter;
  raw->plan = plan;
  raw->input_plans.resize(static_cast<std::size_t>(plan.input_buffer_count));
  if (!FreezeInputWindowPlans(artifact.metadata, plan.tile_count,
                              raw->input_plans)) {
    return rund::AccelCheck{false, "compute_binding_mismatch"};
  }
  const auto freeze_layout = [&](const rund::kernel::ResidentBindingRange &refs,
                                 std::vector<VulkanMapBindingLayout> &layouts) {
    layouts.reserve(static_cast<std::size_t>(refs.count));
    for (std::uint64_t index = 0u; index < refs.count; ++index) {
      const auto *const ref = refs.ref(index);
      if (ref == nullptr || adapter->storage_align == 0u) {
        return false;
      }
      layouts.push_back(VulkanMapBindingLayout{
          .stride = ref->stride_bytes,
          .base = ref->offset_bytes % adapter->storage_align,
      });
    }
    return true;
  };
  if (!freeze_layout(bindings.resident_inputs, raw->input_layouts) ||
      !freeze_layout(bindings.resident_outputs, raw->output_layouts)) {
    return rund::AccelCheck{false, "compute_binding_mismatch"};
  }
  for (const rund::kernel::ReadRoute route : artifact.metadata.read_routes) {
    const auto *const ref = bindings.resident_inputs.ref(route.index);
    if (ref == nullptr || adapter->storage_align == 0u) {
      return rund::AccelCheck{false, "compute_binding_mismatch"};
    }
    const auto found = std::find_if(raw->checks.begin(), raw->checks.end(),
                                    [&](const VulkanMapCheck check) {
                                      return check.binding == route.index;
                                    });
    if (found == raw->checks.end()) {
      raw->checks.push_back(VulkanMapCheck{
          .binding = route.index,
          .limit = route.count,
          .offset = ref->offset_bytes % adapter->storage_align,
          .stride = ref->stride_bytes,
      });
    } else {
      found->limit =
          std::min(found->limit, static_cast<std::uint64_t>(route.count));
    }
  }
  const bool controlled_source = control.active() || !raw->checks.empty();
  std::uint64_t specialized_upper = 0u;
  std::uint64_t final_upper = 0u;
  if (!MapSpecializedSourceUpperBytes(artifact, plan, specialized_upper) ||
      (controlled_source && !VulkanControlledMapSourceUpperBytes(
                                plan, specialized_upper, final_upper))) {
    return rund::AccelCheck{false, "compute_pipeline_capacity"};
  }
  if (!controlled_source) {
    final_upper = specialized_upper;
  }
  rund::kernel::LoweringArtifact strided_artifact =
      owned_artifact == nullptr
          ? SpecializeMap(artifact, plan, bindings, adapter->storage_align,
                          final_upper)
          : SpecializeMapInPlace(std::move(*owned_artifact), plan, bindings,
                                 adapter->storage_align, final_upper);
  rund::kernel::LoweringArtifact controlled_artifact =
      controlled_source
          ? VulkanControlledMapArtifact(std::move(strided_artifact), plan)
          : std::move(strided_artifact);
  if (!controlled_artifact.ok) {
    return rund::AccelCheck{false, controlled_artifact.reason};
  }
  rund::kernel::ComputePlan pipeline_plan = plan;
  if (control.active() || !raw->checks.empty()) {
    ++pipeline_plan.input_buffer_count;
  }
  raw->pipeline = AcquireVulkanCachedPipeline(*adapter, pipeline_plan,
                                              std::move(controlled_artifact));
  if (raw->pipeline == nullptr) {
    return rund::AccelCheck{false, VulkanLastError(adapter)};
  }
  if (controlled_source) {
    const rund::kernel::LoweringArtifact control_artifact =
        VulkanMapControlArtifact(plan);
    raw->control_pipeline = AcquireVulkanCollectivePipeline(
        *adapter, 4u, sizeof(VulkanMapControlPush), plan, control_artifact);
    if (raw->control_pipeline == nullptr) {
      return rund::AccelCheck{false, VulkanLastError(adapter)};
    }
  }
  if (!raw->checks.empty()) {
    const std::uint64_t descriptor_count = raw->checks.size() + 3u;
    if (descriptor_count > std::numeric_limits<std::uint32_t>::max()) {
      return rund::AccelCheck{false, "compute_pipeline_capacity"};
    }
    const rund::kernel::LoweringArtifact check_artifact =
        VulkanMapCheckArtifact(*raw);
    raw->check_pipeline = AcquireVulkanCollectivePipeline(
        *adapter, static_cast<std::uint32_t>(descriptor_count),
        sizeof(VulkanMapControlPush), plan, check_artifact);
    if (raw->check_pipeline == nullptr) {
      return rund::AccelCheck{false, VulkanLastError(adapter)};
    }
  }
  prepared = std::move(owned);
  return rund::AccelCheck{true, "ok"};
#else
  (void)pick;
  (void)plan;
  (void)artifact;
  (void)owned_artifact;
  (void)windows;
  (void)window_count;
  (void)bindings;
  (void)control;
  (void)prepared;
  return rund::AccelCheck{false, "accel_vulkan_loader_unavailable"};
#endif
}

} // namespace

rund::AccelCheck PrepareVulkanMapTemplate(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const VulkanMapTemplateResources> &prepared) {
  return PrepareVulkanMapTemplateImpl(pick, plan, artifact, nullptr, windows,
                                      window_count, bindings, control,
                                      prepared);
}

rund::AccelCheck PrepareVulkanMapOwnedTemplate(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    rund::kernel::LoweringArtifact &&artifact,
    const rund::kernel::ComputeDispatchWindow *const windows,
    const rund::kernel::u64 window_count,
    const rund::kernel::BindingSet &bindings, const BoundControl &control,
    std::shared_ptr<const VulkanMapTemplateResources> &prepared) {
  return PrepareVulkanMapTemplateImpl(pick, plan, artifact, &artifact, windows,
                                      window_count, bindings, control,
                                      prepared);
}

} // namespace rund::node::accel::detail
