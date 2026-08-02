#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;
struct BoundControl;
struct MetalMapTemplateResources;

[[nodiscard]] inline std::uint64_t MetalMapUniqueCheckCount(
    const rund::kernel::LoweringArtifact &artifact) noexcept {
  std::uint64_t count = 0u;
  for (std::size_t index = 0u; index < artifact.metadata.read_routes.size();
       ++index) {
    bool first = true;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (artifact.metadata.read_routes[prior].index ==
          artifact.metadata.read_routes[index].index) {
        first = false;
        break;
      }
    }
    count += first ? 1u : 0u;
  }
  return count;
}

// Exact immutable-template identity used by proved Pipeline recurrence. The
// common compiler owns source semantics; Metal only compares the already
// frozen native specialization dimensions.
[[nodiscard]] bool MetalMapTemplateMatches(
    const MetalMapTemplateResources &prepared, const MetalAdapter &adapter,
    const rund::kernel::ComputePlan &plan,
    const rund::kernel::BindingSet &bindings) noexcept;

[[nodiscard]] rund::AccelCheck PrepareMetalMapTemplate(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const BoundControl &control,
    std::shared_ptr<const MetalMapTemplateResources> &prepared);

// Consumes a one-shot recurrence artifact whose source string was initially
// reserved to the final backend upper. Stride and Pipeline-private guard edits
// stay inside that allocation, which is moved directly into the adapter cache.
[[nodiscard]] rund::AccelCheck PrepareMetalMapOwnedTemplate(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    rund::kernel::LoweringArtifact &&artifact,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const BoundControl &control,
    std::shared_ptr<const MetalMapTemplateResources> &prepared);

[[nodiscard]] rund::AccelCheck PrepareMetalMapRoute(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const BoundControl &control,
    std::shared_ptr<const MetalMapTemplateResources> prepared,
    std::shared_ptr<void> &resources, std::uint32_t iterations = 1u);

// Route-only materialization for a recurrence already proved by the common
// compiler. No LoweringArtifact is accepted here: the source was admitted
// while building `prepared`, and retaining or reconstructing it for another
// route would recreate the removed intermediate source layer.
[[nodiscard]] rund::AccelCheck PrepareMetalMapProvedRoute(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const BoundControl &control,
    std::shared_ptr<const MetalMapTemplateResources> prepared,
    std::shared_ptr<void> &resources, std::uint32_t iterations = 1u);

[[nodiscard]] rund::AccelCheck PrepareMetalMap(
    const rund::AccelDevice &pick, const rund::kernel::ComputePlan &plan,
    const rund::kernel::LoweringArtifact &artifact,
    const rund::kernel::ComputeDispatchWindow *windows,
    rund::kernel::u64 window_count, const rund::kernel::BindingSet &bindings,
    const BoundControl &control,
    std::shared_ptr<void> &resources, std::uint32_t iterations = 1u);
[[nodiscard]] rund::AccelCheck
EncodeMetalMap(MetalAdapter &adapter, const std::shared_ptr<void> &resources,
               void *command_encoder);
[[nodiscard]] rund::AccelCheck
FinishMetalMap(MetalAdapter &adapter, const std::shared_ptr<void> &resources);

} // namespace rund::node::accel::detail
