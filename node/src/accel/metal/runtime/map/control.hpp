#pragma once

#include "resources.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/backend.hpp>
#include <kernel/program/compute/binding/model.hpp>

#include <accel/device.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

struct BoundControl;

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

// These rows are the stable host/device ABI consumed by the Map control
// kernels. Runtime implementation and source materialization live in the
// corresponding control/*.mm owners.
struct MetalMapControlWindow final {
  std::uint64_t begin{};
  std::uint64_t count{};
};

struct MetalMapControlConfig final {
  std::uint32_t has_count{};
  std::uint32_t count_u64{};
  std::uint32_t has_predicate{};
  std::uint32_t predicate_u64{};
  std::uint32_t dispatch_width{};
  std::uint32_t checked{};
  std::uint64_t capacity{};
  std::uint64_t predicate_expected{};
};

static_assert(sizeof(MetalMapControlWindow) == 16u);
static_assert(sizeof(MetalMapControlConfig) == 40u);

[[nodiscard]] rund::kernel::LoweringArtifact
MetalControlledMapArtifact(rund::kernel::LoweringArtifact artifact,
                           const rund::kernel::ComputePlan &plan);

[[nodiscard]] bool
MetalMapControlFinalSourceUpperBytes(std::uint64_t &upper) noexcept;

[[nodiscard]] std::string MetalMapControlSource();

[[nodiscard]] std::pair<std::uint64_t, std::uint64_t>
MetalMapCheckHash(const MetalMapTemplateResources &prepared,
                  const rund::kernel::BindingSet &bindings) noexcept;

[[nodiscard]] rund::kernel::LoweringArtifact
MetalMapCheckArtifact(const MetalMapTemplateResources &prepared,
                      const rund::kernel::BindingSet &bindings);

[[nodiscard]] std::shared_ptr<void>
MetalMapControlPipeline(MetalAdapter &adapter);

[[nodiscard]] bool PrepareMetalMapControl(
    const rund::AccelDevice &pick, const BoundControl &bound,
    const std::vector<rund::kernel::ComputeDispatchWindow> &windows,
    MetalMapEncodeResources &resources);

#endif

} // namespace rund::node::accel::detail
