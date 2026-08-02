#pragma once

#include "../../../sequence/input/window.hpp"
#include "../../resident.hpp"
#include "../resident/bindings.hpp"
#include <kernel/program/compute/graph/schema.hpp>

#include "../../state.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalMapCheck final {
  std::uint32_t binding{};
  std::uint64_t limit{};
};

// Program/layout-owned state. This is built once and shared by transactional
// primary/alternate routes; it contains no route buffer identity or mutable
// execution state.
struct MetalMapTemplateResources final {
  MetalAdapter *adapter = nullptr;
  rund::kernel::ComputePlan plan{};
  std::vector<InputWindowPlan> input_plans{};
  std::vector<std::uint64_t> input_strides{};
  std::vector<std::uint64_t> output_strides{};
  std::vector<MetalMapCheck> checks{};
  std::shared_ptr<void> pipeline{};
  std::shared_ptr<void> control_pipeline{};
  std::shared_ptr<void> check_pipeline{};
};

struct MetalMapEncodeResources {
  MetalAdapter *adapter = nullptr;
  std::shared_ptr<const MetalMapTemplateResources> prepared{};
  rund::kernel::BindingSet bindings{};
  // BindingSet is a non-owning projection. Recurrence history may project
  // proof-owned refs, so retain that owner through the last encode/finish.
  std::shared_ptr<const void> binding_owner{};
  std::vector<rund::kernel::ComputeDispatchWindow> windows{};
  MetalResidentBindings resident{};
  MetalRuntimeBuffer param{};
  rund::kernel::GraphControl control{};
  MetalResidentBufferResult control_count{};
  MetalResidentBufferResult control_predicate{};
  MetalRuntimeBuffer control_args{};
  MetalRuntimeBuffer control_params{};
  MetalRuntimeBuffer control_status{};
  std::shared_ptr<void> control_pipeline{};
  std::shared_ptr<void> check_pipeline{};
  std::uint64_t control_config_offset{};
  std::uint32_t iterations{1u};

  [[nodiscard]] bool controlled() const noexcept {
    return control.has_count() || control.has_predicate() ||
           (prepared != nullptr && !prepared->checks.empty());
  }
};

#endif

} // namespace rund::node::accel::detail
