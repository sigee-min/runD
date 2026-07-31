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

struct MetalMapEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::ComputePlan plan{};
  rund::kernel::BindingSet bindings{};
  std::vector<InputWindowPlan> input_plans{};
  std::vector<MetalMapCheck> checks{};
  std::vector<rund::kernel::ComputeDispatchWindow> windows{};
  MetalResidentBindings resident{};
  MetalRuntimeBuffer param{};
  std::shared_ptr<void> pipeline{};
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
    return control.has_count() || control.has_predicate() || !checks.empty();
  }
};
#endif

} // namespace rund::node::accel::detail
