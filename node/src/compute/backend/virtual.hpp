#pragma once

#include <accel/check.hpp>
#include <rund/compute/status.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute {
class VirtualBacking;
}

namespace rund::compute {
struct Stats;
}

namespace rund::compute::detail {
struct DeviceState;

struct PipelineState;
struct VirtualPipelineState;
struct VirtualRunProjection;
struct VirtualExecutionSlidingPrepared;
class VirtualDeviceVsmRouteProof;
struct VirtualExecutionDeviceVsmPrepared;
struct VirtualExecutionResult;
namespace device_vsm_product_detail {
struct DeviceVsmProductRun;
}

struct VirtualVsmAsyncOps final {
  using Start =
      Status (*)(VirtualPipelineState &, std::span<VirtualBacking *const>,
                 VirtualBacking &, const VirtualRunProjection &,
                 const VirtualExecutionDeviceVsmPrepared &, Stats &,
                 device_vsm_product_detail::DeviceVsmProductRun &) noexcept;
  using Finish = VirtualExecutionResult (*)(
      device_vsm_product_detail::DeviceVsmProductRun &, Status) noexcept;
  using Open =
      bool (*)(const device_vsm_product_detail::DeviceVsmProductRun &) noexcept;
  using Valid =
      bool (*)(const device_vsm_product_detail::DeviceVsmProductRun &) noexcept;
  using Unknown =
      void (*)(device_vsm_product_detail::DeviceVsmProductRun &) noexcept;

  Start start{};
  Finish finish{};
  Open open{};
  Valid valid{};
  Unknown unknown{};

  [[nodiscard]] bool ready() const noexcept {
    return start != nullptr && finish != nullptr && open != nullptr &&
           valid != nullptr && unknown != nullptr;
  }
};

struct VirtualDeviceOps final {
  // Accelerator-only product coordinator.  The common Virtual runner calls
  // these through DeviceOps so CPU closure neither links nor branches through
  // the physical Sliding/Authority implementation.
  Status (*prepare_virtual_sliding_product)(
      VirtualPipelineState &, const VirtualRunProjection &,
      VirtualExecutionSlidingPrepared &) noexcept = nullptr;
  VirtualExecutionResult (*execute_virtual_sliding_product)(
      VirtualPipelineState &, VirtualBacking &, VirtualBacking &,
      const VirtualRunProjection &, const VirtualExecutionSlidingPrepared &,
      Stats &) noexcept = nullptr;
  // Stronger page-coordinate product route. The backend owns the complete
  // recurrence after one submit; no epoch Project/Release/Returned callback
  // is representable through this interface.
  Status (*prepare_virtual_device_vsm_product)(
      VirtualPipelineState &, const VirtualRunProjection &,
      VirtualDeviceVsmRouteProof,
      VirtualExecutionDeviceVsmPrepared &) noexcept = nullptr;
  VirtualExecutionResult (*execute_virtual_device_vsm_product)(
      VirtualPipelineState &, std::span<VirtualBacking *const>,
      VirtualBacking &, const VirtualRunProjection &,
      const VirtualExecutionDeviceVsmPrepared &, Stats &) noexcept = nullptr;
  // Returns BackendUnsupported with `selected=false` when the prepared
  // Pipeline is not a proved service-free recurrence. Once selected, this
  // owns the sole aggregate submit/Final route and may not fall back.
  Status (*run_service_free_direct_product)(
      const std::shared_ptr<PipelineState> &,
      bool &selected) noexcept = nullptr;
  Status (*virtual_pipeline_capability)(const DeviceState &) noexcept = nullptr;
  // Accelerator-owned DeviceVSM seams.  The common runner only probes the
  // pure admission hook and snapshots the cohesive async façade; it never
  // links directly to the accelerator implementation.
  rund::AccelCheck (*admit_virtual_device_vsm)(
      const VirtualPipelineState &, const VirtualRunProjection &,
      const VirtualDeviceVsmRouteProof &) noexcept = nullptr;
  const VirtualVsmAsyncOps *vsm_async = nullptr;
};

} // namespace rund::compute::detail
