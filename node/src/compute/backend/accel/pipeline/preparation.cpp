#include "../../../../accel/kernel/backend/run.hpp"
#include "../../../../accel/kernel/prepared/interface/api.hpp"
#include "../pipeline.hpp"

#include "../../../../accel/context/internal/support.hpp"
#include "../../../../accel/context/local.hpp"
#include "../../../job/state.hpp"
#include "../../../pipeline/state.hpp"
#include "../../../status.hpp"

#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <memory>
#include <utility>

namespace rund::compute::detail::accel_backend {

node::accel::detail::PreparedKernelPipelineReservation
plan_pipeline_preparation(
    const DeviceState &device,
    const std::span<const node::accel::detail::PreparedKernelProgramRoute>
        routes,
    const node::accel::detail::PreparedKernelPipelineShape shape,
    node::accel::detail::PreparedKernelTemplateRegistry &templates) noexcept {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return {};
  }
  return node::accel::detail::PlanPreparedKernelPipelineLimit(
      accel->context, routes, shape, templates);
}

node::accel::detail::PreparedKernelPipeline prepare_pipeline(
    const DeviceState &device,
    const std::span<const node::accel::detail::PreparedKernelRun *const>
        prepared,
    const std::span<const std::uint8_t> barriers,
    const std::span<const std::uint32_t> declared_steps,
    const std::span<const node::accel::detail::BackendRecurrence> recurrences,
    const std::span<const node::accel::detail::BackendPublish> publications,
    const std::uint32_t declared_step_count,
    const std::uint32_t generation_stride, const bool profile_steps,
    node::accel::detail::PreparedKernelTemplateRegistry *const templates) {
  const AccelDeviceState *const accel = accel_device(device);
  if (accel == nullptr) {
    return {};
  }
  return node::accel::detail::PrepareKernelPipeline(
      accel->context, prepared, barriers, declared_steps, recurrences,
      publications, declared_step_count, generation_stride, profile_steps,
      templates);
}

rund::AccelCheck seed_pipeline_generation(
    const node::accel::detail::PreparedKernelPipeline &pipeline,
    const std::uint32_t generation) noexcept {
  return node::accel::detail::SeedPreparedKernelPipelineGeneration(pipeline,
                                                                   generation);
}

} // namespace rund::compute::detail::accel_backend
