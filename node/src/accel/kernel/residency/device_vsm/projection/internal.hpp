#pragma once

#include "../projection.hpp"
#include "../../../prepared/model.hpp"
#include "reduce.hpp"
#include "scan.hpp"
#include "window.hpp"

#include <array>

namespace rund::node::accel::detail::device_vsm_projection {

// This is bounded, invocation-local scratch. It carries references and small
// authenticated summaries between the independent selection, validation, and
// materialization owners; it is never retained in DeviceVsmProof or runtime
// state as a second authority.
struct Candidate final {
  rund::AccelCheck check{false, "device_vsm_projection_request_invalid"};
  const prepared::PipelineState *pipeline{};
  const prepared::PipelineState *peer{};
  const BoundStep *first{};
  rund::kernel::BindingSet bindings{};
  const char *primary_reason{};
  bool pointwise{};

  const BoundStep *graph_map{};
  rund::kernel::BindingSet graph_bindings{};
  const char *graph_reason{};
  bool graph_map_reduce{};

  std::array<const prepared::PipelineState *, DeviceVsmGraphStageCapacity>
      graph_pointwise_pipelines{};
  std::array<const BoundStep *, DeviceVsmGraphStageCapacity>
      graph_pointwise_steps{};
  std::array<rund::kernel::BindingSet, DeviceVsmGraphStageCapacity>
      graph_pointwise_bindings{};
  const char *graph_pointwise_reason{
      "device_vsm_graph_pointwise_stage_invalid"};
  bool graph_pointwise{};

  const prepared::PipelineState *graph_collective{};
  device_vsm_reduce_projection::Authority graph_reduce{};
  const char *graph_reduce_reason{};
  bool graph_collective_reduce{};

  device_vsm_scan_projection::Authority scan{};
  const char *scan_reason{};
  bool scanned{};

  device_vsm_reduce_projection::Authority reduce{};
  const char *reduce_reason{};
  bool reduced{};

  device_vsm_window_projection::WindowAuthority window{};
  const char *window_reason{};
  DeviceVsmWindowRingPlan ring_plan{};
  bool windowed{};

  DeviceVsmGraphResidentProof graph_resident_proof{};
  std::array<PreparedKernelPipeline, DeviceVsmGraphResidentStageCapacity>
      graph_resident_pipelines{};
  std::array<const prepared::PipelineState *,
             DeviceVsmGraphResidentStageCapacity>
      graph_resident_states{};
  std::array<const BoundStep *, DeviceVsmGraphResidentStageCapacity>
      graph_resident_steps{};
  std::array<rund::kernel::BindingSet, DeviceVsmGraphResidentStageCapacity>
      graph_resident_bindings{};
  const char *graph_resident_reason{
      "device_vsm_graph_resident_stage_invalid"};
  bool graph_resident_ready{};
};

[[nodiscard]] bool exact_pointwise_pipeline(
    const prepared::PipelineState &, const BoundStep *&,
    rund::kernel::BindingSet &, const char *&) noexcept;

[[nodiscard]] bool exact_graph_map_pipeline(
    const prepared::PipelineState &, const BoundStep *&,
    rund::kernel::BindingSet &, const char *&) noexcept;

[[nodiscard]] bool same_pointwise_pipeline(
    const prepared::PipelineState &, const BoundStep &,
    const rund::kernel::BindingSet &) noexcept;

[[nodiscard]] bool same_graph_map_pipeline(
    const prepared::PipelineState &, const BoundStep &,
    const rund::kernel::BindingSet &) noexcept;

[[nodiscard]] bool SelectProjectionCandidate(
    const DeviceVsmProofRequest &, Candidate &) noexcept;

[[nodiscard]] rund::AccelCheck ValidateProjectionCandidate(
    const DeviceVsmProofRequest &, const Candidate &) noexcept;

[[nodiscard]] DeviceVsmProofProjection MaterializeProjection(
    const DeviceVsmProofRequest &, const Candidate &) noexcept;

} // namespace rund::node::accel::detail::device_vsm_projection
