#include "internal.hpp"

#include "../source.hpp"
#include "../source/graph_map_reduce.hpp"
#include "../source/graph_map_scan.hpp"
#include "../source/graph_pointwise.hpp"
#include "../source/graph_resident.hpp"
#include "../source/reduce.hpp"
#include "../source/scan.hpp"
#include "../validation.hpp"

#include "../source/graph_resident/model.hpp"

#include <kernel/core/checked.hpp>

#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

namespace rund::node::accel::detail::device_vsm_projection {
namespace {

struct Owner final {
  std::array<std::shared_ptr<void>, DeviceVsmGraphStageCapacity> pipelines{};
  std::size_t pipeline_count{};
  rund::kernel::LoweringArtifact artifact{};
  std::vector<rund::kernel::ComputeDispatchWindow> windows{};
  std::vector<std::byte> parameters{};
};

} // namespace

DeviceVsmProofProjection MaterializeProjection(
    const DeviceVsmProofRequest &request,
    const Candidate &candidate) noexcept {
  try {
    auto owner = std::make_shared<Owner>();
    if (candidate.graph_resident_ready) {
      owner->pipeline_count = request.graph_resident_pipeline_count;
      for (std::size_t stage = 0u; stage < owner->pipeline_count; ++stage) {
        owner->pipelines[stage] =
            candidate.graph_resident_pipelines[stage].owner;
      }
    } else if (candidate.graph_pointwise) {
      owner->pipeline_count = request.graph_pointwise_pipeline_count;
      for (std::size_t stage = 0u; stage < owner->pipeline_count; ++stage) {
        owner->pipelines[stage] = request.graph_pointwise_pipelines[stage].owner;
      }
    } else {
      owner->pipeline_count = 2u;
      owner->pipelines[0u] = request.pipeline.owner;
      owner->pipelines[1u] = candidate.graph_collective_reduce
                                 ? request.graph_collective_pipeline.owner
                                 : request.peer_pipeline.owner;
    }
    DeviceVsmWindowProof window_proof{};
    DeviceVsmGraphMapReduceProof graph_proof{};
    DeviceVsmGraphPointwiseProof graph_pointwise_proof{};
    DeviceVsmScanProof scan_proof{};
    DeviceVsmReduceProof reduce_proof{};
    rund::kernel::ComputePlan plan{};
    if (candidate.graph_resident_ready) {
      std::array<DeviceVsmGraphResidentStageSource,
                 DeviceVsmGraphResidentStageCapacity>
          stages{};
      for (std::size_t stage = 0u;
           stage < request.graph_resident_pipeline_count; ++stage) {
        stages[stage] = DeviceVsmGraphResidentStageSource{
            .artifact =
                &candidate.graph_resident_steps[stage]->step->artifact,
            .input = &candidate.graph_resident_steps[stage]->step->cpu_input,
            .semantic =
                &candidate.graph_resident_steps[stage]->step->map_semantic,
        };
      }
      const DeviceVsmGraphResidentArtifact built =
          BuildDeviceVsmGraphResidentArtifact(
              {stages.data(), request.graph_resident_pipeline_count},
              candidate.graph_resident_proof, request.graph_wavefront,
              request.geometry, request.residents);
      if (!built) {
        return {.check = {false, built.reason}};
      }
      owner->artifact = built.artifact;
      owner->windows.assign(
          1u, rund::kernel::ComputeDispatchWindow{.begin_sequence = 0u,
                                                   .tile_count = 1u});
      plan = built.plan;
    } else if (candidate.pointwise) {
      owner->artifact = candidate.first->step->artifact;
      if (!TransformDeviceVsmSource(
              owner->artifact,
              candidate.first->planned->plan.input_buffer_count,
              candidate.first->planned->plan.output_buffer_count)) {
        return {.check = {false, "device_vsm_source_invalid"}};
      }
      owner->windows.assign(candidate.first->map_windows.data(),
                            candidate.first->map_windows.data() +
                                candidate.first->map_windows.size());
      if (candidate.bindings.param_bytes != 0u) {
        const auto *const begin =
            static_cast<const std::byte *>(candidate.bindings.param_data);
        owner->parameters.assign(
            begin, begin + static_cast<std::size_t>(candidate.bindings.param_bytes));
      }
      plan = candidate.first->planned->plan;
    } else if (candidate.graph_pointwise) {
      std::array<DeviceVsmGraphPointwiseStage, DeviceVsmGraphStageCapacity>
          stages{};
      for (std::size_t stage = 0u;
           stage < request.graph_pointwise_pipeline_count; ++stage) {
        stages[stage] = DeviceVsmGraphPointwiseStage{
            .artifact =
                &candidate.graph_pointwise_steps[stage]->step->artifact,
            .input = &candidate.graph_pointwise_steps[stage]->step->cpu_input,
            .semantic =
                &candidate.graph_pointwise_steps[stage]->step->map_semantic,
        };
      }
      const DeviceVsmGraphPointwiseArtifact built =
          BuildDeviceVsmGraphPointwiseArtifact(
              {stages.data(), request.graph_pointwise_pipeline_count},
              request.graph_pointwise_topology, request.geometry,
              request.graph_pointwise_page_map, request.graph_wavefront);
      if (!built) {
        return {.check = {false, built.reason}};
      }
      owner->artifact = built.artifact;
      owner->windows.assign(
          1u, rund::kernel::ComputeDispatchWindow{.begin_sequence = 0u,
                                                   .tile_count = 1u});
      graph_pointwise_proof = built.proof;
      plan = built.plan;
    } else if (candidate.graph_collective_reduce) {
      const DeviceVsmGraphMapReduceArtifact built =
          BuildDeviceVsmGraphMapReduceArtifact(
              candidate.graph_map->step->artifact,
              candidate.graph_map->step->cpu_input,
              candidate.graph_map->step->map_semantic,
              candidate.graph_reduce.semantic, request.geometry,
              request.graph_wavefront);
      if (!built) {
        return {.check = {false, built.reason}};
      }
      owner->artifact = built.artifact;
      owner->windows.assign(
          1u, rund::kernel::ComputeDispatchWindow{.begin_sequence = 0u,
                                                   .tile_count = 1u});
      graph_proof = built.proof;
      plan = built.plan;
      if (candidate.graph_bindings.param_bytes != 0u) {
        const auto *const begin = static_cast<const std::byte *>(
            candidate.graph_bindings.param_data);
        owner->parameters.assign(
            begin,
            begin + static_cast<std::size_t>(candidate.graph_bindings.param_bytes));
      }
    } else if (candidate.scanned) {
      if (candidate.scan.map.kind == DeviceVsmScanMapKind::CanonicalTotalU64) {
        if (candidate.scan.map_step == nullptr ||
            candidate.scan.map_step->step == nullptr) {
          return {.check = {false, "device_vsm_graph_scan_map_invalid"}};
        }
        const DeviceVsmGraphMapScanArtifact built =
            BuildDeviceVsmGraphMapScanArtifact(
                candidate.scan.map_step->step->artifact,
                candidate.scan.map_step->step->cpu_input,
                candidate.scan.map_step->step->map_semantic,
                candidate.scan.semantic, request.geometry);
        if (!built) {
          return {.check = {false, built.reason}};
        }
        owner->artifact = built.artifact;
        scan_proof = built.proof;
        plan = built.plan;
        if (candidate.scan.map_bindings.param_bytes != 0u) {
          const auto *const begin = static_cast<const std::byte *>(
              candidate.scan.map_bindings.param_data);
          owner->parameters.assign(
              begin,
              begin + static_cast<std::size_t>(
                          candidate.scan.map_bindings.param_bytes));
        }
      } else {
        const DeviceVsmScanArtifact built = BuildDeviceVsmScanArtifact(
            candidate.scan.semantic, candidate.scan.api, request.geometry,
            candidate.scan.map);
        if (!built) {
          return {.check = {false, built.reason}};
        }
        owner->artifact = built.artifact;
        scan_proof = built.proof;
        plan = built.plan;
      }
      owner->windows.assign(
          1u, rund::kernel::ComputeDispatchWindow{.begin_sequence = 0u,
                                                   .tile_count = 1u});
    } else if (candidate.reduced) {
      const DeviceVsmReduceArtifact built = BuildDeviceVsmReduceArtifact(
          candidate.reduce.semantic, candidate.reduce.api, request.geometry);
      if (!built) {
        return {.check = {false, built.reason}};
      }
      owner->artifact = built.artifact;
      owner->windows.assign(
          1u, rund::kernel::ComputeDispatchWindow{.begin_sequence = 0u,
                                                   .tile_count = 1u});
      reduce_proof = built.proof;
      plan = built.plan;
    } else {
      const RangePlan *const range = RangePlanFor(
          candidate.window.step->step->operation);
      const DeviceVsmWindowArtifact built = BuildDeviceVsmWindowArtifact(
          *range, candidate.window.active->plan, request.geometry,
          candidate.window.fusion,
          request.kind == DeviceVsmProjectionKind::WindowRing
              ? candidate.ring_plan
              : DeviceVsmWindowRingPlan{},
          device_vsm_window_projection::map_sources(candidate.window));
      if (!built) {
        return {.check = {false, built.reason}};
      }
      owner->artifact = built.artifact;
      owner->windows.assign(1u, built.window);
      owner->parameters.assign(built.parameters.begin(),
                               built.parameters.end());
      window_proof = built.proof;
      plan = built.plan;
    }
    auto proof = std::make_shared<DeviceVsmProof>();
    proof->identity = request.identity;
    proof->topology = candidate.pointwise              ? DeviceVsmTopology::Pointwise
                      : candidate.graph_resident_ready
                          ? DeviceVsmTopology::GraphResident
                      : candidate.graph_pointwise
                          ? DeviceVsmTopology::GraphPointwise
                      : candidate.graph_collective_reduce
                          ? DeviceVsmTopology::GraphMapReduce
                      : candidate.scanned ? DeviceVsmTopology::Scan
                      : candidate.reduced ? DeviceVsmTopology::Reduce
                                          : DeviceVsmTopology::Window;
    proof->window = window_proof;
    proof->graph_map_reduce = graph_proof;
    proof->graph_pointwise = graph_pointwise_proof;
    proof->graph_resident = candidate.graph_resident_proof;
    proof->graph_wavefront = request.graph_wavefront;
    proof->scan = scan_proof;
    proof->reduce = reduce_proof;
    proof->semantic_owner = owner;
    proof->artifact = &owner->artifact;
    proof->plan = plan;
    proof->residents = request.residents;
    proof->windows = owner->windows.data();
    proof->parameters =
        owner->parameters.empty() ? nullptr : owner->parameters.data();
    proof->geometry = request.geometry;
    proof->output_bytes = request.output_bytes == 0u
                              ? request.geometry.logical_bytes
                              : request.output_bytes;
    proof->window_count = owner->windows.size();
    proof->parameter_bytes = owner->parameters.size();
    proof->width = request.width;
    proof->fixed_common_storage = true;
    if (!device_vsm_proof_valid(*proof)) {
      return {.check = {false, "device_vsm_proof_invalid"}};
    }
    return {.check = {true, "ok"}, .proof = std::move(proof)};
  } catch (const std::bad_alloc &) {
    return {.check = {false, "compute_pipeline_capacity"}};
  } catch (const std::length_error &) {
    return {.check = {false, "compute_pipeline_capacity"}};
  }
}

} // namespace rund::node::accel::detail::device_vsm_projection
