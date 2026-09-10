#include "src/accel/kernel/backend/manifest.hpp"
#include "src/accel/kernel/backend/run.hpp"
#include "src/accel/kernel/backend/template/identity.hpp"
#include "src/accel/kernel/backend/template/source.hpp"
#include "src/accel/kernel/prepared/template/registry.hpp"
#include "src/accel/kernel/step/map/stride.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

#include "map.hpp"

namespace node_accel_contract {

[[nodiscard]] bool MetalMapWordClassPartitionsProgramTemplates() {
  using namespace rund::node::accel::detail;
  using backend_template_plan::program_map_specialization_fingerprint;
  using backend_template_plan::runtime_map_specialization_fingerprint;
  using backend_template_plan::same_program_template;
  using backend_template_plan::same_template;

  KernelExecutionStep step{};
  step.artifact.key.api = rund::kernel::ComputeApi::Metal;
  step.artifact.metadata.binding_accesses = {
      rund::kernel::ComputeBindingAccess::Read,
      rund::kernel::ComputeBindingAccess::Write};
  step.artifact.metadata.read_count = 1u;
  step.artifact.metadata.write_count = 1u;
  step.artifact.metadata.input_element_bytes = {4u};
  step.artifact.metadata.output_element_bytes = {4u};
  step.artifact.metadata.ok = true;
  step.artifact.metadata.reason = "ok";
  if (!step.graph_binding_indices.push_back(0u) ||
      !step.graph_binding_indices.push_back(1u)) {
    return false;
  }
  step.graph_binding_indices_ok = true;
  const std::array roles{rund::kernel::BufferRole::Read,
                         rund::kernel::BufferRole::Write};
  const KernelExecution execution{
      .graph_roles = roles,
      .steps = std::span<const KernelExecutionStep>{&step, 1u},
  };

  const std::shared_ptr<void> kernel_owner = std::make_shared<int>(1);
  const rund::AccelKernel kernel{
      .kernel_id = 1u,
      .graph_id_hi = 2u,
      .graph_id_lo = 3u,
      .node_count = 1u,
      .api = rund::AccelApi::Metal,
      .context_id = 4u,
      .owner = kernel_owner,
  };
  const std::array aligned_bindings{
      PreparedKernelProgramBindingIdentity{
          .offset_bytes = 0u,
          .element_bytes = 4u,
          .stride_bytes = 4u,
          .count = 8u,
          .usage = rund::kernel::kResidentUsageRead,
      },
      PreparedKernelProgramBindingIdentity{
          .offset_bytes = 0u,
          .element_bytes = 4u,
          .stride_bytes = 4u,
          .count = 8u,
          .usage = rund::kernel::kResidentUsageWrite,
      },
  };
  auto aligned_peer_bindings = aligned_bindings;
  aligned_peer_bindings[0u].offset_bytes = 4u;
  aligned_peer_bindings[1u].offset_bytes = 8u;
  auto bytewise_bindings = aligned_bindings;
  bytewise_bindings[0u].offset_bytes = 1u;
  const PreparedKernelProgramRoute aligned{
      .kernel = &kernel,
      .tile_count = 8u,
      .program_bindings = aligned_bindings,
  };
  const PreparedKernelProgramRoute aligned_peer{
      .kernel = &kernel,
      .tile_count = 8u,
      .program_bindings = aligned_peer_bindings,
  };
  const PreparedKernelProgramRoute bytewise{
      .kernel = &kernel,
      .tile_count = 8u,
      .program_bindings = bytewise_bindings,
  };
  const std::array public_routes{aligned, aligned_peer, bytewise};
  std::uint64_t public_template_count = 0u;
  for (std::size_t index = 0u; index < public_routes.size(); ++index) {
    bool seen = false;
    for (std::size_t prior = 0u; prior < index; ++prior) {
      if (same_program_template(execution, public_routes[index],
                                public_routes[prior], 1u)) {
        seen = true;
        break;
      }
    }
    public_template_count += seen ? 0u : 1u;
  }
  const auto aligned_fingerprint =
      program_map_specialization_fingerprint(execution, aligned);
  const auto bytewise_fingerprint =
      program_map_specialization_fingerprint(execution, bytewise);
  if (!same_program_template(execution, aligned, aligned_peer, 1u) ||
      same_program_template(execution, aligned, bytewise, 1u) ||
      public_template_count != 2u || !aligned_fingerprint.ok ||
      !bytewise_fingerprint.ok ||
      (aligned_fingerprint.hi == bytewise_fingerprint.hi &&
       aligned_fingerprint.lo == bytewise_fingerprint.lo)) {
    return false;
  }

  PlannedStep planned{};
  planned.plan = rund::kernel::ComputePlan{
      .tile_count = 8u,
      .api = rund::kernel::ComputeApi::Metal,
      .input_buffer_count = 1u,
      .output_buffer_count = 1u,
      .input_bytes_per_tile = 4u,
      .output_bytes_per_tile = 4u,
      .bytes_per_tile = 8u,
      .dispatch_window_tiles = 8u,
      .dispatch_count = 1u,
      .ok = true,
      .reason = "ok",
  };
  planned.artifact = &step.artifact;
  rund::AccelDevice device{};
  struct RuntimeRoute final {
    RunBinds refs{};
    StepBinds bindings{};
    BoundStep bound{};
    BackendRun run{};
  };
  const std::shared_ptr<void> resident = std::make_shared<int>(2);
  const auto build_runtime_route = [&](RuntimeRoute &route,
                                       const std::uint64_t input_offset) {
    const bool input = route.refs.push(
        rund::kernel::ResidentBufferRef{
            .id = 1u,
            .bytes = 64u,
            .offset_bytes = input_offset,
            .element_bytes = 4u,
            .stride_bytes = 4u,
            .count = 8u,
            .usage = rund::kernel::kResidentUsageRead,
        },
        resident);
    const bool output = route.refs.push(
        rund::kernel::ResidentBufferRef{
            .id = 2u,
            .bytes = 64u,
            .element_bytes = 4u,
            .stride_bytes = 4u,
            .count = 8u,
            .usage = rund::kernel::kResidentUsageWrite,
        },
        resident);
    route.bindings.inputs.bind(route.refs, 1u);
    route.bindings.outputs.bind(route.refs, 1u);
    const bool indices =
        route.bindings.inputs.push(0u) && route.bindings.outputs.push(1u);
    route.bound = BoundStep{
        .step = &step,
        .planned = &planned,
        .bindings = route.bindings,
    };
    route.run = BackendRun{
        .pick = &device,
        .execution = &execution,
        .steps = &route.bound,
        .step_count = 1u,
    };
    return input && output && indices;
  };
  RuntimeRoute aligned_runtime{};
  RuntimeRoute bytewise_runtime{};
  if (!build_runtime_route(aligned_runtime, 0u) ||
      !build_runtime_route(bytewise_runtime, 1u)) {
    return false;
  }
  const auto aligned_runtime_fingerprint =
      runtime_map_specialization_fingerprint(aligned_runtime.run);
  const auto bytewise_runtime_fingerprint =
      runtime_map_specialization_fingerprint(bytewise_runtime.run);
  return !same_template(aligned_runtime.run, bytewise_runtime.run, 1u) &&
         aligned_runtime_fingerprint.ok && bytewise_runtime_fingerprint.ok &&
         (aligned_runtime_fingerprint.hi != bytewise_runtime_fingerprint.hi ||
          aligned_runtime_fingerprint.lo != bytewise_runtime_fingerprint.lo);
}

} // namespace node_accel_contract
