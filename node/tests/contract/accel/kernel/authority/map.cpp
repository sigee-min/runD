#include "src/accel/kernel/backend/manifest.hpp"
#include "src/accel/kernel/backend/source_recipe.hpp"
#include "src/accel/kernel/backend/template_plan.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "src/accel/metal/kernel/manifest.hpp"
#include "src/accel/metal/kernel/pipeline/source.hpp"
#include "src/accel/metal/runtime/map/api.hpp"
#include "src/accel/metal/runtime/map/control.hpp"
#include "src/accel/metal/runtime/map/resources.hpp"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

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

[[nodiscard]] bool MapSourceSpecializationIsSingleOwnerAndExact() {
  using namespace rund::node::accel::detail;
  static_assert(MapSpecializationEditCapacity ==
                3u * rund::kernel::kMaxComputeBindingCount);
  static_assert(MetalMapWordBytes == 4u);
  static_assert(MetalMapBindingWordAligned(
      rund::kernel::ResidentBufferRef{.offset_bytes = 4u, .stride_bytes = 8u}));
  static_assert(!MetalMapBindingWordAligned(
      rund::kernel::ResidentBufferRef{.offset_bytes = 1u, .stride_bytes = 8u}));
  static_assert(!MetalMapBindingWordAligned(
      rund::kernel::ResidentBufferRef{.offset_bytes = 4u, .stride_bytes = 6u}));

  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = rund::kernel::ComputeApi::Metal;
  artifact.kind = rund::kernel::LoweringArtifactKind::MetalSource;
  artifact.source_text = "constant uint RundStride_read_78 = 4u;\n"
                         "constant uint RundBase_read_78 = 0u;\n"
                         "constant uint RundStride_write_79 = 4u;\n"
                         "constant uint RundBase_write_79 = 0u;\n"
                         "kernel void rund_compute_map_test(\n"
                         "    const device uchar* read_78 [[buffer(1)]],\n"
                         "    device uchar* write_79 [[buffer(2)]],\n"
                         "    uint gid [[thread_position_in_grid]]) {}\n";
  artifact.source_text_upper_bytes = artifact.source_text.size();
  artifact.metadata.input_element_bytes = {4u};
  artifact.metadata.output_element_bytes = {4u};
  artifact.metadata.binding_accesses = {
      rund::kernel::ComputeBindingAccess::Read,
      rund::kernel::ComputeBindingAccess::Write};
  artifact.metadata.binding_names = {"x", "y"};
  artifact.metadata.ok = true;
  artifact.metadata.reason = "ok";
  artifact.ok = true;
  artifact.reason = "ok";

  rund::kernel::ComputePlan plan{
      .api = rund::kernel::ComputeApi::Metal,
      .input_buffer_count = 1u,
      .output_buffer_count = 1u,
  };
  KernelExecutionStep step{};
  step.artifact = artifact;
  std::uint64_t retained = 0u;
  std::uint64_t transient = std::numeric_limits<std::uint64_t>::max();
  constexpr std::uint64_t LiteralGrowthPerBinding = 38u;
  if (!backend_template_plan::map_source_upper(step, plan, retained,
                                               transient) ||
      retained != artifact.source_text.size() + 2u * LiteralGrowthPerBinding ||
      transient != 0u) {
    return false;
  }
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const PreparedBackendManifest map_manifest =
      BuildMetalBackendManifest(step, plan, nullptr, 1u);
  if (!map_manifest.ok || map_manifest.capture_binding_slot_upper != 4u ||
      map_manifest.cold_source_transient_bytes != 0u ||
      map_manifest.cold_cache_source_storage_bytes <=
          map_manifest.cold_cache_source_bytes) {
    return false;
  }
  KernelExecutionStep controlled_step = step;
  controlled_step.control.count_source = rund::kernel::GraphControlSource::U32;
  controlled_step.control.count_binding = 0u;
  controlled_step.control.capacity = 1u;
  const PreparedBackendManifest controlled_manifest =
      BuildMetalBackendManifest(controlled_step, plan, nullptr, 1u);
  rund::kernel::ComputePlan binding_overflow = plan;
  binding_overflow.input_buffer_count = kMetalPipelineGuardBinding;
  const PreparedBackendManifest rejected_manifest =
      BuildMetalBackendManifest(step, binding_overflow, nullptr, 1u);
  if (!controlled_manifest.ok ||
      controlled_manifest.capture_binding_slot_upper != 6u ||
      rejected_manifest.ok) {
    return false;
  }
#endif

  const std::array input{rund::kernel::ResidentBufferRef{
      .bytes = 64u,
      .offset_bytes = 3u,
      .element_bytes = 4u,
      .stride_bytes = 8u,
      .count = 8u,
      .usage = rund::kernel::kResidentUsageRead}};
  const std::array output{rund::kernel::ResidentBufferRef{
      .bytes = 64u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 8u,
      .usage = rund::kernel::kResidentUsageWrite}};
  const rund::kernel::BindingSet bindings{
      .resident_inputs =
          rund::kernel::ResidentBindingRange{.refs = input.data(),
                                             .storage_count = input.size(),
                                             .count = input.size()},
      .resident_outputs =
          rund::kernel::ResidentBindingRange{.refs = output.data(),
                                             .storage_count = output.size(),
                                             .count = output.size()},
  };
  const std::string expected =
      "constant uint RundStride_read_78 = 8u;\n"
      "constant uint RundBase_read_78 = 3u;\n"
      "constant uint RundStride_write_79 = 4u;\n"
      "constant uint RundBase_write_79 = 0u;\n"
      "kernel void rund_compute_map_test(\n"
      "    const device uchar* read_78 [[buffer(1)]],\n"
      "    device uint* write_79 [[buffer(2)]],\n"
      "    uint gid [[thread_position_in_grid]]) {}\n";
  const rund::kernel::LoweringArtifact specialized =
      SpecializeMap(artifact, plan, bindings, 16u);
  std::uint64_t specialized_storage_upper = 0u;
  if (!specialized.ok || specialized.source_text != expected ||
      specialized.source_text.size() != expected.size() ||
      specialized.source_text_upper_bytes != retained ||
      !backend_source_recipe::string_external_storage_upper_bytes(
          retained, specialized_storage_upper) ||
      !backend_source_recipe::string_external_storage_within(
          specialized.source_text, specialized_storage_upper)) {
    return false;
  }

  const auto offset_unaligned = SpecializeMap(artifact, plan, bindings, 1u);
  if (!offset_unaligned.ok ||
      offset_unaligned.source_text.find("const device uchar* read_78") ==
          std::string_view::npos ||
      offset_unaligned.source_text.find("device uint* write_79") ==
          std::string_view::npos ||
      offset_unaligned.source_text.find("RundBase_read_78 = 0u") ==
          std::string_view::npos) {
    return false;
  }

  auto aligned_input = input;
  aligned_input[0].offset_bytes = 4u;
  const rund::kernel::BindingSet aligned_bindings{
      .resident_inputs =
          rund::kernel::ResidentBindingRange{.refs = aligned_input.data(),
                                             .storage_count =
                                                 aligned_input.size(),
                                             .count = aligned_input.size()},
      .resident_outputs = bindings.resident_outputs,
  };
  const auto aligned = SpecializeMap(artifact, plan, aligned_bindings, 1u);
  if (!aligned.ok ||
      aligned.source_text.find("const device uint* read_78") ==
          std::string_view::npos ||
      aligned.source_text.find("device uint* write_79") ==
          std::string_view::npos) {
    return false;
  }

  auto wrong_buffer_artifact = artifact;
  constexpr std::string_view OutputBinding = "write_79 [[buffer(";
  const std::size_t output_buffer =
      wrong_buffer_artifact.source_text.find("write_79 [[buffer(2)]]");
  if (output_buffer == std::string::npos) {
    return false;
  }
  wrong_buffer_artifact.source_text[output_buffer + OutputBinding.size()] = '3';
  if (SpecializeMap(wrong_buffer_artifact, plan, aligned_bindings, 1u).ok) {
    return false;
  }

  auto stride_unaligned_input = aligned_input;
  stride_unaligned_input[0].stride_bytes = 6u;
  const rund::kernel::BindingSet stride_unaligned_bindings{
      .resident_inputs =
          rund::kernel::ResidentBindingRange{
              .refs = stride_unaligned_input.data(),
              .storage_count = stride_unaligned_input.size(),
              .count = stride_unaligned_input.size()},
      .resident_outputs = bindings.resident_outputs,
  };
  const auto stride_unaligned =
      SpecializeMap(artifact, plan, stride_unaligned_bindings, 1u);
  if (!stride_unaligned.ok ||
      stride_unaligned.source_text.find("const device uchar* read_78") ==
          std::string_view::npos ||
      stride_unaligned.source_text.find("device uint* write_79") ==
          std::string_view::npos) {
    return false;
  }

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalAdapter adapter{};
  MetalMapTemplateResources prepared{};
  prepared.adapter = &adapter;
  prepared.plan = plan;
  prepared.input_strides = {8u};
  prepared.output_strides = {4u};
  prepared.output_word_mask = 1u;
  if (!MetalMapTemplateMatches(prepared, adapter, plan, bindings) ||
      MetalMapTemplateMatches(prepared, adapter, plan, aligned_bindings)) {
    return false;
  }
#endif

  rund::kernel::ComputePlan oversized = plan;
  oversized.input_buffer_count = rund::kernel::kMaxComputeBindingCount + 1u;
  oversized.output_buffer_count = 0u;
  const rund::kernel::LoweringArtifact rejected =
      SpecializeMap(artifact, oversized, bindings, 16u);
  return !rejected.ok && rejected.reason != nullptr &&
         std::string_view{rejected.reason} == "compute_pipeline_capacity";
}

[[nodiscard]] bool MetalMapCheckSourceHasOneGuardAuthority() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using namespace rund::node::accel::detail;
  MetalMapTemplateResources prepared{};
  prepared.plan.scalar = rund::kernel::ComputeScalar::Lane32;
  prepared.plan.domain = rund::kernel::ComputeDomain::U32;
  prepared.plan.op_hash_hi = 0x6d61702e63686563ull;
  prepared.plan.op_hash_lo = 0x6b2e736f75726365ull;
  prepared.checks.push_back(MetalMapCheck{.binding = 0u, .limit = 123u});
  const rund::kernel::ResidentBufferRef input{
      .bytes = 64u,
      .element_bytes = 4u,
      .stride_bytes = 8u,
      .count = 8u,
      .usage = rund::kernel::kResidentUsageRead,
  };
  rund::kernel::BindingSet bindings{};
  bindings.resident_inputs = rund::kernel::ResidentBindingRange{
      .refs = &input,
      .storage_count = 1u,
      .count = 1u,
  };

  const KernelPreparationScope scope{KernelPreparationMode::PipelinePrivate};
  rund::kernel::LoweringArtifact artifact =
      MetalMapCheckArtifact(prepared, bindings);
  std::uint64_t raw_upper = 0u;
  std::uint64_t guarded_upper = 0u;
  const bool upper_ok =
      MetalMapCheckSourceUpperBytes(1u, DecimalDigitCount(input.stride_bytes),
                                    DecimalDigitCount(123u), raw_upper);
  const bool guarded_ok = upper_ok && PipelinePrivateMetalSourceUpperBytes(
                                          raw_upper, 1u, true, guarded_upper);
  if (!artifact.ok || !guarded_ok || artifact.source_text.size() != raw_upper ||
      artifact.source_text_upper_bytes != raw_upper ||
      guarded_upper <= raw_upper) {
    return false;
  }
  const std::size_t check_capacity = artifact.source_text.capacity();
  std::string guarded = PipelinePrivateMetalSource(
      std::move(artifact.source_text), guarded_upper);
  std::uint64_t guarded_storage_upper = 0u;
  if (guarded.size() != guarded_upper || guarded.capacity() != check_capacity ||
      guarded.find("buffer(30)") == std::string::npos ||
      !backend_source_recipe::string_external_storage_upper_bytes(
          guarded_upper, guarded_storage_upper) ||
      !backend_source_recipe::string_external_storage_within(
          guarded, guarded_storage_upper)) {
    return false;
  }

  std::string over_capacity = guarded;
  over_capacity.reserve(static_cast<std::size_t>(guarded_storage_upper + 64u));
  if (!PipelinePrivateMetalSource(std::move(over_capacity), guarded_upper)
           .empty()) {
    return false;
  }

  std::uint64_t control_guarded_upper = 0u;
  if (!PipelinePrivateMetalSourceUpperBytes(MetalMapControlSourceText().size(),
                                            1u, true, control_guarded_upper)) {
    return false;
  }
  std::string control_source = MetalMapControlSource();
  const std::size_t control_capacity = control_source.capacity();
  control_source = PipelinePrivateMetalSource(std::move(control_source),
                                              control_guarded_upper);
  if (control_source.size() != control_guarded_upper ||
      control_source.capacity() != control_capacity) {
    return false;
  }

  rund::kernel::LoweringArtifact controlled_input{};
  controlled_input.key.api = rund::kernel::ComputeApi::Metal;
  controlled_input.key.op_hash_hi = 0x1111111111111111ull;
  controlled_input.key.op_hash_lo = 0x2222222222222222ull;
  controlled_input.kind = rund::kernel::LoweringArtifactKind::MetalSource;
  constexpr std::string_view CanonicalControlledSource =
      "// artifact_variant=canonical\n"
      "kernel void "
      "rund_compute_map_1111111111111111_2222222222222222(\n"
      "    uint gid [[thread_position_in_grid]]) {\n"
      "}\n";
  controlled_input.source_text_upper_bytes = CanonicalControlledSource.size();
  controlled_input.ok = true;
  controlled_input.reason = "ok";
  const rund::kernel::ComputePlan controlled_plan{
      .api = rund::kernel::ComputeApi::Metal,
      .input_buffer_count = 1u,
      .output_buffer_count = 1u,
  };
  std::uint64_t controlled_upper = 0u;
  std::uint64_t controlled_guarded_upper = 0u;
  if (!MetalControlledMapSourceUpperBytes(controlled_plan,
                                          CanonicalControlledSource.size(),
                                          controlled_upper) ||
      !PipelinePrivateMetalSourceUpperBytes(controlled_upper, 1u, true,
                                            controlled_guarded_upper)) {
    return false;
  }
  const auto canonical_recipe =
      [source = CanonicalControlledSource]<typename Sink>(Sink &sink) noexcept(
          noexcept(sink.append(std::string_view{}))) {
        return sink.append(source);
      };
  controlled_input.source_text = backend_source_recipe::materialize(
      canonical_recipe, CanonicalControlledSource.size(),
      controlled_guarded_upper);
  const std::size_t controlled_capacity =
      controlled_input.source_text.capacity();
  rund::kernel::LoweringArtifact controlled =
      MetalControlledMapArtifact(std::move(controlled_input), controlled_plan);
  if (!controlled.ok || controlled.source_text.size() != controlled_upper ||
      controlled.source_text.capacity() != controlled_capacity ||
      controlled.source_text.find("artifact_variant=controlled") ==
          std::string::npos ||
      controlled.source_text.find("_controlled(") == std::string::npos) {
    return false;
  }
  controlled.source_text = PipelinePrivateMetalSource(
      std::move(controlled.source_text), controlled_guarded_upper);
  return controlled.source_text.size() == controlled_guarded_upper &&
         controlled.source_text.capacity() == controlled_capacity;
#else
  return true;
#endif
}

} // namespace node_accel_contract
