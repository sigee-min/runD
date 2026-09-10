#include "src/accel/kernel/backend/template/source.hpp"
#include "src/accel/kernel/backend/manifest.hpp"
#include "src/accel/kernel/backend/run.hpp"
#include "src/accel/kernel/backend/source/storage.hpp"
#include "src/accel/kernel/backend/template/identity.hpp"
#include "src/accel/kernel/prepared/template/registry.hpp"
#include "src/accel/kernel/step/map/stride.hpp"
#include "src/accel/kernel/step/map/stride/local.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "src/accel/metal/kernel/manifest.hpp"
#include "src/accel/metal/kernel/pipeline/source.hpp"
#include "src/accel/metal/runtime/api.hpp"
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

#include "../map.hpp"

namespace node_accel_contract {

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

} // namespace node_accel_contract
