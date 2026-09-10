#include "local.hpp"
#include "source/local.hpp"

#include "src/accel/range_aggregate/plan.hpp"
#include "src/accel/window/shape.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/window/plan.hpp>

#include <array>
#include <cstdio>
#include <limits>
#include <string>

namespace rund_node_test_pipeline_residency::device_vsm_test {

rund::kernel::LoweringArtifact
CanonicalArtifact(const rund::kernel::ComputeApi api) {
  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = api;
  artifact.key.variant = rund::kernel::LoweringArtifactVariant::Canonical;
  artifact.key.op_hash_hi = 0x11u;
  artifact.key.op_hash_lo = 0x22u;
  artifact.kind = api == rund::kernel::ComputeApi::Metal
                      ? rund::kernel::LoweringArtifactKind::MetalSource
                      : rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.metadata.ok = true;
  artifact.metadata.read_count = 1u;
  artifact.metadata.write_count = 1u;
  if (api == rund::kernel::ComputeApi::Metal) {
    artifact.source_text =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "// artifact_variant=canonical\n"
        "kernel void rund_compute_map_0000000000000011_0000000000000022(\n"
        "    constant uchar* rund_params [[buffer(0)]],\n"
        "    const device uint* read_input [[buffer(1)]],\n"
        "    device uint* write_output [[buffer(2)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "  write_output[gid] = read_input[gid] + 1u;\n"
        "}\n";
  } else {
    artifact.source_text =
        "#version 450\n"
        "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
        "// artifact_variant=canonical\n"
        "layout(set = 0, binding = 0, std430) readonly buffer Params { uint "
        "rund_params[]; };\n"
        "layout(set = 0, binding = 1, std430) readonly buffer Input { uint "
        "read_input[]; };\n"
        "layout(set = 0, binding = 2, std430) writeonly buffer Output { uint "
        "write_output[]; };\n"
        "layout(push_constant) uniform RundDispatch {\n"
        "  uint tile_count;\n"
        "  uint iterations;\n"
        "} rund_dispatch;\n"
        "void main() {\n"
        "  const uint gid = gl_GlobalInvocationID.x;\n"
        "  if (gid >= rund_dispatch.tile_count) { return; }\n"
        "  write_output[gid] = read_input[gid] + 1u;\n"
        "}\n";
  }
  artifact.source_text_upper_bytes = artifact.source_text.size();
  artifact.ok = true;
  artifact.reason = "ok";
  return artifact;
}

namespace {

[[nodiscard]] std::size_t Count(const std::string &text,
                                const std::string &needle) noexcept {
  std::size_t count = 0u;
  std::size_t cursor = 0u;
  while ((cursor = text.find(needle, cursor)) != std::string::npos) {
    ++count;
    cursor += needle.size();
  }
  return count;
}

[[nodiscard]] bool Transform(const rund::kernel::ComputeApi api) {
  auto artifact = CanonicalArtifact(api);
  if (!accel::TransformDeviceVsmSource(artifact, 1u, 1u) || !artifact.ok ||
      artifact.key.variant !=
          rund::kernel::LoweringArtifactVariant::DeviceVsm ||
      artifact.source_text.find("artifact_variant=device_vsm") ==
          std::string::npos ||
      artifact.source_text.find("rund_page += rund_vsm.width") ==
          std::string::npos ||
      artifact.source_text.find("rund_page_begin") == std::string::npos ||
      artifact.source_text.find("rund_vsm_result[5]") == std::string::npos ||
      artifact.source_text.find("rund_vsm_result[10]") == std::string::npos ||
      artifact.source_text.find("rund_vsm_result[11]") == std::string::npos ||
      artifact.source_text.find("rund_vsm_result[12]") == std::string::npos ||
      artifact.source_text.find("rund_vsm_ring_state") == std::string::npos ||
      artifact.source_text.find("rund_vsm_ring_scratch") == std::string::npos ||
      artifact.source_text.find("rund_vsm_input_scratch_0") ==
          std::string::npos ||
      artifact.source_text.find("rund_vsm_input_alias_0") ==
          std::string::npos ||
      artifact.source_text.find("rund_vsm_output_alias") == std::string::npos ||
      artifact.source_text.find("artifact_variant=canonical") !=
          std::string::npos ||
      artifact.source_text_upper_bytes < artifact.source_text.size()) {
    return false;
  }
  if (api == rund::kernel::ComputeApi::Metal) {
    return Count(artifact.source_text, "rund_compute_map_") == 1u &&
           artifact.source_text.find("_device_vsm(") != std::string::npos &&
           artifact.source_text.find("const ulong gid = ulong(rund_slot)") !=
               std::string::npos &&
           artifact.source_text.find(
               "rund_vsm_input_scratch_0[rund_scratch_word + rund_word] = "
               "rund_vsm_input_alias_0[rund_global_word + rund_word]") !=
               std::string::npos &&
           artifact.source_text.find("threadgroup_position_in_grid") !=
               std::string::npos &&
           artifact.source_text.find("gl_GlobalInvocationID") ==
               std::string::npos;
  }
  return artifact.source_text.find("gl_WorkGroupID.x") != std::string::npos &&
         artifact.source_text.find(
             "const uint gid = rund_slot * rund_vsm.payload_elements") !=
             std::string::npos &&
         artifact.source_text.find(
             "rund_vsm_input_scratch_0[rund_scratch_word + rund_word] = "
             "rund_vsm_input_alias_0[rund_global_word + rund_word]") !=
             std::string::npos &&
         artifact.source_text.find("RundDeviceVsmDispatch") !=
             std::string::npos &&
         artifact.source_text.find("rund_dispatch.tile_count") ==
             std::string::npos;
}

[[nodiscard]] bool TransactionalFailure() {
  auto artifact = CanonicalArtifact(rund::kernel::ComputeApi::Metal);
  artifact.source_text += "// artifact_variant=canonical\n";
  const auto before = artifact;
  return !accel::TransformDeviceVsmSource(artifact, 1u, 1u) &&
         artifact.key == before.key &&
         artifact.source_text == before.source_text &&
         artifact.source_text_upper_bytes == before.source_text_upper_bytes;
}

[[nodiscard]] constexpr accel::DeviceVsmPageGeometry WindowGeometry() noexcept {
  constexpr std::uint64_t FrameElements = 24u;
  constexpr std::uint64_t PayloadElements = 16u;
  constexpr std::uint64_t Radius = 4u;
  return accel::DeviceVsmPageGeometry{
      .logical_bytes = 5u * PayloadElements * sizeof(std::uint32_t) - 12u,
      .payload_bytes = PayloadElements * sizeof(std::uint32_t),
      .frame_bytes = FrameElements * sizeof(std::uint32_t),
      .read_prefix_bytes = Radius * sizeof(std::uint32_t),
      .target_offset_bytes = Radius * sizeof(std::uint32_t),
      .read_suffix_bytes = Radius * sizeof(std::uint32_t),
      .page_count = 5u,
      .element_bytes = sizeof(std::uint32_t),
  };
}

[[nodiscard]] constexpr accel::DeviceVsmPageGeometry
MultipassWindowGeometry() noexcept {
  constexpr std::uint64_t FrameElements = 16u;
  constexpr std::uint64_t PayloadElements = 12u;
  constexpr std::uint64_t Radius = 2u;
  return accel::DeviceVsmPageGeometry{
      .logical_bytes = 5u * PayloadElements * sizeof(std::uint32_t) - 12u,
      .payload_bytes = PayloadElements * sizeof(std::uint32_t),
      .frame_bytes = FrameElements * sizeof(std::uint32_t),
      .read_prefix_bytes = Radius * sizeof(std::uint32_t),
      .target_offset_bytes = Radius * sizeof(std::uint32_t),
      .read_suffix_bytes = Radius * sizeof(std::uint32_t),
      .page_count = 5u,
      .element_bytes = sizeof(std::uint32_t),
  };
}

[[nodiscard]] bool
WindowSource(const rund::kernel::ComputeApi api,
             const accel::RangeBoundary boundary, const bool shared,
             const accel::DeviceVsmWindowFusion fusion = {}) {
  constexpr std::uint64_t FrameElements = 24u;
  constexpr std::uint64_t PayloadElements = 16u;
  constexpr std::uint64_t Radius = 4u;
  const rund::kernel::WindowPlan semantic =
      rund::kernel::PlanWindow(rund::kernel::WindowDesc{
          .op = rund::kernel::WindowOp::Sum,
          .element = rund::kernel::WindowElement::U32,
          .boundary = boundary == accel::RangeBoundary::Clamp
                          ? rund::kernel::WindowBoundary::Clamp
                          : rund::kernel::WindowBoundary::Clip,
          .domain = rund::kernel::ComputeDomain::U32,
          .input_count = FrameElements,
          .output_count = FrameElements,
          .window_size = Radius * 2u + 1u,
          .stride = 1u,
          .pad_left = Radius,
      });
  const auto shape = accel::WindowRangeShape(semantic);
  const std::uint8_t support =
      accel::RangeSupportBit(accel::RangeSupport::Direct) |
      (shared ? accel::RangeSupportBit(accel::RangeSupport::SharedHalo) : 0u);
  const auto capabilities = accel::RangeCaps::gpu(
      api == rund::kernel::ComputeApi::Metal ? accel::RangeSource::Metal
                                             : accel::RangeSource::Vulkan,
      accel::kRangeWidth64Bit, 64u, shared ? 4u : 0u, shared ? 32768u : 0u,
      std::numeric_limits<std::uint32_t>::max(),
      std::numeric_limits<std::uint32_t>::max(), support);
  if (!semantic || !shape.has_value() || !capabilities.has_value()) {
    return false;
  }
  const accel::RangePlan range = accel::PlanRange(*shape, *capabilities);
  const accel::DeviceVsmWindowArtifact built =
      accel::BuildDeviceVsmWindowArtifact(range, semantic, WindowGeometry(),
                                          fusion);
  const accel::DeviceVsmWindowArtifact plain =
      fusion.active() ? accel::BuildDeviceVsmWindowArtifact(range, semantic,
                                                            WindowGeometry())
                      : accel::DeviceVsmWindowArtifact{};
  const accel::DeviceVsmWindowArtifact malformed =
      fusion.active()
          ? accel::BuildDeviceVsmWindowArtifact(
                range, semantic, WindowGeometry(),
                accel::DeviceVsmWindowFusion{
                    .before = {.kind =
                                   static_cast<accel::DeviceVsmWindowMapKind>(
                                       255u),
                               .immediate = 3u},
                    .after = fusion.after,
                    .stage_count = fusion.stage_count})
          : accel::DeviceVsmWindowArtifact{};
  const accel::DeviceVsmWindowArtifact gapped =
      fusion.before_third.active()
          ? accel::BuildDeviceVsmWindowArtifact(
                range, semantic, WindowGeometry(),
                accel::DeviceVsmWindowFusion{.before = fusion.before,
                                             .before_third =
                                                 fusion.before_third,
                                             .stage_count = 3u})
          : accel::DeviceVsmWindowArtifact{};
  const bool selected =
      shared ? range.candidate().disposition() == accel::RangePath::SharedHalo
             : range.candidate().disposition() == accel::RangePath::Direct;
  return range.ok() && selected && built && built.proof.shared_halo == shared &&
         built.proof.fusion == fusion &&
         built.proof.semantic.boundary == semantic.boundary &&
         built.plan.param_bytes == accel::DeviceVsmWindowParameterBytes &&
         built.window.tile_count == PayloadElements &&
         built.artifact.source_text.find("rund_page += rund_vsm.width") !=
             std::string::npos &&
         built.artifact.source_text.find("rund_vsm_result[5]") !=
             std::string::npos &&
         (!fusion.active() ||
          (plain && !malformed && !gapped &&
           plain.artifact.key != built.artifact.key &&
           built.artifact.source_text.find(" + 3u") != std::string::npos &&
           built.artifact.source_text.find(" * 2u") != std::string::npos &&
           (!fusion.before_third.active() ||
            (built.artifact.source_text.find(" + 5u") != std::string::npos &&
             built.artifact.source_text.find(" * 13u") !=
                 std::string::npos)))) &&
         (shared == (built.artifact.source_text.find("range_tile[") !=
                     std::string::npos));
}

[[nodiscard]] bool
WindowMultipassSource(const rund::kernel::ComputeApi api,
                      const rund::kernel::WindowOp operation) {
  constexpr std::uint64_t FrameElements = 16u;
  constexpr std::uint64_t Radius = 2u;
  const rund::kernel::WindowPlan semantic =
      rund::kernel::PlanWindow(rund::kernel::WindowDesc{
          .op = operation,
          .element = rund::kernel::WindowElement::U32,
          .boundary = rund::kernel::WindowBoundary::Clip,
          .domain = rund::kernel::ComputeDomain::U32,
          .input_count = FrameElements,
          .output_count = FrameElements,
          .window_size = Radius * 2u + 1u,
          .stride = 1u,
          .pad_left = Radius,
      });
  const auto shape = accel::WindowRangeShape(semantic);
  const std::uint8_t support =
      accel::RangeSupportBit(accel::RangeSupport::Direct) |
      accel::RangeSupportBit(accel::RangeSupport::PrefixDifference) |
      accel::RangeSupportBit(accel::RangeSupport::BlockPrefixSuffix);
  const auto capabilities = accel::RangeCaps::gpu(
      api == rund::kernel::ComputeApi::Metal ? accel::RangeSource::Metal
                                             : accel::RangeSource::Vulkan,
      accel::kRangeWidth64Bit, 64u, 4u, 32768u,
      std::numeric_limits<std::uint32_t>::max(),
      std::numeric_limits<std::uint32_t>::max(), support);
  if (!semantic || !shape.has_value() || !capabilities.has_value()) {
    return false;
  }
  const accel::RangePlan range = accel::PlanRange(*shape, *capabilities);
  const accel::DeviceVsmWindowArtifact built =
      accel::BuildDeviceVsmWindowArtifact(range, semantic,
                                          MultipassWindowGeometry());
  const accel::RangePath expected = operation == rund::kernel::WindowOp::Sum
                                        ? accel::RangePath::PrefixDifference
                                        : accel::RangePath::BlockPrefixSuffix;
  const std::string marker =
      operation == rund::kernel::WindowOp::Sum
          ? (api == rund::kernel::ComputeApi::Metal ? "prefix += sample"
                                                    : "prefix += rund_sample")
          : "page_cursor = rund_vsm.page_count";
  return range.ok() && range.candidate().disposition() == expected && built &&
         built.proof.range_path == expected && built.proof.mutates_input &&
         built.proof.range_stage_count >= 2u &&
         built.artifact.source_text.find(marker) != std::string::npos;
}

} // namespace

bool CheckSource() {
  const accel::DeviceVsmWindowFusion fusion{
      .before = {.kind = accel::DeviceVsmWindowMapKind::AddWrapU32Immediate,
                 .immediate = 3u},
      .after = {.kind = accel::DeviceVsmWindowMapKind::MulWrapU32Immediate,
                .immediate = 2u},
      .stage_count = 3u};
  const accel::DeviceVsmWindowFusion maximum_fusion{
      .before = {.kind = accel::DeviceVsmWindowMapKind::AddWrapU32Immediate,
                 .immediate = 3u},
      .before_second = {.kind =
                            accel::DeviceVsmWindowMapKind::MulWrapU32Immediate,
                        .immediate = 2u},
      .before_third = {.kind =
                           accel::DeviceVsmWindowMapKind::AddWrapU32Immediate,
                       .immediate = 5u},
      .after = {.kind = accel::DeviceVsmWindowMapKind::MulWrapU32Immediate,
                .immediate = 7u},
      .after_second = {.kind =
                           accel::DeviceVsmWindowMapKind::AddWrapU32Immediate,
                       .immediate = 11u},
      .after_third = {.kind =
                          accel::DeviceVsmWindowMapKind::MulWrapU32Immediate,
                      .immediate = 13u},
      .stage_count = 7u};
  const std::array checks{
      Transform(rund::kernel::ComputeApi::Metal),
      Transform(rund::kernel::ComputeApi::Vulkan),
      WindowSource(rund::kernel::ComputeApi::Metal, accel::RangeBoundary::Clamp,
                   true),
      WindowSource(rund::kernel::ComputeApi::Vulkan,
                   accel::RangeBoundary::Clamp, false),
      WindowMultipassSource(rund::kernel::ComputeApi::Metal,
                            rund::kernel::WindowOp::Sum),
      WindowMultipassSource(rund::kernel::ComputeApi::Vulkan,
                            rund::kernel::WindowOp::Sum),
      WindowMultipassSource(rund::kernel::ComputeApi::Metal,
                            rund::kernel::WindowOp::Min),
      WindowMultipassSource(rund::kernel::ComputeApi::Vulkan,
                            rund::kernel::WindowOp::Max),
      WindowSource(rund::kernel::ComputeApi::Metal, accel::RangeBoundary::Clamp,
                   true, fusion),
      WindowSource(rund::kernel::ComputeApi::Vulkan,
                   accel::RangeBoundary::Clamp, true, fusion),
      WindowSource(rund::kernel::ComputeApi::Metal, accel::RangeBoundary::Clamp,
                   true, maximum_fusion),
      WindowSource(rund::kernel::ComputeApi::Vulkan,
                   accel::RangeBoundary::Clamp, false, maximum_fusion),
      source_detail::CheckGraphMapReduceSource(rund::kernel::ComputeApi::Metal),
      source_detail::CheckGraphMapReduceSource(
          rund::kernel::ComputeApi::Vulkan),
      TransactionalFailure(),
  };
  for (std::size_t index = 0u; index < checks.size(); ++index) {
    if (!checks[index]) {
      std::fprintf(stderr, "DeviceVsm source check=%zu failed\n", index);
      return false;
    }
  }
  return true;
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
