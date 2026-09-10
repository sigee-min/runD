#include "local.hpp"

#include "src/accel/kernel/residency/device_vsm/source/graph_pointwise.hpp"

#include <kernel/program/compute/lowering/model.hpp>

#include <array>
#include <cstdint>
#include <cstdio>

namespace rund_node_test_pipeline_residency::device_vsm_test {
namespace {

using rund::kernel::IrOp;
namespace lowering = rund::kernel::compute_lowering_detail;

struct Stage final {
  rund::kernel::LoweringArtifact artifact{};
  lowering::ComputeInputAdmission input{};
  accel::MapSemantic semantic{};
};

[[nodiscard]] Stage BuildStage(const rund::kernel::ComputeApi api,
                               const std::uint64_t identity,
                               const bool indexed) {
  auto artifact = CanonicalArtifact(api);
  artifact.key.scalar = rund::kernel::ComputeScalar::Lane64;
  artifact.key.domain = rund::kernel::ComputeDomain::U64;
  artifact.key.op_hash_hi = identity;
  artifact.key.op_hash_lo = identity ^ 0x9e3779b97f4a7c15ull;
  artifact.key.canonical_ir_hash_hi = identity + 1u;
  artifact.key.canonical_ir_hash_lo = identity + 2u;
  constexpr std::uint8_t U64Mode = lowering::DomainModeFor(
      rund::kernel::ComputeScalar::Lane64, rund::kernel::ComputeDomain::U64);
  lowering::ParsedIR parsed{
      .name = indexed ? "device-vsm-graph-pointwise-indexed"
                      : "device-vsm-graph-pointwise-add",
      .scalar_mode = U64Mode,
      .bindings = {lowering::ParsedBinding{.kind = lowering::kReadBindingKind,
                                           .numeric_mode = U64Mode,
                                           .name = "input",
                                           .element_bytes =
                                               sizeof(std::uint64_t)},
                   lowering::ParsedBinding{.kind = lowering::kWriteBindingKind,
                                           .numeric_mode = U64Mode,
                                           .name = "output",
                                           .element_bytes =
                                               sizeof(std::uint64_t)}},
      .nodes = {lowering::ParsedNode{
                    .op = static_cast<std::uint8_t>(IrOp::Read), .aux = 0u},
                lowering::ParsedNode{
                    .op = static_cast<std::uint8_t>(IrOp::Constant), .lhs = 7u},
                lowering::ParsedNode{.op = static_cast<std::uint8_t>(IrOp::Add),
                                     .lhs = 1u,
                                     .rhs = 2u}},
      .ok = true,
      .reason = "ok"};
  if (indexed) {
    parsed.nodes.push_back(
        lowering::ParsedNode{.op = static_cast<std::uint8_t>(IrOp::Index)});
    parsed.nodes.push_back(lowering::ParsedNode{
        .op = static_cast<std::uint8_t>(IrOp::Add), .lhs = 3u, .rhs = 4u});
  }
  parsed.nodes.push_back(lowering::ParsedNode{
      .op = static_cast<std::uint8_t>(IrOp::Write),
      .lhs = indexed ? 5u : 3u,
      .rhs = static_cast<std::uint32_t>(rund::kernel::IrWriteMode::Value),
      .aux = 1u});
  return Stage{
      .artifact = artifact,
      .input =
          lowering::ComputeInputAdmission{
              .key = artifact.key,
              .parsed = std::move(parsed),
              .ok = true,
              .reason = "ok",
          },
      .semantic =
          accel::MapSemantic{
              .kind = accel::MapSemanticKind::AddWrapU64Immediate,
              .recurrence_total = true,
              .immediate = 7u,
          },
  };
}

[[nodiscard]] Stage BuildJoinStage(const rund::kernel::ComputeApi api,
                                   const std::uint64_t identity) {
  Stage stage = BuildStage(api, identity, false);
  constexpr std::uint8_t U64Mode = lowering::DomainModeFor(
      rund::kernel::ComputeScalar::Lane64, rund::kernel::ComputeDomain::U64);
  stage.artifact.metadata.read_count = 2u;
  stage.input.parsed.bindings.insert(
      stage.input.parsed.bindings.begin() + 1,
      lowering::ParsedBinding{.kind = lowering::kReadBindingKind,
                              .numeric_mode = U64Mode,
                              .name = "right",
                              .element_bytes = sizeof(std::uint64_t)});
  stage.input.parsed.nodes = {
      lowering::ParsedNode{.op = static_cast<std::uint8_t>(IrOp::Read),
                           .aux = 0u},
      lowering::ParsedNode{.op = static_cast<std::uint8_t>(IrOp::Read),
                           .aux = 1u},
      lowering::ParsedNode{
          .op = static_cast<std::uint8_t>(IrOp::Add), .lhs = 1u, .rhs = 2u},
      lowering::ParsedNode{
          .op = static_cast<std::uint8_t>(IrOp::Write),
          .lhs = 3u,
          .rhs = static_cast<std::uint32_t>(rund::kernel::IrWriteMode::Value),
          .aux = 2u},
  };
  return stage;
}

[[nodiscard]] constexpr accel::DeviceVsmPageGeometry Geometry() noexcept {
  return accel::DeviceVsmPageGeometry{
      .logical_bytes = 5u * 16u * sizeof(std::uint64_t) - 56u,
      .payload_bytes = 16u * sizeof(std::uint64_t),
      .frame_bytes = 16u * sizeof(std::uint64_t),
      .page_count = 5u,
      .element_bytes = sizeof(std::uint64_t),
  };
}

[[nodiscard]] constexpr accel::DeviceVsmGraphWavefrontProof
Wavefront() noexcept {
  return accel::DeviceVsmGraphWavefrontProof{
      .same_dispatch = {0u, 0u, 3u},
      .prior_dispatch = {4u, 0u, 0u},
      .map_stage = 0u,
      .collective_stage = 2u,
      .stage_count = 3u,
      .frame_capacity = 2u,
      .batch_count = 3u,
  };
}

[[nodiscard]] constexpr accel::DeviceVsmGraphPointwiseTopology
Topology() noexcept {
  using Kind = accel::DeviceVsmGraphValueSourceKind;
  return accel::DeviceVsmGraphPointwiseTopology{
      .stages = {accel::DeviceVsmGraphPointwiseStageTopology{
                     .inputs = {accel::DeviceVsmGraphValueSource{
                         .kind = Kind::ExternalInput, .index = 0u}},
                     .input_count = 1u},
                 accel::DeviceVsmGraphPointwiseStageTopology{
                     .inputs = {accel::DeviceVsmGraphValueSource{
                         .kind = Kind::ExternalInput, .index = 0u}},
                     .input_count = 1u},
                 accel::DeviceVsmGraphPointwiseStageTopology{
                     .inputs = {accel::DeviceVsmGraphValueSource{
                                    .kind = Kind::StageOutput, .index = 0u},
                                accel::DeviceVsmGraphValueSource{
                                    .kind = Kind::StageOutput, .index = 1u}},
                     .input_count = 2u}},
      .stage_count = 3u,
      .external_input_count = 1u,
  };
}

[[nodiscard]] constexpr accel::DeviceVsmGraphWavefrontProof
FrontierWavefront() noexcept {
  return accel::DeviceVsmGraphWavefrontProof{
      .same_dispatch = {0u, 1u, 2u, 4u, 8u},
      .map_stage = 0u,
      .collective_stage = 4u,
      .stage_count = 5u,
      .frame_capacity = 2u,
      .batch_count = 3u,
  };
}

[[nodiscard]] constexpr accel::DeviceVsmGraphPointwiseTopology
FrontierTopology() noexcept {
  using Kind = accel::DeviceVsmGraphValueSourceKind;
  return accel::DeviceVsmGraphPointwiseTopology{
      .stages = {accel::DeviceVsmGraphPointwiseStageTopology{
                     .inputs = {accel::DeviceVsmGraphValueSource{
                         .kind = Kind::ExternalInput, .index = 0u}},
                     .input_count = 1u},
                 accel::DeviceVsmGraphPointwiseStageTopology{
                     .inputs = {accel::DeviceVsmGraphValueSource{
                                    .kind = Kind::StageOutput, .index = 0u},
                                accel::DeviceVsmGraphValueSource{
                                    .kind = Kind::ExternalInput, .index = 1u}},
                     .input_count = 2u},
                 accel::DeviceVsmGraphPointwiseStageTopology{
                     .inputs = {accel::DeviceVsmGraphValueSource{
                                    .kind = Kind::StageOutput, .index = 1u},
                                accel::DeviceVsmGraphValueSource{
                                    .kind = Kind::ExternalInput, .index = 2u}},
                     .input_count = 2u},
                 accel::DeviceVsmGraphPointwiseStageTopology{
                     .inputs = {accel::DeviceVsmGraphValueSource{
                         .kind = Kind::StageOutput, .index = 2u}},
                     .input_count = 1u},
                 accel::DeviceVsmGraphPointwiseStageTopology{
                     .inputs = {accel::DeviceVsmGraphValueSource{
                         .kind = Kind::StageOutput, .index = 3u}},
                     .input_count = 1u}},
      .stage_count = 5u,
      .external_input_count = 3u,
  };
}

[[nodiscard]] std::array<accel::DeviceVsmGraphPointwiseStage, 3u>
Authorities(const Stage &first, const Stage &second,
            const Stage &third) noexcept {
  return {accel::DeviceVsmGraphPointwiseStage{.artifact = &first.artifact,
                                              .input = &first.input,
                                              .semantic = &first.semantic},
          accel::DeviceVsmGraphPointwiseStage{.artifact = &second.artifact,
                                              .input = &second.input,
                                              .semantic = &second.semantic},
          accel::DeviceVsmGraphPointwiseStage{.artifact = &third.artifact,
                                              .input = &third.input,
                                              .semantic = &third.semantic}};
}

[[nodiscard]] std::array<accel::DeviceVsmGraphPointwiseStage, 5u>
FrontierAuthorities(const Stage &first, const Stage &second, const Stage &third,
                    const Stage &fourth, const Stage &fifth) noexcept {
  return {accel::DeviceVsmGraphPointwiseStage{.artifact = &first.artifact,
                                              .input = &first.input,
                                              .semantic = &first.semantic},
          accel::DeviceVsmGraphPointwiseStage{.artifact = &second.artifact,
                                              .input = &second.input,
                                              .semantic = &second.semantic},
          accel::DeviceVsmGraphPointwiseStage{.artifact = &third.artifact,
                                              .input = &third.input,
                                              .semantic = &third.semantic},
          accel::DeviceVsmGraphPointwiseStage{.artifact = &fourth.artifact,
                                              .input = &fourth.input,
                                              .semantic = &fourth.semantic},
          accel::DeviceVsmGraphPointwiseStage{.artifact = &fifth.artifact,
                                              .input = &fifth.input,
                                              .semantic = &fifth.semantic}};
}

[[nodiscard]] bool CheckApi(const rund::kernel::ComputeApi api) {
  const Stage first = BuildStage(api, 0x101u, false);
  const Stage second = BuildStage(api, 0x202u, true);
  const Stage third = BuildJoinStage(api, 0x303u);
  const auto stages = Authorities(first, second, third);
  const accel::DeviceVsmGraphPointwiseArtifact built =
      accel::BuildDeviceVsmGraphPointwiseArtifact(stages, Topology(),
                                                  Geometry(), Wavefront());
  const Stage frontier_first = BuildStage(api, 0x401u, false);
  const Stage frontier_second = BuildJoinStage(api, 0x402u);
  const Stage frontier_third = BuildJoinStage(api, 0x403u);
  const Stage frontier_fourth = BuildStage(api, 0x404u, true);
  const Stage frontier_fifth = BuildStage(api, 0x405u, false);
  const auto frontier_stages =
      FrontierAuthorities(frontier_first, frontier_second, frontier_third,
                          frontier_fourth, frontier_fifth);
  const accel::DeviceVsmGraphPointwiseArtifact built_frontier =
      accel::BuildDeviceVsmGraphPointwiseArtifact(
          frontier_stages, FrontierTopology(), Geometry(), FrontierWavefront());
  auto parameterized = second;
  parameterized.artifact.metadata.param_storage.push_back(1u);
  const auto parameterized_stages = Authorities(first, parameterized, third);
  const accel::DeviceVsmGraphPointwiseArtifact rejected_parameter =
      accel::BuildDeviceVsmGraphPointwiseArtifact(
          parameterized_stages, Topology(), Geometry(), Wavefront());
  auto supported = second;
  supported.input.parsed.nodes[2u].op = static_cast<std::uint8_t>(IrOp::Mul);
  const auto supported_stages = Authorities(first, supported, third);
  const accel::DeviceVsmGraphPointwiseArtifact accepted_operation =
      accel::BuildDeviceVsmGraphPointwiseArtifact(supported_stages, Topology(),
                                                  Geometry(), Wavefront());
  auto unsupported = second;
  unsupported.input.parsed.nodes[2u].op =
      static_cast<std::uint8_t>(IrOp::DivUnsigned);
  const auto unsupported_stages = Authorities(first, unsupported, third);
  const accel::DeviceVsmGraphPointwiseArtifact rejected_operation =
      accel::BuildDeviceVsmGraphPointwiseArtifact(
          unsupported_stages, Topology(), Geometry(), Wavefront());
  auto wrong_api = second;
  wrong_api.artifact.key.api = api == rund::kernel::ComputeApi::Metal
                                   ? rund::kernel::ComputeApi::Vulkan
                                   : rund::kernel::ComputeApi::Metal;
  wrong_api.input.key = wrong_api.artifact.key;
  const auto wrong_api_stages = Authorities(first, wrong_api, third);
  const accel::DeviceVsmGraphPointwiseArtifact rejected_type =
      accel::BuildDeviceVsmGraphPointwiseArtifact(wrong_api_stages, Topology(),
                                                  Geometry(), Wavefront());
  auto wrong_wavefront = Wavefront();
  wrong_wavefront.stage_count = 2u;
  const accel::DeviceVsmGraphPointwiseArtifact rejected_wavefront =
      accel::BuildDeviceVsmGraphPointwiseArtifact(stages, Topology(),
                                                  Geometry(), wrong_wavefront);
  auto wrong_topology = Topology();
  wrong_topology.stages[2u].inputs[1u].index = 2u;
  const accel::DeviceVsmGraphPointwiseArtifact rejected_topology =
      accel::BuildDeviceVsmGraphPointwiseArtifact(stages, wrong_topology,
                                                  Geometry(), Wavefront());
  const bool valid =
      built && built.proof.stage_count == 3u &&
      built.proof.wavefront == Wavefront() && built.plan.dispatch_count == 1u &&
      built_frontier && built_frontier.proof.stage_count == 5u &&
      built_frontier.proof.topology == FrontierTopology() &&
      built_frontier.plan.input_buffer_count == 3u &&
      built_frontier.plan.dispatch_count == 1u &&
      built_frontier.artifact.source_text.find(
          "72756e645f67726170685f696e7075745f32") != std::string::npos &&
      !rejected_parameter && accepted_operation && !rejected_operation &&
      !rejected_type && !rejected_wavefront && !rejected_topology &&
      built.artifact.source_text.find("graph_pointwise.fused_dag") !=
          std::string::npos &&
      built.artifact.source_text.find("selected == 0") != std::string::npos &&
      built.artifact.source_text.find("result[8]") != std::string::npos &&
      built.artifact.source_text.find("result[10]") != std::string::npos &&
      built.artifact.source_text.find("result[11]") != std::string::npos &&
      built.artifact.source_text.find("result[12]") != std::string::npos &&
      built.artifact.source_text.find("ring_state") != std::string::npos &&
      built.artifact.source_text.find("ring_scratch") != std::string::npos &&
      built.artifact.source_text.find("input_scratch_0") != std::string::npos &&
      built.artifact.source_text.find("input_alias_0") != std::string::npos &&
      built.artifact.source_text.find("input_scratch_0[scratch_word + word] = "
                                      "input_alias_0[global_word + word]") !=
          std::string::npos &&
      built.artifact.source_text.find("output_alias") != std::string::npos;
  if (!valid) {
    std::fprintf(
        stderr,
        "DeviceVsm GraphPointwise source api=%u built=%d(%s) "
        "frontier=%d(%s) parameter=%d supported=%d unsupported=%d type=%d "
        "wavefront=%d topology=%d io=%d\n",
        static_cast<unsigned>(api), static_cast<bool>(built), built.reason,
        static_cast<bool>(built_frontier), built_frontier.reason,
        static_cast<bool>(rejected_parameter),
        static_cast<bool>(accepted_operation),
        static_cast<bool>(rejected_operation), static_cast<bool>(rejected_type),
        static_cast<bool>(rejected_wavefront),
        static_cast<bool>(rejected_topology),
        built_frontier.artifact.source_text.find(
            "72756e645f67726170685f696e7075745f32") != std::string::npos);
  }
  return valid;
}

} // namespace

bool CheckGraphPointwiseSource() {
  return CheckApi(rund::kernel::ComputeApi::Metal) &&
         CheckApi(rund::kernel::ComputeApi::Vulkan);
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
