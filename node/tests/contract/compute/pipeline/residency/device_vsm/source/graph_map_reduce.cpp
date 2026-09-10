#include "local.hpp"

#include "src/accel/kernel/residency/device_vsm/source/graph_map_reduce.hpp"

#include <kernel/program/compute/lowering/model.hpp>
#include <kernel/program/compute/lowering/names.hpp>

#include <cstdio>
#include <string>

namespace rund_node_test_pipeline_residency::device_vsm_test::source_detail {

[[nodiscard]] bool
CheckGraphMapReduceSource(const rund::kernel::ComputeApi api) {
  auto artifact = CanonicalArtifact(api);
  artifact.key.scalar = rund::kernel::ComputeScalar::Lane64;
  artifact.key.domain = rund::kernel::ComputeDomain::U64;
  constexpr std::uint64_t Immediate = 0xfedcba9876543210ull;
  const accel::MapSemantic semantic{
      .kind = accel::MapSemanticKind::AddWrapU64Immediate,
      .recurrence_total = true,
      .immediate = static_cast<std::uint32_t>(Immediate),
      .maximum = static_cast<std::uint32_t>(Immediate >> 32u),
  };
  using rund::kernel::IrOp;
  namespace lowering = rund::kernel::compute_lowering_detail;
  constexpr std::uint8_t U64Mode = lowering::DomainModeFor(
      rund::kernel::ComputeScalar::Lane64, rund::kernel::ComputeDomain::U64);
  lowering::ComputeInputAdmission input{
      .key = artifact.key,
      .parsed =
          lowering::ParsedIR{
              .name = "device-vsm-typed-map",
              .scalar_mode = U64Mode,
              .bindings = {lowering::ParsedBinding{
                               .kind = lowering::kReadBindingKind,
                               .numeric_mode = U64Mode,
                               .name = "input",
                               .element_bytes = sizeof(std::uint64_t)},
                           lowering::ParsedBinding{
                               .kind = lowering::kWriteBindingKind,
                               .numeric_mode = U64Mode,
                               .name = "output",
                               .element_bytes = sizeof(std::uint64_t)}},
              .nodes = {lowering::ParsedNode{
                            .op = static_cast<std::uint8_t>(IrOp::Read),
                            .aux = 0u},
                        lowering::ParsedNode{
                            .op = static_cast<std::uint8_t>(IrOp::Constant),
                            .lhs = static_cast<std::uint32_t>(Immediate),
                            .rhs =
                                static_cast<std::uint32_t>(Immediate >> 32u)},
                        lowering::ParsedNode{
                            .op = static_cast<std::uint8_t>(IrOp::Add),
                            .lhs = 1u,
                            .rhs = 2u},
                        lowering::ParsedNode{
                            .op = static_cast<std::uint8_t>(IrOp::Write),
                            .lhs = 3u,
                            .rhs =
                                static_cast<std::uint32_t>(
                                    rund::kernel::IrWriteMode::Value),
                            .aux = 1u}},
              .ok = true,
              .reason = "ok"},
      .ok = true,
      .reason = "ok"};
  constexpr accel::DeviceVsmPageGeometry geometry{
      .logical_bytes = 5u * 16u * sizeof(std::uint64_t) - 24u,
      .payload_bytes = 16u * sizeof(std::uint64_t),
      .frame_bytes = 16u * sizeof(std::uint64_t),
      .page_count = 5u,
      .element_bytes = sizeof(std::uint64_t),
  };
  constexpr accel::DeviceVsmGraphWavefrontProof wavefront{
      .same_dispatch = {0u, 1u},
      .prior_dispatch = {2u, 0u},
      .map_stage = 0u,
      .collective_stage = 1u,
      .stage_count = 2u,
      .frame_capacity = 2u,
      .batch_count = 3u,
  };
  const auto reduce = [](const rund::kernel::ReduceOp operation) {
    return rund::kernel::ReducePlan{
        .op = operation,
        .element = rund::kernel::ReduceElement::U64,
        .element_count = 16u,
        .element_bytes = sizeof(std::uint64_t),
        .count_source = rund::kernel::ComputeCountSource::BufferU64,
        .ok = true,
    };
  };
  const accel::DeviceVsmGraphMapReduceArtifact built =
      accel::BuildDeviceVsmGraphMapReduceArtifact(
          artifact, input, semantic, reduce(rund::kernel::ReduceOp::Sum),
          geometry, wavefront);
  auto invalid = semantic;
  invalid.recurrence_total = false;
  const accel::DeviceVsmGraphMapReduceArtifact rejected =
      accel::BuildDeviceVsmGraphMapReduceArtifact(
          artifact, input, invalid, reduce(rund::kernel::ReduceOp::Sum),
          geometry, wavefront);
  auto wrong_mode = input;
  wrong_mode.parsed.scalar_mode = lowering::DomainModeFor(
      rund::kernel::ComputeScalar::Lane64, rund::kernel::ComputeDomain::U32);
  const accel::DeviceVsmGraphMapReduceArtifact mismatched =
      accel::BuildDeviceVsmGraphMapReduceArtifact(
          artifact, wrong_mode, semantic, reduce(rund::kernel::ReduceOp::Sum),
          geometry, wavefront);
  auto swapped_roles = wavefront;
  swapped_roles.map_stage = 1u;
  swapped_roles.collective_stage = 0u;
  const accel::DeviceVsmGraphMapReduceArtifact reordered =
      accel::BuildDeviceVsmGraphMapReduceArtifact(
          artifact, input, semantic, reduce(rund::kernel::ReduceOp::Sum),
          geometry, swapped_roles);
  const accel::DeviceVsmGraphMapReduceArtifact count =
      accel::BuildDeviceVsmGraphMapReduceArtifact(
          artifact, input, semantic,
          reduce(rund::kernel::ReduceOp::CountNonzero), geometry, wavefront);
  const accel::DeviceVsmGraphMapReduceArtifact minimum =
      accel::BuildDeviceVsmGraphMapReduceArtifact(
          artifact, input, semantic, reduce(rund::kernel::ReduceOp::Min),
          geometry, wavefront);
  const accel::DeviceVsmGraphMapReduceArtifact maximum =
      accel::BuildDeviceVsmGraphMapReduceArtifact(
          artifact, input, semantic, reduce(rund::kernel::ReduceOp::Max),
          geometry, wavefront);
  const bool valid =
      built && count && minimum && maximum && !rejected && !mismatched &&
      !reordered && built.proof.semantic.op == rund::kernel::ReduceOp::Sum &&
      count.proof.semantic.op == rund::kernel::ReduceOp::CountNonzero &&
      minimum.proof.semantic.op == rund::kernel::ReduceOp::Min &&
      maximum.proof.semantic.op == rund::kernel::ReduceOp::Max &&
      built.proof.workgroup_width == 256u &&
      built.proof.wavefront == wavefront && built.plan.dispatch_count == 1u &&
      built.artifact.source_text.find("device_vsm.typed_map") !=
          std::string::npos &&
      built.artifact.source_text.find("result[8]") != std::string::npos &&
      built.artifact.source_text.find("node[3].op=add") != std::string::npos &&
      built.artifact.source_text.find("first_page = batch * 2u") !=
          std::string::npos &&
      built.artifact.source_text.find("if (selected == 0u)") !=
          std::string::npos &&
      built.artifact.source_text.find("else if (selected == 1u)") !=
          std::string::npos &&
      built.artifact.source_text.find("wavefront_valid == 0u") !=
          std::string::npos &&
      built.artifact.source_text.find("result[6]") != std::string::npos &&
      count.artifact.source_text.find("count_nonzero") != std::string::npos &&
      minimum.artifact.source_text.find("graph_map_reduce.min") !=
          std::string::npos &&
      maximum.artifact.source_text.find("graph_map_reduce.max") !=
          std::string::npos &&
      built.artifact.source_text.find("4275878552") != std::string::npos &&
      built.artifact.source_text.find("1985229328") != std::string::npos;
  if (!valid) {
    std::fprintf(stderr,
                 "DeviceVsm graph map/reduce api=%u built=%d(%s) "
                 "count=%d(%s) min=%d(%s) max=%d(%s) rejected=%d\n",
                 static_cast<unsigned>(api), static_cast<bool>(built),
                 built.reason, static_cast<bool>(count), count.reason,
                 static_cast<bool>(minimum), minimum.reason,
                 static_cast<bool>(maximum), maximum.reason,
                 static_cast<bool>(rejected));
  }
  return valid;
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test::source_detail
