#include "local.hpp"

#include "src/accel/kernel/residency/device_vsm/source/graph_map_scan.hpp"

#include <kernel/program/compute/lowering/model.hpp>
#include <kernel/program/compute/lowering/names.hpp>
#include <kernel/program/compute/scan/plan.hpp>

#include <cstdio>
#include <string>

namespace rund_node_test_pipeline_residency::device_vsm_test {
namespace {

using rund::kernel::IrOp;
namespace lowering = rund::kernel::compute_lowering_detail;

struct TypedMap final {
  rund::kernel::LoweringArtifact artifact{};
  lowering::ComputeInputAdmission input{};
  accel::MapSemantic semantic{};
};

[[nodiscard]] TypedMap Map(const rund::kernel::ComputeApi api) {
  constexpr std::uint64_t Mask = 0x55u;
  constexpr std::uint64_t Factor = 3u;
  constexpr std::uint64_t Add = 7u;
  constexpr std::uint8_t U64Mode = lowering::DomainModeFor(
      rund::kernel::ComputeScalar::Lane64, rund::kernel::ComputeDomain::U64);
  TypedMap map{};
  map.artifact = CanonicalArtifact(api);
  map.artifact.key.scalar = rund::kernel::ComputeScalar::Lane64;
  map.artifact.key.domain = rund::kernel::ComputeDomain::U64;
  map.input =
      lowering::ComputeInputAdmission{
          .key = map.artifact.key,
          .parsed =
              lowering::ParsedIR{
                  .name = "device-vsm-graph-map-scan",
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
                  .nodes =
                      {lowering::ParsedNode{
                           .op = static_cast<std::uint8_t>(IrOp::Read),
                           .aux = 0u},
                       lowering::ParsedNode{
                           .op = static_cast<std::uint8_t>(IrOp::Constant),
                           .lhs = static_cast<std::uint32_t>(Mask)},
                       lowering::ParsedNode{
                           .op = static_cast<std::uint8_t>(IrOp::BitXor),
                           .lhs = 1u,
                           .rhs = 2u},
                       lowering::ParsedNode{
                           .op = static_cast<std::uint8_t>(IrOp::Constant),
                           .lhs = static_cast<std::uint32_t>(Factor)},
                       lowering::ParsedNode{
                           .op = static_cast<std::uint8_t>(IrOp::Mul),
                           .lhs = 3u,
                           .rhs = 4u},
                       lowering::ParsedNode{
                           .op = static_cast<std::uint8_t>(IrOp::Constant),
                           .lhs = static_cast<std::uint32_t>(Add)},
                       lowering::ParsedNode{
                           .op = static_cast<std::uint8_t>(IrOp::Add),
                           .lhs = 5u,
                           .rhs = 6u},
                       lowering::ParsedNode{.op =
                                                static_cast<std::uint8_t>(
                                                    IrOp::Write),
                                            .lhs = 7u,
                                            .rhs = static_cast<std::uint32_t>(rund::
                                                                                  kernel::IrWriteMode::Value),
                                            .aux = 1u}},
                  .ok = true,
                  .reason = "ok"},
          .ok = true,
          .reason = "ok"};
  map.semantic.recurrence_total = true;
  return map;
}

[[nodiscard]] TypedMap BinaryMap(const rund::kernel::ComputeApi api) {
  constexpr std::uint8_t U64Mode = lowering::DomainModeFor(
      rund::kernel::ComputeScalar::Lane64, rund::kernel::ComputeDomain::U64);
  TypedMap map{};
  map.artifact = CanonicalArtifact(api);
  map.artifact.key.scalar = rund::kernel::ComputeScalar::Lane64;
  map.artifact.key.domain = rund::kernel::ComputeDomain::U64;
  map.artifact.metadata.read_count = 2u;
  map.input =
      lowering::ComputeInputAdmission{
          .key = map.artifact.key,
          .parsed =
              lowering::ParsedIR{
                  .name = "device-vsm-binary-map-scan",
                  .scalar_mode = U64Mode,
                  .bindings = {lowering::ParsedBinding{
                                   .kind = lowering::kReadBindingKind,
                                   .numeric_mode = U64Mode,
                                   .name = "left",
                                   .element_bytes = sizeof(std::uint64_t)},
                               lowering::ParsedBinding{
                                   .kind = lowering::kReadBindingKind,
                                   .numeric_mode = U64Mode,
                                   .name = "right",
                                   .element_bytes = sizeof(std::uint64_t)},
                               lowering::ParsedBinding{
                                   .kind = lowering::kWriteBindingKind,
                                   .numeric_mode = U64Mode,
                                   .name = "output",
                                   .element_bytes = sizeof(std::uint64_t)}},
                  .nodes =
                      {lowering::ParsedNode{
                           .op = static_cast<std::uint8_t>(IrOp::Read),
                           .aux = 0u},
                       lowering::ParsedNode{
                           .op = static_cast<std::uint8_t>(IrOp::Read),
                           .aux = 1u},
                       lowering::ParsedNode{.op =
                                                static_cast<std::
                                                                uint8_t>(IrOp::Add),
                                            .lhs = 1u,
                                            .rhs = 2u},
                       lowering::ParsedNode{.op =
                                                static_cast<std::uint8_t>(IrOp::Write),
                                            .lhs = 3u,
                                            .rhs = static_cast<std::uint32_t>(rund::kernel::IrWriteMode::Value),
                                            .aux = 2u}},
                  .ok = true,
                  .reason = "ok"},
          .ok = true,
          .reason = "ok"};
  map.semantic.recurrence_total = true;
  return map;
}

[[nodiscard]] TypedMap MaximumInputMap(const rund::kernel::ComputeApi api) {
  constexpr std::uint32_t InputCount = 7u;
  constexpr std::uint8_t U64Mode = lowering::DomainModeFor(
      rund::kernel::ComputeScalar::Lane64, rund::kernel::ComputeDomain::U64);
  TypedMap map{};
  map.artifact = CanonicalArtifact(api);
  map.artifact.key.scalar = rund::kernel::ComputeScalar::Lane64;
  map.artifact.key.domain = rund::kernel::ComputeDomain::U64;
  map.artifact.metadata.read_count = InputCount;
  map.input.key = map.artifact.key;
  map.input.parsed.name = "device-vsm-maximum-input-map-scan";
  map.input.parsed.scalar_mode = U64Mode;
  for (std::uint32_t input = 0u; input < InputCount; ++input) {
    map.input.parsed.bindings.push_back(lowering::ParsedBinding{
        .kind = lowering::kReadBindingKind,
        .numeric_mode = U64Mode,
        .name = "input" + std::to_string(input),
        .element_bytes = sizeof(std::uint64_t),
    });
    map.input.parsed.nodes.push_back(lowering::ParsedNode{
        .op = static_cast<std::uint8_t>(IrOp::Read), .aux = input});
  }
  map.input.parsed.bindings.push_back(lowering::ParsedBinding{
      .kind = lowering::kWriteBindingKind,
      .numeric_mode = U64Mode,
      .name = "output",
      .element_bytes = sizeof(std::uint64_t),
  });
  std::uint32_t sum = 1u;
  for (std::uint32_t input = 2u; input <= InputCount; ++input) {
    map.input.parsed.nodes.push_back(lowering::ParsedNode{
        .op = static_cast<std::uint8_t>(IrOp::Add),
        .lhs = sum,
        .rhs = input,
    });
    sum = static_cast<std::uint32_t>(map.input.parsed.nodes.size());
  }
  map.input.parsed.nodes.push_back(lowering::ParsedNode{
      .op = static_cast<std::uint8_t>(IrOp::Write),
      .lhs = sum,
      .rhs = static_cast<std::uint32_t>(rund::kernel::IrWriteMode::Value),
      .aux = InputCount,
  });
  map.input.parsed.ok = true;
  map.input.parsed.reason = "ok";
  map.input.ok = true;
  map.input.reason = "ok";
  map.semantic.recurrence_total = true;
  return map;
}

[[nodiscard]] rund::kernel::ScanPlan Scan(const rund::kernel::ScanOp op) {
  return rund::kernel::PlanScan({
      .op = op,
      .element = rund::kernel::ScanElement::U64,
      .element_count = 16u,
      .block_size = 8u,
  });
}

[[nodiscard]] accel::DeviceVsmPageGeometry
Geometry(const rund::kernel::ScanOp op) noexcept {
  const bool exclusive = op == rund::kernel::ScanOp::ExclusiveSum;
  const std::uint64_t element = sizeof(std::uint64_t);
  const std::uint64_t payload = (16u - (exclusive ? 1u : 0u)) * element;
  return accel::DeviceVsmPageGeometry{
      .logical_bytes = 5u * payload - 3u * element,
      .payload_bytes = payload,
      .frame_bytes = 16u * element,
      .read_prefix_bytes = exclusive ? element : 0u,
      .target_offset_bytes = exclusive ? element : 0u,
      .page_count = 5u,
      .element_bytes = sizeof(std::uint64_t),
  };
}

[[nodiscard]] bool Build(const rund::kernel::ComputeApi api,
                         const rund::kernel::ScanOp op, const bool binary) {
  const TypedMap map = binary ? BinaryMap(api) : Map(api);
  const auto built = accel::BuildDeviceVsmGraphMapScanArtifact(
      map.artifact, map.input, map.semantic, Scan(op), Geometry(op));
  auto incomplete = map.semantic;
  incomplete.recurrence_total = false;
  const auto rejected = accel::BuildDeviceVsmGraphMapScanArtifact(
      map.artifact, map.input, incomplete, Scan(op), Geometry(op));
  auto wrong_mode = map.input;
  wrong_mode.parsed.scalar_mode = lowering::DomainModeFor(
      rund::kernel::ComputeScalar::Lane64, rund::kernel::ComputeDomain::U32);
  const auto mismatched = accel::BuildDeviceVsmGraphMapScanArtifact(
      map.artifact, wrong_mode, map.semantic, Scan(op), Geometry(op));
  const std::size_t shared =
      built.artifact.source_text.find("shared uint64_t carry_low");
  const std::size_t main = built.artifact.source_text.find("void main()");
  const bool vulkan_scope = api != rund::kernel::ComputeApi::Vulkan ||
                            (shared != std::string::npos &&
                             main != std::string::npos && shared < main);
  const bool valid =
      built && !rejected && !mismatched && vulkan_scope &&
      built.proof.map.kind == accel::DeviceVsmScanMapKind::CanonicalTotalU64 &&
      built.proof.semantic.op == op && built.proof.stage_count == 2u &&
      built.proof.workgroup_width == 256u && built.plan.dispatch_count == 1u &&
      built.plan.input_buffer_count == (binary ? 2u : 1u) &&
      built.plan.input_bytes_per_tile ==
          (binary ? 2u : 1u) * sizeof(std::uint64_t) &&
      built.plan.bytes_per_tile == (binary ? 3u : 2u) * sizeof(std::uint64_t) &&
      built.artifact.source_text.find("device_vsm.typed_map") !=
          std::string::npos &&
      built.artifact.source_text.find("device_vsm.graph_map_scan") !=
          std::string::npos &&
      (binary || built.artifact.source_text.find("node[3].op=bit_xor") !=
                     std::string::npos) &&
      (binary || built.artifact.source_text.find("node[5].op=mul") !=
                     std::string::npos) &&
      built.artifact.source_text.find(
          binary ? "node[3].op=add" : "node[7].op=add") != std::string::npos;
  if (!valid) {
    std::fprintf(
        stderr,
        "DeviceVsm graph map/scan api=%u op=%u built=%u(%s) "
        "binary=%u rejected=%u mismatched=%u scope=%u inputs=%llu\n",
        static_cast<unsigned>(api), static_cast<unsigned>(op),
        static_cast<unsigned>(static_cast<bool>(built)), built.reason,
        static_cast<unsigned>(binary),
        static_cast<unsigned>(static_cast<bool>(rejected)),
        static_cast<unsigned>(static_cast<bool>(mismatched)),
        static_cast<unsigned>(vulkan_scope),
        static_cast<unsigned long long>(built.plan.input_buffer_count));
  }
  return valid;
}

[[nodiscard]] bool BuildMaximum(const rund::kernel::ComputeApi api,
                                const rund::kernel::ScanOp op) {
  constexpr std::uint64_t InputCount = 7u;
  const TypedMap map = MaximumInputMap(api);
  const auto built = accel::BuildDeviceVsmGraphMapScanArtifact(
      map.artifact, map.input, map.semantic, Scan(op), Geometry(op));
  const bool valid =
      built && built.plan.input_buffer_count == InputCount &&
      built.plan.output_buffer_count == 1u &&
      built.plan.input_bytes_per_tile == InputCount * sizeof(std::uint64_t) &&
      built.plan.output_bytes_per_tile == sizeof(std::uint64_t) &&
      built.plan.bytes_per_tile == (InputCount + 1u) * sizeof(std::uint64_t) &&
      built.artifact.source_text.find("device_vsm.typed_map") !=
          std::string::npos &&
      built.artifact.source_text.find("device_vsm.graph_map_scan") !=
          std::string::npos &&
      built.artifact.source_text.find("node[12].op=add") != std::string::npos;
  if (!valid) {
    std::fprintf(
        stderr,
        "DeviceVsm maximum graph map/scan api=%u op=%u "
        "built=%u(%s) inputs=%llu\n",
        static_cast<unsigned>(api), static_cast<unsigned>(op),
        static_cast<unsigned>(static_cast<bool>(built)), built.reason,
        static_cast<unsigned long long>(built.plan.input_buffer_count));
  }
  return valid;
}

} // namespace

bool CheckGraphMapScanSource() {
  for (const bool binary : {false, true}) {
    if (!Build(rund::kernel::ComputeApi::Metal,
               rund::kernel::ScanOp::InclusiveSum, binary) ||
        !Build(rund::kernel::ComputeApi::Metal,
               rund::kernel::ScanOp::ExclusiveSum, binary) ||
        !Build(rund::kernel::ComputeApi::Vulkan,
               rund::kernel::ScanOp::InclusiveSum, binary) ||
        !Build(rund::kernel::ComputeApi::Vulkan,
               rund::kernel::ScanOp::ExclusiveSum, binary)) {
      return false;
    }
  }
  return BuildMaximum(rund::kernel::ComputeApi::Metal,
                      rund::kernel::ScanOp::InclusiveSum) &&
         BuildMaximum(rund::kernel::ComputeApi::Metal,
                      rund::kernel::ScanOp::ExclusiveSum) &&
         BuildMaximum(rund::kernel::ComputeApi::Vulkan,
                      rund::kernel::ScanOp::InclusiveSum) &&
         BuildMaximum(rund::kernel::ComputeApi::Vulkan,
                      rund::kernel::ScanOp::ExclusiveSum);
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
