#include "local.hpp"

#include "src/accel/kernel/residency/device_vsm/source/scan.hpp"

#include <kernel/program/compute/scan/plan.hpp>

namespace rund_node_test_pipeline_residency::device_vsm_test {
namespace {

[[nodiscard]] bool Build(const rund::kernel::ComputeApi api,
                         const rund::kernel::ScanOp operation,
                         const rund::kernel::ScanElement element) {
  const std::uint64_t element_bytes = element == rund::kernel::ScanElement::U32
                                          ? sizeof(std::uint32_t)
                                          : sizeof(std::uint64_t);
  constexpr std::uint64_t FrameElements = 16u;
  const bool exclusive = operation == rund::kernel::ScanOp::ExclusiveSum;
  const std::uint64_t payload =
      (FrameElements - (exclusive ? 1u : 0u)) * element_bytes;
  const rund::kernel::ScanPlan semantic = rund::kernel::PlanScan({
      .op = operation,
      .element = element,
      .element_count = FrameElements,
      .block_size = 8u,
  });
  const accel::DeviceVsmPageGeometry geometry{
      .logical_bytes = 5u * payload - 3u * element_bytes,
      .payload_bytes = payload,
      .frame_bytes = FrameElements * element_bytes,
      .read_prefix_bytes = exclusive ? element_bytes : 0u,
      .target_offset_bytes = exclusive ? element_bytes : 0u,
      .page_count = 5u,
      .element_bytes = static_cast<std::uint32_t>(element_bytes),
  };
  const accel::DeviceVsmScanArtifact built =
      accel::BuildDeviceVsmScanArtifact(semantic, api, geometry);
  if (!built || built.proof.semantic.op != operation ||
      built.proof.workgroup_width != 256u || built.plan.dispatch_count != 1u ||
      built.plan.tile_count != geometry.logical_bytes / element_bytes ||
      built.artifact.source_text.find("rund_result[7]") == std::string::npos ||
      built.artifact.source_text.find("compute.device_vsm.scan") ==
          std::string::npos) {
    return false;
  }
  const char *const write = element == rund::kernel::ScanElement::U32
                                ? "rund_output[index] = uint(value)"
                                : "rund_output[index] = value";
  return built.artifact.source_text.find(write) != std::string::npos;
}

} // namespace

bool CheckScanSource() {
  constexpr accel::DeviceVsmPageGeometry geometry{
      .logical_bytes = 5u * 16u * sizeof(std::uint32_t),
      .payload_bytes = 16u * sizeof(std::uint32_t),
      .frame_bytes = 16u * sizeof(std::uint32_t),
      .page_count = 5u,
      .element_bytes = sizeof(std::uint32_t),
  };
  const rund::kernel::ScanPlan invalid = rund::kernel::PlanScan({
      .op = rund::kernel::ScanOp::InclusiveSum,
      .element = rund::kernel::ScanElement::U32,
      .element_count = 16u,
      .block_size = 8u,
  });
  const bool u64 =
      Build(rund::kernel::ComputeApi::Metal, rund::kernel::ScanOp::InclusiveSum,
            rund::kernel::ScanElement::U64) &&
      Build(rund::kernel::ComputeApi::Metal, rund::kernel::ScanOp::ExclusiveSum,
            rund::kernel::ScanElement::U64) &&
      Build(rund::kernel::ComputeApi::Vulkan,
            rund::kernel::ScanOp::InclusiveSum,
            rund::kernel::ScanElement::U64) &&
      Build(rund::kernel::ComputeApi::Vulkan,
            rund::kernel::ScanOp::ExclusiveSum, rund::kernel::ScanElement::U64);
  const bool u32 =
      Build(rund::kernel::ComputeApi::Metal, rund::kernel::ScanOp::InclusiveSum,
            rund::kernel::ScanElement::U32) &&
      Build(rund::kernel::ComputeApi::Metal, rund::kernel::ScanOp::ExclusiveSum,
            rund::kernel::ScanElement::U32) &&
      Build(rund::kernel::ComputeApi::Vulkan,
            rund::kernel::ScanOp::InclusiveSum,
            rund::kernel::ScanElement::U32) &&
      Build(rund::kernel::ComputeApi::Vulkan,
            rund::kernel::ScanOp::ExclusiveSum, rund::kernel::ScanElement::U32);
  auto malformed = geometry;
  malformed.element_bytes = 0u;
  return u64 && u32 &&
         !accel::BuildDeviceVsmScanArtifact(
             invalid, rund::kernel::ComputeApi::Metal, malformed);
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
