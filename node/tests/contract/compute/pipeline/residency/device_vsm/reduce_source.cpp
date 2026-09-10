#include "local.hpp"

#include "src/accel/kernel/residency/device_vsm/source/reduce.hpp"

#include <kernel/program/compute/reduce/plan.hpp>

namespace rund_node_test_pipeline_residency::device_vsm_test {
namespace {

[[nodiscard]] bool Build(const rund::kernel::ComputeApi api,
                         const rund::kernel::ReduceElement element,
                         const rund::kernel::ReduceOp operation) {
  const std::uint64_t element_bytes =
      element == rund::kernel::ReduceElement::U32 ? sizeof(std::uint32_t)
                                                  : sizeof(std::uint64_t);
  constexpr std::uint64_t FrameElements = 16u;
  const rund::kernel::ReducePlan semantic = rund::kernel::PlanReduce({
      .op = operation,
      .element = element,
      .element_count = FrameElements,
      .block_size = 8u,
  });
  const accel::DeviceVsmPageGeometry geometry{
      .logical_bytes = (5u * FrameElements - 3u) * element_bytes,
      .payload_bytes = FrameElements * element_bytes,
      .frame_bytes = FrameElements * element_bytes,
      .page_count = 5u,
      .element_bytes = static_cast<std::uint32_t>(element_bytes),
  };
  const accel::DeviceVsmReduceArtifact built =
      accel::BuildDeviceVsmReduceArtifact(semantic, api, geometry);
  if (!built || built.proof.semantic.op != operation ||
      built.proof.semantic.element != element ||
      built.proof.workgroup_width != 256u || built.plan.dispatch_count != 1u ||
      built.plan.tile_count != 77u ||
      built.artifact.source_text.find("rund.compute.device_vsm.reduce.") ==
          std::string::npos) {
    return false;
  }
  const bool sum = operation == rund::kernel::ReduceOp::Sum;
  const bool count = operation == rund::kernel::ReduceOp::CountNonzero;
  const bool minimum = operation == rund::kernel::ReduceOp::Min;
  const char *const operation_marker = sum       ? "reduce.sum"
                                       : count   ? "reduce.count_nonzero"
                                       : minimum ? "reduce.min"
                                                 : "reduce.max";
  const char *const write =
      sum || count
          ? (api == rund::kernel::ComputeApi::Metal ? "output[0] = "
                                                    : "output_values[0] = ")
          : (api == rund::kernel::ComputeApi::Metal
                 ? "output[0] = partial[0]"
                 : "output_values[0] = partial[0]");
  const char *const output_counter =
      api == rund::kernel::ComputeApi::Metal
          ? "atomic_store_explicit(&result[4], 1u"
          : "atomicExchange(result[4], 1u)";
  const char *const operation_body = sum       ? "failed_page"
                                     : count   ? "input_values[index] != 0"
                                     : minimum ? "min(partial[local]"
                                               : "max(partial[local]";
  const char *const metal_count_body = "input[index] != 0";
  return built.artifact.source_text.find(operation_marker) !=
             std::string::npos &&
         built.artifact.source_text.find(write) != std::string::npos &&
         built.artifact.source_text.find(output_counter) != std::string::npos &&
         built.artifact.source_text.find(
             count && api == rund::kernel::ComputeApi::Metal
                 ? metal_count_body
                 : operation_body) != std::string::npos;
}

} // namespace

bool CheckReduceSource() {
  bool exact = true;
  for (const rund::kernel::ComputeApi api :
       {rund::kernel::ComputeApi::Metal, rund::kernel::ComputeApi::Vulkan}) {
    for (const rund::kernel::ReduceElement element :
         {rund::kernel::ReduceElement::U32, rund::kernel::ReduceElement::U64}) {
      for (const rund::kernel::ReduceOp operation :
           {rund::kernel::ReduceOp::Sum, rund::kernel::ReduceOp::CountNonzero,
            rund::kernel::ReduceOp::Min, rund::kernel::ReduceOp::Max}) {
        exact = exact && Build(api, element, operation);
      }
    }
  }
  auto semantic = rund::kernel::PlanReduce({
      .op = static_cast<rund::kernel::ReduceOp>(0xffu),
      .element = rund::kernel::ReduceElement::U32,
      .element_count = 16u,
      .block_size = 8u,
  });
  constexpr accel::DeviceVsmPageGeometry geometry{
      .logical_bytes = 77u * sizeof(std::uint32_t),
      .payload_bytes = 16u * sizeof(std::uint32_t),
      .frame_bytes = 16u * sizeof(std::uint32_t),
      .page_count = 5u,
      .element_bytes = sizeof(std::uint32_t),
  };
  return exact && !accel::BuildDeviceVsmReduceArtifact(
                      semantic, rund::kernel::ComputeApi::Metal, geometry);
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
