#include "contract/program/compute/backend/lowering/local.hpp"
#include "local.hpp"

#include <kernel/program/compute/lowering/vulkan/shape.hpp>

namespace program_compute_contract {

using namespace lowering_support;

int VulkanLoweringBase() {
  const auto op = BuildFixedLane32Op(7);
  const rund::kernel::LoweringArtifact first =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Vulkan);
  const rund::kernel::LoweringArtifact second =
      rund::kernel::LowerComputeIR(op.ir(), rund::kernel::ComputeApi::Vulkan);

  TEST_ASSERT(first.ok);
  TEST_ASSERT(second.ok);
  TEST_ASSERT(first.kind == rund::kernel::LoweringArtifactKind::VulkanSource);
  TEST_ASSERT(first.key == second.key);
  TEST_ASSERT(first.source_text == second.source_text);
  TEST_ASSERT(first.canonical_ir_bytes == op.ir().canonical_bytes);
  TEST_ASSERT(second.canonical_ir_bytes == op.ir().canonical_bytes);
  TEST_ASSERT(first.metadata.ok);
  TEST_ASSERT(first.metadata.map.api == rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(first.metadata.map.scalar == rund::kernel::ComputeScalar::Lane32);
  TEST_ASSERT(first.metadata.map.input_buffer_count == 2u);
  TEST_ASSERT(first.metadata.map.input_bytes_per_tile == 8u);
  TEST_ASSERT(first.metadata.map.output_bytes_per_tile == 4u);
  TEST_ASSERT(first.metadata.map.param_bytes == 4u);
  TEST_ASSERT(first.metadata.read_count == 2u);
  TEST_ASSERT(first.metadata.write_count == 1u);
  TEST_ASSERT(first.metadata.input_element_bytes.size() == 2u);
  TEST_ASSERT(first.metadata.input_element_bytes[0] == 4u);
  TEST_ASSERT(first.metadata.input_element_bytes[1] == 4u);
  TEST_ASSERT(!first.source_text.empty());
  TEST_ASSERT(first.source_text.find("rund.compute.vulkan.source") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("#version 450") != std::string_view::npos);
  TEST_ASSERT(first.source_text.find("layout(local_size_x = 256") !=
              std::string_view::npos);
  TEST_ASSERT(
      first.source_text.find("layout(push_constant) uniform RundDispatch") !=
      std::string_view::npos);
  TEST_ASSERT(first.source_text.find("uint tile_count;") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("uint iterations;") !=
              std::string_view::npos);
  TEST_ASSERT(rund::kernel::compute_lowering_detail::kVulkanMapPushBytes ==
              2u * sizeof(std::uint32_t));
  TEST_ASSERT(first.source_text.find("gl_GlobalInvocationID.x") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("gid >= rund_dispatch.tile_count") !=
              std::string_view::npos);
  TEST_ASSERT(
      rund::kernel::compute_lowering_detail::VulkanMapGroupsForTiles(1u) == 1u);
  TEST_ASSERT(rund::kernel::compute_lowering_detail::VulkanMapGroupsForTiles(
                  256u) == 1u);
  TEST_ASSERT(rund::kernel::compute_lowering_detail::VulkanMapGroupsForTiles(
                  257u) == 2u);
  TEST_ASSERT(first.source_text.find("layout(set = 0, binding = 0, std430) "
                                     "readonly buffer RundParams") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("layout(set = 0, binding = 1, std430) "
                                     "readonly buffer read_706f73_Buffer") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("layout(set = 0, binding = 2, std430) "
                                     "readonly buffer read_76656c_Buffer") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("layout(set = 0, binding = 3, std430) "
                                     "buffer write_6f7574_Buffer") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("binding[0].kind=param") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("binding[1].kind=read") !=
              std::string_view::npos);
  TEST_ASSERT(first.source_text.find("binding[3].kind=write") !=
              std::string_view::npos);

  const auto signed_divide = rund::kernel::LowerComputeIR(
      backend_lowering_support::BuildI32DivideOp().ir(),
      rund::kernel::ComputeApi::Vulkan);
  const auto unsigned_divide = rund::kernel::LowerComputeIR(
      backend_lowering_support::BuildU64DivideOp().ir(),
      rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(signed_divide.ok);
  TEST_ASSERT(unsigned_divide.ok);
  TEST_ASSERT(signed_divide.source_text.find("RundDivSigned") !=
              std::string_view::npos);
  TEST_ASSERT(unsigned_divide.source_text.find("RundDivUnsigned") !=
              std::string_view::npos);

  const auto index = rund::kernel::LowerComputeIR(
      backend_lowering_support::BuildU64IndexOp().ir(),
      rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(index.ok);
  TEST_ASSERT(index.metadata.map.input_buffer_count == 0u);
  TEST_ASSERT(index.source_text.find("].op=index") != std::string_view::npos);
  TEST_ASSERT(index.source_text.find("uint64_t(gid)") !=
              std::string_view::npos);

  const auto uniform = rund::kernel::LowerComputeIR(
      backend_lowering_support::BuildI32UniformReadIr(),
      rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(uniform.ok);
  TEST_ASSERT(uniform.metadata.direct_read_mask == 0u);
  TEST_ASSERT(uniform.metadata.uniform_read_mask == 0x1u);
  TEST_ASSERT(uniform.source_text.find("].op=read_uniform") !=
              std::string_view::npos);
  TEST_ASSERT(uniform.source_text.find(
                  "LoadI32_read_756e69666f726d("
                  "RundBase_read_756e69666f726d)") !=
              std::string_view::npos);
  TEST_ASSERT(uniform.source_text.find(
                  "RundBase_read_756e69666f726d + gid") ==
              std::string_view::npos);

  const auto mask = rund::kernel::LowerComputeIR(
      backend_lowering_support::BuildU64MaskOp().ir(),
      rund::kernel::ComputeApi::Vulkan);
  TEST_ASSERT(mask.ok);
  TEST_ASSERT(mask.metadata.map.input_bytes_per_tile == 8u);
  TEST_ASSERT(mask.metadata.map.output_bytes_per_tile == 4u);
  TEST_ASSERT(mask.source_text.find("void StoreI32_") !=
              std::string_view::npos);
  TEST_ASSERT(mask.source_text.find("uint(node_") != std::string_view::npos);
  return 0;
}

} // namespace program_compute_contract
