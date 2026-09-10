#include "support.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "../../../../../src/accel/kernel/step/map/stride.hpp"
#include "../../../../../src/accel/vulkan/kernel/pipeline/source.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/binding/model.hpp>

#include <array>
#include <cstdio>
#include <string>
#endif

namespace rund::node::vulkan_cache_contract {

[[nodiscard]] int MapBias() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = rund::kernel::ComputeApi::Vulkan;
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.metadata.binding_accesses = {
      rund::kernel::ComputeBindingAccess::Read,
      rund::kernel::ComputeBindingAccess::Read,
      rund::kernel::ComputeBindingAccess::Write};
  artifact.metadata.binding_names = {"input1", "input10", "out"};
  artifact.metadata.input_element_bytes = {4u, 4u};
  artifact.metadata.output_element_bytes = {4u};
  artifact.metadata.read_count = 2u;
  artifact.metadata.write_count = 1u;
  artifact.metadata.ok = true;
  artifact.metadata.reason = "ok";
  artifact.source_text = "const uint RundBase_read_696e70757431 = 0u;\n"
                         "const uint RundStride_read_696e70757431 = 4u;\n"
                         "const uint RundBase_read_696e7075743130 = 0u;\n"
                         "const uint RundStride_read_696e7075743130 = 4u;\n"
                         "const uint RundBase_write_6f7574 = 0u;\n"
                         "const uint RundStride_write_6f7574 = 4u;\n"
                         "load(RundBase_read_696e70757431 + "
                         "load(RundBase_read_696e7075743130 + gid * "
                         "RundStride_read_696e7075743130) * "
                         "RundStride_read_696e70757431);\n"
                         "store(RundBase_write_6f7574 + gid * "
                         "RundStride_write_6f7574);\n";
  artifact.ok = true;
  artifact.reason = "ok";
  artifact.source_text_upper_bytes = artifact.source_text.size();

  const rund::kernel::ComputePlan plan{.api = rund::kernel::ComputeApi::Vulkan,
                                       .input_buffer_count = 2u,
                                       .output_buffer_count = 1u};
  const std::array inputs{
      rund::kernel::ResidentBufferRef{.bytes = 64u,
                                      .offset_bytes = 4u,
                                      .element_bytes = 4u,
                                      .stride_bytes = 8u,
                                      .count = 4u,
                                      .usage =
                                          rund::kernel::kResidentUsageRead},
      rund::kernel::ResidentBufferRef{.bytes = 64u,
                                      .offset_bytes = 8u,
                                      .element_bytes = 4u,
                                      .stride_bytes = 4u,
                                      .count = 4u,
                                      .usage =
                                          rund::kernel::kResidentUsageRead},
  };
  const rund::kernel::ResidentBufferRef output{
      .bytes = 64u,
      .offset_bytes = 12u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 4u,
      .usage = rund::kernel::kResidentUsageWrite};
  rund::kernel::BindingSet bindings{};
  bindings.resident_inputs =
      rund::kernel::ResidentBindingRange{.refs = inputs.data(),
                                         .storage_count = inputs.size(),
                                         .count = inputs.size()};
  bindings.resident_outputs = rund::kernel::ResidentBindingRange{
      .refs = &output, .storage_count = 1u, .count = 1u};

  const rund::kernel::LoweringArtifact specialized =
      SpecializeMap(artifact, plan, bindings, 16u);
  if (!specialized.ok ||
      specialized.source_text.find(
          "const uint RundStride_read_696e70757431 = 8u;") ==
          std::string::npos ||
      specialized.source_text.find(
          "const uint RundBase_read_696e70757431 = 4u;") == std::string::npos ||
      specialized.source_text.find(
          "const uint RundBase_read_696e7075743130 = 8u;") ==
          std::string::npos ||
      specialized.source_text.find(
          "const uint RundBase_read_696e7075743130 = 4u;") !=
          std::string::npos ||
      specialized.source_text.find("const uint RundBase_write_6f7574 = 12u;") ==
          std::string::npos) {
    std::fprintf(stderr, "map bias failed reason=%s source=%s\n",
                 specialized.reason, specialized.source_text.c_str());
    return 35;
  }
  if (SpecializeMap(artifact, plan, bindings, 0u).ok) {
    return 36;
  }
  return 0;
#else
  return 0;
#endif
}

} // namespace rund::node::vulkan_cache_contract
