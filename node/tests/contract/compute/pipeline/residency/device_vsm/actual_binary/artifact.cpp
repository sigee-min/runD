#include "internal.hpp"

namespace rund_node_test_pipeline_residency::device_vsm_test::binary_detail {
namespace {

struct BinaryOwner final {
  rund::kernel::LoweringArtifact artifact{};
  std::array<rund::kernel::ComputeDispatchWindow, 1u> windows{};
};

} // namespace

[[nodiscard]] rund::kernel::ComputeApi
ApiFor(const rund::compute::Backend backend) noexcept {
  return backend == rund::compute::Backend::Metal
             ? rund::kernel::ComputeApi::Metal
             : rund::kernel::ComputeApi::Vulkan;
}

[[nodiscard]] rund::kernel::LoweringArtifact
BinaryArtifact(const rund::kernel::ComputeApi api) {
  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = api;
  artifact.key.variant = rund::kernel::LoweringArtifactVariant::Canonical;
  artifact.key.op_hash_hi = 0x42494e4152595653ull;
  artifact.key.op_hash_lo = 0x4144445f55333200ull;
  artifact.kind = api == rund::kernel::ComputeApi::Metal
                      ? rund::kernel::LoweringArtifactKind::MetalSource
                      : rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.metadata.ok = true;
  artifact.metadata.read_count = 2u;
  artifact.metadata.write_count = 1u;
  if (api == rund::kernel::ComputeApi::Metal) {
    artifact.source_text =
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "// artifact_variant=canonical\n"
        "kernel void rund_compute_map_42494e4152595653_4144445f55333200(\n"
        "    constant uchar* rund_params [[buffer(0)]],\n"
        "    const device uint* read_first [[buffer(1)]],\n"
        "    const device uint* read_second [[buffer(2)]],\n"
        "    device uint* write_output [[buffer(3)]],\n"
        "    uint gid [[thread_position_in_grid]]) {\n"
        "  write_output[gid] = read_first[gid] + read_second[gid];\n"
        "}\n";
  } else {
    artifact.source_text =
        "#version 450\n"
        "layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;\n"
        "// artifact_variant=canonical\n"
        "layout(set = 0, binding = 0, std430) readonly buffer Params { uint "
        "rund_params[]; };\n"
        "layout(set = 0, binding = 1, std430) readonly buffer First { uint "
        "read_first[]; };\n"
        "layout(set = 0, binding = 2, std430) readonly buffer Second { uint "
        "read_second[]; };\n"
        "layout(set = 0, binding = 3, std430) writeonly buffer Output { uint "
        "write_output[]; };\n"
        "layout(push_constant) uniform RundDispatch {\n"
        "  uint tile_count;\n"
        "  uint iterations;\n"
        "} rund_dispatch;\n"
        "void main() {\n"
        "  const uint gid = gl_GlobalInvocationID.x;\n"
        "  if (gid >= rund_dispatch.tile_count) { return; }\n"
        "  write_output[gid] = read_first[gid] + read_second[gid];\n"
        "}\n";
  }
  artifact.source_text_upper_bytes = artifact.source_text.size();
  artifact.ok = true;
  artifact.reason = "ok";
  return artifact;
}

[[nodiscard]] accel::DeviceVsmResidentBinding
Binding(const accel::UploadRoute &route,
        const accel::DeviceVsmResidentRole role) noexcept {
  return accel::DeviceVsmResidentBinding{
      .role = role,
      .backing = route.resident,
      .handle = route.handle,
  };
}

[[nodiscard]] std::shared_ptr<accel::DeviceVsmProof>
BuildProof(const rund::kernel::ComputeApi api, const std::uint64_t pages,
           const std::array<accel::UploadRoute, 3u> &routes) {
  constexpr std::uint64_t PayloadElements = 16u;
  constexpr std::uint64_t ElementBytes = sizeof(std::uint32_t);
  const std::uint64_t logical_elements = pages * PayloadElements - 3u;
  auto owner = std::make_shared<BinaryOwner>();
  owner->artifact = BinaryArtifact(api);
  if (!accel::TransformDeviceVsmSource(owner->artifact, 2u, 1u)) {
    return {};
  }
  owner->windows[0u] = rund::kernel::ComputeDispatchWindow{
      .begin_sequence = 0u,
      .tile_count = PayloadElements,
  };
  auto proof = std::make_shared<accel::DeviceVsmProof>();
  proof->identity = accel::DeviceVsmIdentity{
      .hi = api == rund::kernel::ComputeApi::Metal ? 0x4d4554414c423256ull
                                                   : 0x56554c4b414e4232ull,
      .lo = pages,
  };
  proof->semantic_owner = owner;
  proof->artifact = &owner->artifact;
  proof->plan = rund::kernel::ComputePlan{
      .tile_count = PayloadElements,
      .op_hash_hi = owner->artifact.key.op_hash_hi,
      .op_hash_lo = owner->artifact.key.op_hash_lo,
      .api = api,
      .input_buffer_count = 2u,
      .output_buffer_count = 1u,
      .input_bytes_per_tile = 2u * ElementBytes,
      .output_bytes_per_tile = ElementBytes,
      .bytes_per_tile = 3u * ElementBytes,
      .dispatch_window_tiles = PayloadElements,
      .dispatch_count = 1u,
      .fixed_authoritative = true,
      .ok = true,
      .reason = "ok",
  };
  proof->residents.rows[0u] =
      Binding(routes[0u], accel::DeviceVsmResidentRole::Input);
  proof->residents.rows[1u] =
      Binding(routes[1u], accel::DeviceVsmResidentRole::Input);
  proof->residents.rows[2u] =
      Binding(routes[2u], accel::DeviceVsmResidentRole::Output);
  proof->residents.count = 3u;
  proof->residents.input_count = 2u;
  proof->residents.output_count = 1u;
  proof->windows = owner->windows.data();
  proof->geometry = accel::DeviceVsmPageGeometry{
      .logical_bytes = logical_elements * ElementBytes,
      .payload_bytes = PayloadElements * ElementBytes,
      .frame_bytes = PayloadElements * ElementBytes,
      .page_count = pages,
      .element_bytes = ElementBytes,
  };
  proof->output_bytes = proof->geometry.logical_bytes;
  proof->window_count = owner->windows.size();
  proof->width = 2u;
  proof->fixed_common_storage = true;
  return proof;
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test::binary_detail
