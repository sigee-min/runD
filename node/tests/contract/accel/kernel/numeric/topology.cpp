#include "src/accel/vulkan/numeric/source.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace node_accel_contract {
namespace {

[[nodiscard]] bool Contains(const std::string &source,
                            const char *const token) {
  return source.find(token) != std::string::npos;
}

[[nodiscard]] bool IsParallelNumericSource(const std::string &source,
                                           const char *const lanes,
                                           const bool global_exchange) {
  return Contains(source, lanes) &&
         Contains(source, "gl_LocalInvocationID.x") &&
         Contains(source, "gl_WorkGroupID.x") &&
         Contains(source, "barrier();") &&
         Contains(source, "memoryBarrierShared();") &&
         (!global_exchange || Contains(source, "memoryBarrierBuffer();"));
}

[[nodiscard]] bool IsGlobalTransformSource(const std::string &source) {
  return Contains(source, "#define RUND_TRANSFORM_LANES 256u") &&
         Contains(source, "layout(local_size_x = RUND_TRANSFORM_LANES) in;") &&
         Contains(source, "layout(push_constant) uniform TransformStage") &&
         Contains(source, "shared RUND_TRANSFORM_SCALAR block_r["
                          "RUND_TRANSFORM_LANES]") &&
         Contains(source, "if (span == uint64_t(1))") &&
         Contains(source, "for (uint64_t local_span = uint64_t(2);") &&
         Contains(source, "gl_GlobalInvocationID.x") &&
         Contains(source, "transform_stage.span");
}

} // namespace

[[nodiscard]] bool VulkanNumericSourcesUseParallelTopology() {
  using rund::node::accel::detail::FactorSource;
  using rund::node::accel::detail::FactorSource64;
  using rund::node::accel::detail::MatrixSource;
  using rund::node::accel::detail::MatrixSource64;
  using rund::node::accel::detail::SolveSource;
  using rund::node::accel::detail::SolveSource64;
  using rund::node::accel::detail::SpectrumSource;
  using rund::node::accel::detail::SpectrumSource64;
  using rund::node::accel::detail::TransformSource;
  using rund::node::accel::detail::TransformSource64;

  const std::array<std::string, 10u> sources{
      MatrixSource(),       MatrixSource64(),   TransformSource(),
      TransformSource64(),  FactorSource(),     FactorSource64(),
      SolveSource(),        SolveSource64(),    SpectrumSource(),
      SpectrumSource64(),
  };
  using Bytes = bool (*)(std::uint64_t &) noexcept;
  const std::array<Bytes, 10u> source_bytes{
      rund::node::accel::detail::MatrixSourceBytes,
      rund::node::accel::detail::MatrixSource64Bytes,
      rund::node::accel::detail::TransformSourceBytes,
      rund::node::accel::detail::TransformSource64Bytes,
      rund::node::accel::detail::FactorSourceBytes,
      rund::node::accel::detail::FactorSource64Bytes,
      rund::node::accel::detail::SolveSourceBytes,
      rund::node::accel::detail::SolveSource64Bytes,
      rund::node::accel::detail::SpectrumSourceBytes,
      rund::node::accel::detail::SpectrumSource64Bytes,
  };
  for (std::size_t index = 0u; index < sources.size(); ++index) {
    std::uint64_t bytes = 0u;
    if (!source_bytes[index](bytes) || bytes != sources[index].size()) {
      return false;
    }
  }

  const auto tiled_matrix = [](const std::string &source) {
    return Contains(source, "#define RUND_MATRIX_TILE_SIDE 32u") &&
           Contains(source, "#define RUND_MATRIX_TILE_LANES 128u") &&
           Contains(source,
                    "layout(local_size_x = RUND_MATRIX_TILE_LANES) in;") &&
           Contains(source, "shared RUND_MATRIX_SCALAR left_tile["
                            "RUND_MATRIX_TILE_CELLS]") &&
           Contains(source, "shared RUND_MATRIX_SCALAR right_tile["
                            "RUND_MATRIX_TILE_CELLS]") &&
           Contains(source, "gl_LocalInvocationID.x") &&
           Contains(source, "gl_WorkGroupID.x") &&
           Contains(source, "for (uint64_t base = uint64_t(0);") &&
           Contains(source, "for (uint64_t offset = uint64_t(0);") &&
           Contains(source, "RUND_MATRIX_SCALAR sum13") &&
           Contains(source, "memoryBarrierShared();") &&
           Contains(source, "barrier();");
  };
  const std::array<bool, 10u> contracts{
      tiled_matrix(sources[0]),
      tiled_matrix(sources[1]),
      IsGlobalTransformSource(sources[2]),
      IsGlobalTransformSource(sources[3]),
      IsParallelNumericSource(sources[4], "layout(local_size_x = 32) in;",
                              false),
      IsParallelNumericSource(sources[5], "layout(local_size_x = 32) in;",
                              false),
      IsParallelNumericSource(sources[6], "layout(local_size_x = 32) in;",
                              false),
      IsParallelNumericSource(sources[7], "layout(local_size_x = 32) in;",
                              false),
      IsParallelNumericSource(sources[8], "layout(local_size_x = 32) in;",
                              false),
      IsParallelNumericSource(sources[9],
                              "layout(local_size_x = 32) in;", false),
  };
  for (const bool accepted : contracts) {
    if (!accepted) {
      return false;
    }
  }
  return true;
}

} // namespace node_accel_contract
