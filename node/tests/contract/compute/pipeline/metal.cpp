#include "local.hpp"

#include "src/accel/kernel/preparation.hpp"
#include "src/accel/metal/pipeline/guard.hpp"

#include <string>
#include <string_view>

namespace rund_node_test_pipeline {
namespace {

[[nodiscard]] std::size_t Occurrences(const std::string_view source,
                                      const std::string_view value) noexcept {
  std::size_t count = 0u;
  std::size_t offset = 0u;
  while ((offset = source.find(value, offset)) != std::string_view::npos) {
    ++count;
    offset += value.size();
  }
  return count;
}

} // namespace

[[nodiscard]] int CheckMetalGuardTransform() {
  using namespace rund::node::accel::detail;
  const std::string source = R"MSL(#include <metal_stdlib>
using namespace metal;
#define RUND_KERNEL(name) name
kernel void rund_empty() {}
kernel void RUND_KERNEL(rund_macro)(
    uint gid [[thread_position_in_grid]]) { (void)gid; }
)MSL";
  {
    const KernelPreparationScope standalone{KernelPreparationMode::Standalone};
    if (PipelinePrivateMetalSource(source) != source) {
      return 1;
    }
  }
  const KernelPreparationScope pipeline{KernelPreparationMode::PipelinePrivate};
  const std::string guarded = PipelinePrivateMetalSource(source);
  constexpr std::string_view argument =
      "device const uint *rund_pipeline_guard [[buffer(30)]]";
  constexpr std::string_view check =
      "if (rund_pipeline_guard[0] != 0u) { return; }";
  if (guarded.empty() ||
      guarded.find("rund_empty(device const uint") == std::string::npos ||
      guarded.find("uint gid [[thread_position_in_grid]], device const uint") ==
          std::string::npos ||
      Occurrences(guarded, argument) != 2u ||
      Occurrences(guarded, check) != 2u) {
    return 2;
  }
  if (!PipelinePrivateMetalSource(
           "kernel void occupied(device uint *value [[buffer(30)]]) {}")
           .empty()) {
    return 3;
  }
  if (!PipelinePrivateMetalSource("kernel void malformed(").empty()) {
    return 4;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
