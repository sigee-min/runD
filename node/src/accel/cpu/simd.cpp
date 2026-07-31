#include <node/accel/cpu/simd.hpp>

#include "simd/context.hpp"
#include "simd/dispatch.hpp"

#include <cstddef>
#include <vector>
namespace rund::node::accel {

[[nodiscard]] CpuSimdRunResult
RunCpuSimd(const rund::kernel::ComputeIR &ir, const rund::kernel::CpuCaps &caps,
           const rund::kernel::LoweringArtifact &artifact,
           const rund::kernel::BindingSet &bindings) {
  const cpu_simd_detail::CpuSimdDispatch dispatch =
      cpu_simd_detail::PrepareCpuSimdDispatch(ir, caps, artifact, bindings);
  if (!dispatch.prepared.ok || dispatch.run == nullptr) {
    return cpu_simd_detail::RejectRun(caps, dispatch.prepared.reason);
  }
  static thread_local std::vector<std::max_align_t> scratch;
  const std::size_t bytes = dispatch.scratch_bytes(dispatch.prepared);
  scratch.resize((bytes + sizeof(std::max_align_t) - 1u) /
                 sizeof(std::max_align_t));
  cpu_simd_detail::CpuSimdBindingStorage binding_storage{};
  const cpu_simd_detail::CpuSimdBindingView binding_view =
      cpu_simd_detail::BindingView(bindings, binding_storage);
  return dispatch.run(
      dispatch.prepared,
      cpu_simd_detail::CpuSimdInvocation{
          .bindings = &binding_view,
          .count = bindings.tile_count,
      },
      cpu_simd_detail::CpuSimdScratch{
          scratch.data(), scratch.size() * sizeof(std::max_align_t)});
}

} // namespace rund::node::accel
