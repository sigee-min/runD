#pragma once

namespace rund::node::accel::cpu_simd_detail {

[[nodiscard]] CpuSimdRunResult
RUND_CPU_SIMD_RUN(const PreparedRun &prepared,
                  const CpuSimdInvocation &invocation,
                  const CpuSimdScratch scratch) {
  const RunScratch memory = PrepareRunScratch(prepared, scratch);
  if (!memory) {
    return RejectRun(prepared, "cpu_simd_scratch_invalid");
  }
  if (invocation.bindings == nullptr || invocation.count == 0u ||
      invocation.begin > ~u64{0u} - invocation.count) {
    return RejectRun(prepared, "cpu_simd_tile_count_invalid");
  }
  const CpuSimdBindingView &view = *invocation.bindings;
  const u64 end = invocation.begin + invocation.count;
  if (end > view.tile_count || view.read_count != prepared.read_count ||
      view.write_count != prepared.write_count ||
      (prepared.read_count != 0u && view.reads == nullptr) ||
      view.writes == nullptr ||
      (prepared.uses_index && view.logical_offset > ~u64{0u} - (end - 1u))) {
    return RejectRun(prepared, "cpu_simd_tile_count_invalid");
  }

  Values values(memory.values, memory.wide, memory.wide_valid);
  ExecuteOnce(prepared, view, values);
  if (!values) {
    return RejectRun(prepared, values.reason());
  }
  const LoopCount count =
      ExecuteLoop(prepared, view, invocation.begin, invocation.count, values);
  if (!values) {
    return RejectRun(prepared, values.reason());
  }
  return AcceptRun(prepared, count.tiles, count.vectors, count.tails);
}

[[nodiscard]] std::size_t
RUND_CPU_SIMD_SCRATCH_BYTES(const PreparedRun &prepared) noexcept {
  return RequiredScratchBytes(prepared);
}

} // namespace rund::node::accel::cpu_simd_detail
