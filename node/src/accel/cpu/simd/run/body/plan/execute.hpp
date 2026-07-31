#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

inline void ExecuteOnce(const PreparedRun &prepared,
                        const CpuSimdBindingView &bindings,
                        Values &values) noexcept {
  for (std::size_t index = 0u; index < prepared.once_count; ++index) {
    const Instruction &instruction = prepared.instructions[index];
    values.invalidate(instruction.value_index);
    const ExecuteFn executor = ExecutorFor(instruction.full_executor_slot);
    executor(instruction, prepared, bindings, 0u, kLaneCount, values);
  }
}

struct LoopCount final {
  u64 tiles = 0u;
  u64 vectors = 0u;
  u64 tails = 0u;
  const char *reason = "ok";
};

[[nodiscard]] LoopCount ExecuteLoop(const PreparedRun &prepared,
                                    const CpuSimdBindingView &bindings,
                                    const u64 begin, const u64 tile_count,
                                    Values &values) noexcept {
  LoopCount count{};
  const u64 full = tile_count - tile_count % kLaneCount;
  for (u64 offset = 0u; offset < full; offset += kLaneCount) {
    const u64 base = begin + offset;
    for (std::size_t index = prepared.once_count;
         index < prepared.instructions.size(); ++index) {
      const Instruction &instruction = prepared.instructions[index];
      values.invalidate(instruction.value_index);
      const ExecuteFn executor = ExecutorFor(instruction.full_executor_slot);
      executor(instruction, prepared, bindings, base, kLaneCount, values);
      if (!values) {
        count.reason = values.reason();
        return count;
      }
    }
    count.tiles += kLaneCount;
    ++count.vectors;
  }
  const auto tail = static_cast<std::size_t>(tile_count - full);
  if (tail == 0u) {
    return count;
  }
  for (std::size_t index = prepared.once_count;
       index < prepared.instructions.size(); ++index) {
    const Instruction &instruction = prepared.instructions[index];
    values.invalidate(instruction.value_index);
    const ExecuteFn executor = ExecutorFor(instruction.tail_executor_slot);
    executor(instruction, prepared, bindings, begin + full, tail, values);
    if (!values) {
      count.reason = values.reason();
      return count;
    }
  }
  count.tiles += tail;
  ++count.tails;
  return count;
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
