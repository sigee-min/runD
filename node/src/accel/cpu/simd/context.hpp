#pragma once

#include <node/accel/cpu/simd.hpp>

#include <kernel/program/compute/lowering/model.hpp>
#include <kernel/program/compute/retention.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rund::node::accel::cpu_simd_detail {

using CpuSimdExecutorSlot = std::uint8_t;

inline constexpr std::size_t kCpuSimdBaseExecutorCount =
    static_cast<std::size_t>(rund::kernel::IrOp::ReadUniform) + 1u;
inline constexpr CpuSimdExecutorSlot kCpuSimdReadFullExecutorSlot =
    static_cast<CpuSimdExecutorSlot>(kCpuSimdBaseExecutorCount);
inline constexpr CpuSimdExecutorSlot kCpuSimdReadStridedFullExecutorSlot =
    static_cast<CpuSimdExecutorSlot>(kCpuSimdBaseExecutorCount + 1u);
inline constexpr CpuSimdExecutorSlot kCpuSimdWriteFullExecutorSlot =
    static_cast<CpuSimdExecutorSlot>(kCpuSimdBaseExecutorCount + 2u);
inline constexpr std::size_t kCpuSimdExecutorCount =
    kCpuSimdBaseExecutorCount + 3u;
inline constexpr CpuSimdExecutorSlot kCpuSimdInvalidExecutorSlot = 0xffu;

static_assert(kCpuSimdExecutorCount < kCpuSimdInvalidExecutorSlot);

[[nodiscard]] constexpr CpuSimdExecutorSlot
CpuSimdBaseExecutorSlot(const std::uint8_t op) noexcept {
  return static_cast<std::size_t>(op) < kCpuSimdBaseExecutorCount
             ? static_cast<CpuSimdExecutorSlot>(op)
             : kCpuSimdInvalidExecutorSlot;
}

[[nodiscard]] constexpr bool
CpuSimdExecutorSlotValid(const CpuSimdExecutorSlot slot) noexcept {
  return static_cast<std::size_t>(slot) < kCpuSimdExecutorCount;
}

struct PreparedInstruction final {
  rund::kernel::compute_lowering_detail::ParsedNode node{};
  rund::kernel::u64 binding_slot = 0u;
  rund::kernel::u64 immediate = 0u;
  rund::kernel::u64 element_bytes = 0u;
  rund::kernel::u32 value_index = 0u;
  CpuSimdExecutorSlot full_executor_slot = kCpuSimdInvalidExecutorSlot;
  CpuSimdExecutorSlot tail_executor_slot = kCpuSimdInvalidExecutorSlot;
};

static_assert(sizeof(PreparedInstruction) == 56u,
              "prepared selector slots must preserve the compact ABI size");

struct CpuSimdReadBinding final {
  const unsigned char *data = nullptr;
  std::size_t stride = 0u;
};

struct CpuSimdWriteBinding final {
  unsigned char *data = nullptr;
  std::size_t stride = 0u;
};

inline constexpr std::size_t kCpuSimdBindingCapacity =
    rund::kernel::kMaxComputeBindingCount;

struct CpuSimdBindingStorage final {
  std::array<CpuSimdReadBinding, kCpuSimdBindingCapacity> reads{};
  std::array<CpuSimdWriteBinding, kCpuSimdBindingCapacity> writes{};
};

struct CpuSimdBindingView final {
  const CpuSimdReadBinding *reads = nullptr;
  rund::kernel::u64 read_count = 0u;
  const CpuSimdWriteBinding *writes = nullptr;
  rund::kernel::u64 write_count = 0u;
  rund::kernel::u64 logical_offset = 0u;
  rund::kernel::u64 tile_count = 0u;
};

struct CpuSimdInvocation final {
  const CpuSimdBindingView *bindings = nullptr;
  rund::kernel::u64 begin = 0u;
  rund::kernel::u64 count = 0u;
};

struct PreparedRun {
  std::vector<PreparedInstruction> instructions;
  std::vector<rund::kernel::ComputeFixedFormat> value_formats;
  std::size_t once_count = 0u;
  rund::kernel::u32 read_count = 0u;
  rund::kernel::u32 write_count = 0u;
  rund::kernel::ComputeDomain domain = rund::kernel::ComputeDomain::Fixed;
  rund::kernel::CpuSimdStrategy strategy =
      rund::kernel::CpuSimdStrategy::Scalar;
  bool uses_index = false;
  bool ok = false;
  const char *reason = "cpu_simd_run_invalid";

  // Counts allocations owned below this object. The dispatch and PreparedRun
  // inline storage are counted by the enclosing CPU program.
  [[nodiscard]] rund::kernel::u64
  retained_dynamic_memory_bytes() const noexcept {
    using rund::kernel::compute_retained_detail::Add;
    using rund::kernel::compute_retained_detail::VectorCapacityBytes;
    return Add(VectorCapacityBytes(instructions),
               VectorCapacityBytes(value_formats));
  }
};

[[nodiscard]] inline CpuSimdBindingView
BindingView(const rund::kernel::BindingSet &bindings,
            CpuSimdBindingStorage &storage) noexcept {
  if (bindings.input_buffer_count > storage.reads.size() ||
      bindings.output_buffer_count > storage.writes.size()) {
    return {};
  }
  for (std::size_t index = 0u; index < bindings.input_buffer_count; ++index) {
    const rund::kernel::BufferSpan &span = bindings.input_buffers[index];
    storage.reads[index] = CpuSimdReadBinding{
        .data = static_cast<const unsigned char *>(span.data),
        .stride = static_cast<std::size_t>(span.stride_bytes),
    };
  }
  const bool staged = bindings.output_buffer_count == 0u;
  const std::size_t write_count = staged ? 1u : bindings.output_buffer_count;
  if (write_count > storage.writes.size()) {
    return {};
  }
  for (std::size_t index = 0u; index < write_count; ++index) {
    storage.writes[index] = staged
                                ? CpuSimdWriteBinding{
                                      .data = static_cast<unsigned char *>(
                                          bindings.staged_output),
                                      .stride = static_cast<std::size_t>(
                                          bindings.staged_output_stride),
                                  }
                                : CpuSimdWriteBinding{
                                      .data = static_cast<unsigned char *>(
                                          bindings.output_buffers[index].data),
                                      .stride = static_cast<std::size_t>(
                                          bindings.output_buffers[index]
                                              .stride_bytes),
                                  };
  }
  return CpuSimdBindingView{
      .reads = storage.reads.data(),
      .read_count = bindings.input_buffer_count,
      .writes = storage.writes.data(),
      .write_count = write_count,
      .logical_offset = bindings.logical_offset,
      .tile_count = bindings.tile_count,
  };
}

struct CpuSimdScratch {
  void *data = nullptr;
  std::size_t bytes = 0u;
};

[[nodiscard]] inline CpuSimdRunResult
RejectRun(const rund::kernel::CpuCaps &caps,
          const char *const reason) noexcept {
  return CpuSimdRunResult{
      .reason = reason == nullptr ? "cpu_simd_run_invalid" : reason,
      .strategy = caps.strategy,
      .rejected_count = 1u,
  };
}

[[nodiscard]] inline CpuSimdRunResult
RejectRun(const PreparedRun &prepared, const char *const reason) noexcept {
  return CpuSimdRunResult{
      .reason = reason == nullptr ? "cpu_simd_run_invalid" : reason,
      .strategy = prepared.strategy,
      .rejected_count = 1u,
  };
}

[[nodiscard]] inline CpuSimdRunResult
AcceptRun(const PreparedRun &prepared, const rund::kernel::u64 processed_tiles,
          const rund::kernel::u64 vector_chunks,
          const rund::kernel::u64 tail_chunks) noexcept {
  return CpuSimdRunResult{
      .ok = true,
      .reason = "ok",
      .strategy = prepared.strategy,
      .processed_tiles = processed_tiles,
      .vector_chunk_count = vector_chunks,
      .tail_chunk_count = tail_chunks,
  };
}

[[nodiscard]] inline bool CheckedOffset(const rund::kernel::u64 index,
                                        const rund::kernel::u64 stride,
                                        const rund::kernel::u64 bytes,
                                        rund::kernel::u64 &offset) noexcept {
  constexpr rund::kernel::u64 kMax = ~rund::kernel::u64{0u};
  if (stride != 0u && index > kMax / stride) {
    return false;
  }
  offset = index * stride;
  return offset <= kMax - bytes;
}

[[nodiscard]] PreparedRun
PrepareRun(const rund::kernel::ComputeIR &ir, const rund::kernel::CpuCaps &caps,
           const rund::kernel::compute_lowering_detail::ParsedIR &parsed,
           const rund::kernel::BindingSet &bindings);

[[nodiscard]] CpuSimdRunResult
RunFixedLane32(const PreparedRun &prepared, const CpuSimdInvocation &invocation,
               CpuSimdScratch scratch);

[[nodiscard]] std::size_t
ScratchBytesFixedLane32(const PreparedRun &prepared) noexcept;

[[nodiscard]] CpuSimdRunResult
RunFixedLane64(const PreparedRun &prepared, const CpuSimdInvocation &invocation,
               CpuSimdScratch scratch);

[[nodiscard]] std::size_t
ScratchBytesFixedLane64(const PreparedRun &prepared) noexcept;

} // namespace rund::node::accel::cpu_simd_detail
