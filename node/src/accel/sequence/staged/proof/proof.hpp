#pragma once

#include <kernel/program/compute/backend.hpp>

namespace rund::node::accel::detail {

class StagedProof {
 public:
  constexpr StagedProof() noexcept = default;

  [[nodiscard]] constexpr bool ok() const noexcept { return ok_; }

  [[nodiscard]] constexpr bool bulk() const noexcept {
    return ok_ && bulk_;
  }

  [[nodiscard]] constexpr bool matches(
      const rund::kernel::BindingSet& bindings,
      const rund::kernel::ComputeDispatchWindow& window) const noexcept {
    return ok_ && window_.begin_sequence == window.begin_sequence &&
           window_.tile_count == window.tile_count &&
           tile_count_ == bindings.tile_count &&
           output_bytes_per_tile_ == bindings.output_bytes_per_tile &&
           input_buffers_ == bindings.input_buffers &&
           input_buffer_count_ == bindings.input_buffer_count &&
           staged_output_ == bindings.staged_output &&
           staged_output_stride_ == bindings.staged_output_stride &&
           staged_output_count_ == bindings.staged_output_count &&
           sequence_tiles_ == bindings.sequence_tiles &&
           sequence_tile_count_ == bindings.sequence_tile_count;
  }

 private:
  friend StagedProof PlanStagedWindow(
      const rund::kernel::BindingSet& bindings,
      const rund::kernel::ComputeDispatchWindow& window) noexcept;

  constexpr StagedProof(const bool ok,
                        const bool bulk,
                        const rund::kernel::BindingSet& bindings,
                        const rund::kernel::ComputeDispatchWindow window)
      : ok_(ok),
        bulk_(bulk),
        window_(window),
        tile_count_(bindings.tile_count),
        output_bytes_per_tile_(bindings.output_bytes_per_tile),
        input_buffers_(bindings.input_buffers),
        input_buffer_count_(bindings.input_buffer_count),
        staged_output_(bindings.staged_output),
        staged_output_stride_(bindings.staged_output_stride),
        staged_output_count_(bindings.staged_output_count),
        sequence_tiles_(bindings.sequence_tiles),
        sequence_tile_count_(bindings.sequence_tile_count) {}

  bool ok_ = false;
  bool bulk_ = false;
  rund::kernel::ComputeDispatchWindow window_{};
  rund::kernel::u64 tile_count_ = 0u;
  rund::kernel::u64 output_bytes_per_tile_ = 0u;
  const rund::kernel::BufferSpan* input_buffers_ = nullptr;
  rund::kernel::u64 input_buffer_count_ = 0u;
  void* staged_output_ = nullptr;
  rund::kernel::u64 staged_output_stride_ = 0u;
  rund::kernel::u64 staged_output_count_ = 0u;
  const rund::kernel::u64* sequence_tiles_ = nullptr;
  rund::kernel::u64 sequence_tile_count_ = 0u;
};

}  // namespace rund::node::accel::detail
