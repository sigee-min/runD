#pragma once

#include <kernel/program/compute/model.hpp>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

namespace rund::kernel {

struct BufferSpan {
  const void *data = nullptr;
  u64 element_bytes = 0u;
  u64 stride_bytes = 0u;
  u64 count = 0u;

  template <typename T>
  [[nodiscard]] static constexpr BufferSpan
  contiguous(const T *const values, const std::size_t count) noexcept {
    static_assert(std::is_object_v<T>,
                  "compute input buffers require object element storage");
    return BufferSpan{
        .data = values,
        .element_bytes = static_cast<u64>(sizeof(T)),
        .stride_bytes = static_cast<u64>(sizeof(T)),
        .count = static_cast<u64>(count),
    };
  }
};

struct OutputSpan {
  void *data = nullptr;
  u64 element_bytes = 0u;
  u64 stride_bytes = 0u;
  u64 count = 0u;

  template <typename T>
  [[nodiscard]] static constexpr OutputSpan
  contiguous(T *const values, const std::size_t count) noexcept {
    static_assert(std::is_object_v<T>,
                  "compute output buffers require object element storage");
    return OutputSpan{
        .data = values,
        .element_bytes = static_cast<u64>(sizeof(T)),
        .stride_bytes = static_cast<u64>(sizeof(T)),
        .count = static_cast<u64>(count),
    };
  }
};

inline constexpr u32 kResidentUsageRead = 1u;
inline constexpr u32 kResidentUsageWrite = 2u;

struct ResidentBufferRef {
  u64 id = 0u;
  u64 bytes = 0u;
  u64 offset_bytes = 0u;
  u64 element_bytes = 0u;
  u64 stride_bytes = 0u;
  u64 count = 0u;
  u32 usage = 0u;
};

// A view over resident bindings. The refs and handles have one owner; indices
// select a deterministic per-step order without copying shared_ptr handles or
// duplicating resident descriptors. Validation consumes the checked accessors
// below rather than trusting caller-provided counts or indices.
struct ResidentBindingRange {
  const ResidentBufferRef *refs = nullptr;
  const std::shared_ptr<void> *handles = nullptr;
  const u64 *indices = nullptr;
  u64 storage_count = 0u;
  u64 count = 0u;

  [[nodiscard]] constexpr bool empty() const noexcept {
    return refs == nullptr && handles == nullptr && indices == nullptr &&
           storage_count == 0u && count == 0u;
  }

  [[nodiscard]] constexpr const ResidentBufferRef *
  ref(const u64 position) const noexcept {
    if (refs == nullptr || position >= count) {
      return nullptr;
    }
    const u64 index = indices == nullptr ? position : indices[position];
    return index < storage_count ? &refs[index] : nullptr;
  }

  [[nodiscard]] const std::shared_ptr<void> *
  handle(const u64 position) const noexcept {
    if (handles == nullptr || position >= count) {
      return nullptr;
    }
    const u64 index = indices == nullptr ? position : indices[position];
    return index < storage_count ? &handles[index] : nullptr;
  }

  [[nodiscard]] constexpr bool has_refs() const noexcept {
    if (count == 0u) {
      return empty();
    }
    if (refs == nullptr || storage_count == 0u) {
      return false;
    }
    for (u64 position = 0u; position < count; ++position) {
      if (ref(position) == nullptr) {
        return false;
      }
    }
    return true;
  }

  [[nodiscard]] bool has_handles() const noexcept {
    if (count == 0u) {
      return empty();
    }
    if (handles == nullptr || storage_count == 0u) {
      return false;
    }
    for (u64 position = 0u; position < count; ++position) {
      const std::shared_ptr<void> *const owner = handle(position);
      if (owner == nullptr || *owner == nullptr) {
        return false;
      }
    }
    return true;
  }
};

struct InputShapeBytes {
  const u64 *data = nullptr;
  u64 count = 0u;
};

struct BindingObligations {
  u64 tile_count = 0u;
  u64 input_buffer_count = 0u;
  u64 output_buffer_count = 1u;
  u64 input_bytes_per_tile = 0u;
  u64 output_bytes_per_tile = 0u;
  u64 param_bytes = 0u;
  bool validate_staged_output = true;
  const u64 *input_element_bytes = nullptr;
  u64 input_element_byte_count = 0u;
  const u64 *input_counts = nullptr;
  u64 input_count_count = 0u;
  const u64 *output_element_bytes = nullptr;
  u64 output_element_byte_count = 0u;
  u64 uniform_input_element_bytes = 0u;
  u64 uniform_output_element_bytes = 0u;
  // Only the Map resident-view path may opt into byte strides wider than the
  // element. Collective primitives retain their exact-contiguous contract.
  bool allow_resident_stride = false;
};

struct BindingValidation {
  bool ok = false;
  const char *reason = "compute_binding_invalid";

  [[nodiscard]] constexpr explicit operator bool() const noexcept { return ok; }
};

struct BindingSet {
  u64 phase_id = 0u;
  u64 tile_count = 0u;
  u64 lane_count = 0u;
  u64 logical_offset = 0u;
  u64 op_hash_hi = 0u;
  u64 op_hash_lo = 0u;
  ComputeApi api = ComputeApi::Metal;
  ComputeScalar scalar = ComputeScalar::Lane32;
  ComputeDomain domain = ComputeDomain::Fixed;
  u64 input_bytes_per_tile = 0u;
  u64 output_bytes_per_tile = 0u;
  u64 param_bytes = 0u;
  u64 metadata_bytes_per_tile = 0u;
  const BufferSpan *input_buffers = nullptr;
  u64 input_buffer_count = 0u;
  const OutputSpan *output_buffers = nullptr;
  u64 output_buffer_count = 0u;
  const u64 *input_element_bytes = nullptr;
  u64 input_element_byte_count = 0u;
  const u64 *output_element_bytes = nullptr;
  u64 output_element_byte_count = 0u;
  const void *param_data = nullptr;
  u64 param_data_bytes = 0u;
  void *staged_output = nullptr;
  u64 staged_output_stride = 0u;
  u64 staged_output_count = 0u;
  const u64 *sequence_tiles = nullptr;
  u64 sequence_tile_count = 0u;
  bool ok = false;
  const char *reason = "compute_binding_invalid";
  ResidentBindingRange resident_inputs{};
  ResidentBindingRange resident_outputs{};

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return ok && (has_staged_outputs() || has_resident_output());
  }

  [[nodiscard]] constexpr bool has_staged_outputs() const noexcept {
    if (tile_count == 0u || output_bytes_per_tile == 0u) {
      return false;
    }
    if (output_buffer_count != 0u) {
      if (output_buffers == nullptr) {
        return false;
      }
      for (u64 index = 0u; index < output_buffer_count; ++index) {
        const OutputSpan &span = output_buffers[index];
        if (span.data == nullptr || span.element_bytes == 0u ||
            span.stride_bytes < span.element_bytes || span.count < tile_count) {
          return false;
        }
      }
      return true;
    }
    return staged_output != nullptr &&
           staged_output_stride >= output_bytes_per_tile &&
           staged_output_count >= tile_count;
  }

  [[nodiscard]] constexpr bool has_resident_output() const noexcept {
    return tile_count != 0u && output_bytes_per_tile != 0u &&
           resident_outputs.count != 0u && resident_outputs.has_refs();
  }

  template <typename T>
  [[nodiscard]] static BindingSet staged_outputs(T *const outputs,
                                                 const std::size_t count) {
    return BindingSet{
        .tile_count = static_cast<u64>(count),
        .output_bytes_per_tile = static_cast<u64>(sizeof(T)),
        .output_buffer_count = 0u,
        .staged_output = outputs,
        .staged_output_stride = static_cast<u64>(sizeof(T)),
        .staged_output_count = static_cast<u64>(count),
        .ok = outputs != nullptr && count != 0u,
        .reason = outputs != nullptr && count != 0u ? "ok"
                                                    : "compute_binding_invalid",
    };
  }

  template <typename T>
  [[nodiscard]] bool read_staged(const u64 tile_index, T &value) const {
    if constexpr (!std::is_trivially_copyable_v<T>) {
      return false;
    } else {
      constexpr u64 value_size = static_cast<u64>(sizeof(T));
      constexpr std::uintptr_t value_alignment =
          static_cast<std::uintptr_t>(alignof(T));
      const u64 max = ~u64{0u};
      if (tile_index >= tile_count || output_bytes_per_tile != value_size) {
        return false;
      }
      const void *output = staged_output;
      u64 stride = staged_output_stride;
      u64 count = staged_output_count;
      if (output_buffer_count == 1u && output_buffers != nullptr) {
        output = output_buffers[0].data;
        stride = output_buffers[0].stride_bytes;
        count = output_buffers[0].count;
      }
      if (output == nullptr || stride < value_size || tile_index >= count) {
        return false;
      }
      if (stride != 0u && tile_index > max / stride) {
        return false;
      }
      const u64 offset = tile_index * stride;
      if (offset > max - value_size) {
        return false;
      }
      const std::uintptr_t address = reinterpret_cast<std::uintptr_t>(output);
      if (value_alignment > 1u &&
          (address % value_alignment != 0u || stride % value_alignment != 0u)) {
        return false;
      }
      const auto *const bytes = static_cast<const std::byte *>(output);
      std::memcpy(&value, bytes + static_cast<std::size_t>(offset), sizeof(T));
      return true;
    }
  }

  [[nodiscard]] bool sequence_tile_at(const u64 sequence_index,
                                      u64 &tile_index) const noexcept {
    if (sequence_tiles == nullptr) {
      if (!has_resident_output() || sequence_tile_count != 0u ||
          sequence_index >= tile_count) {
        return false;
      }
      tile_index = sequence_index;
      return true;
    }
    if (sequence_index >= sequence_tile_count) {
      return false;
    }
    tile_index = sequence_tiles[sequence_index];
    return tile_index < tile_count;
  }
};

} // namespace rund::kernel
