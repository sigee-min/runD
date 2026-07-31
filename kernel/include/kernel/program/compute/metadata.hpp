#pragma once

#include <kernel/program/compute/retention.hpp>
#include <kernel/program/compute/ir.hpp>
#include <kernel/program/compute/model.hpp>

#include <string>
#include <vector>

namespace rund::kernel {

struct ReadRoute final {
  u32 source = 0u;
  u32 index = 0u;
  u32 count = 0u;

  [[nodiscard]] constexpr bool
  operator==(const ReadRoute &) const noexcept = default;
};

struct ExecutionMetadata {
  ComputeMap map{};
  std::vector<u8> param_storage{};
  std::vector<u64> input_element_bytes{};
  std::vector<u64> output_element_bytes{};
  std::vector<ComputeBindingAccess> binding_accesses{};
  std::vector<std::string> binding_names{};
  std::vector<ReadRoute> read_routes{};
  u64 direct_read_mask = 0u;
  u64 read_count = 0u;
  u64 write_count = 0u;
  bool uses_index = false;
  bool ok = false;
  const char *reason = "compute_ir_invalid";

  [[nodiscard]] explicit operator bool() const noexcept { return ok; }

  // Counts allocations owned below this object. The inline object itself is
  // counted by its enclosing owner.
  [[nodiscard]] u64 retained_dynamic_memory_bytes() const noexcept {
    using compute_retained_detail::Add;
    using compute_retained_detail::StringExternalStorageBytes;
    using compute_retained_detail::VectorCapacityBytes;
    u64 bytes = VectorCapacityBytes(param_storage);
    bytes = Add(bytes, VectorCapacityBytes(input_element_bytes));
    bytes = Add(bytes, VectorCapacityBytes(output_element_bytes));
    bytes = Add(bytes, VectorCapacityBytes(binding_accesses));
    bytes = Add(bytes, VectorCapacityBytes(binding_names));
    bytes = Add(bytes, VectorCapacityBytes(read_routes));
    for (const std::string &name : binding_names) {
      bytes = Add(bytes, StringExternalStorageBytes(name));
    }
    return bytes;
  }
};

[[nodiscard]] u64 RequiredInputCount(const ExecutionMetadata &metadata,
                                     u64 binding,
                                     u64 tile_count) noexcept;

[[nodiscard]] ExecutionMetadata BuildExecutionMetadata(const ComputeIR &ir,
                                                       ComputeApi api);

} // namespace rund::kernel
