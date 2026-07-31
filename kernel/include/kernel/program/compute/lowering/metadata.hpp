#pragma once

#include <kernel/program/compute/lowering/names.hpp>
#include <kernel/program/compute/metadata.hpp>

#include <algorithm>
#include <tuple>

namespace rund::kernel {
namespace compute_lowering_detail {

inline void AppendParamStorage(std::vector<u8> &storage,
                               const ParsedIR &parsed) {
  storage.clear();
  for (const ParsedBinding &binding : parsed.bindings) {
    if (binding.kind == kParamBindingKind) {
      storage.insert(storage.end(), binding.value_bytes.begin(),
                     binding.value_bytes.end());
    }
  }
}

[[nodiscard]] inline ExecutionMetadata
MetadataFromParsed(const ComputeIR &ir, const ComputeApi api,
                   const ParsedIR &parsed) {
  ExecutionMetadata metadata{};
  metadata.map = ComputeMap{
      .op_hash_hi = ir.op_hash_hi,
      .op_hash_lo = ir.op_hash_lo,
      .api = api,
      .scalar = ir.scalar,
      .domain = ir.domain,
      .fixed_format = ir.fixed_format,
      .output_buffer_count = 0u,
      .metadata_bytes_per_tile = kTileIndexMetadataBytes,
  };

  std::vector<u32> read_slots(parsed.bindings.size(), ~u32{0u});
  for (std::size_t binding_index = 0u;
       binding_index < parsed.bindings.size(); ++binding_index) {
    const ParsedBinding &binding = parsed.bindings[binding_index];
    if (binding.kind == kParamBindingKind) {
      metadata.map.param_bytes += binding.element_bytes;
    } else if (binding.kind == kReadBindingKind) {
      read_slots[binding_index] = static_cast<u32>(metadata.read_count);
      ++metadata.read_count;
      ++metadata.map.input_buffer_count;
      metadata.map.input_bytes_per_tile += binding.element_bytes;
      metadata.input_element_bytes.push_back(binding.element_bytes);
      metadata.binding_accesses.push_back(ComputeBindingAccess::Read);
      metadata.binding_names.push_back(binding.name);
    } else if (binding.kind == kWriteBindingKind) {
      ++metadata.write_count;
      ++metadata.map.output_buffer_count;
      metadata.map.output_bytes_per_tile += binding.element_bytes;
      metadata.output_element_bytes.push_back(binding.element_bytes);
      metadata.binding_accesses.push_back(ComputeBindingAccess::Write);
      metadata.binding_names.push_back(binding.name);
    }
  }

  AppendParamStorage(metadata.param_storage, parsed);
  for (const ParsedNode &node : parsed.nodes) {
    metadata.uses_index = metadata.uses_index ||
                          node.op == static_cast<u8>(IrOp::Index);
    if (node.op == static_cast<u8>(IrOp::Read) &&
        read_slots[node.aux] < 64u) {
      metadata.direct_read_mask |= u64{1u} << read_slots[node.aux];
    }
    if (node.op == static_cast<u8>(IrOp::ReadAt)) {
      const ReadRoute route{.source = read_slots[node.aux],
                            .index = read_slots[node.lhs],
                            .count = node.rhs};
      if (std::find(metadata.read_routes.begin(), metadata.read_routes.end(),
                    route) == metadata.read_routes.end()) {
        metadata.read_routes.push_back(route);
      }
    }
  }
  std::sort(metadata.read_routes.begin(), metadata.read_routes.end(),
            [](const ReadRoute left, const ReadRoute right) {
              return std::tie(left.source, left.index, left.count) <
                     std::tie(right.source, right.index, right.count);
            });
  metadata.ok = metadata.write_count != 0u;
  metadata.reason = metadata.ok ? "ok" : "compute_ir_node_invalid";
  return metadata;
}

} // namespace compute_lowering_detail

} // namespace rund::kernel
