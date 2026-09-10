#include "../internal.hpp"

#include <kernel/program/compute/lowering/metal/fixed.hpp>
#include <kernel/program/compute/lowering/metal/source.hpp>

namespace rund::node::accel::detail::device_vsm_graph_resident_source::detail {

std::string metal_source(
    const rund::kernel::ArtifactKey &key,
    const std::span<const DeviceVsmGraphResidentStageSource> stages,
    const DeviceVsmGraphResidentProof &proof,
    const DeviceVsmGraphWavefrontProof &wavefront) {
  namespace lowering = rund::kernel::compute_lowering_detail;
  if (stages.size() < 2u || stages.size() != proof.stage_count ||
      wavefront.stage_count != proof.stage_count ||
      proof.owner_binding_count == 0u ||
      !device_vsm_graph_resident_type_valid(proof.type)) {
    return {};
  }
  const std::size_t endpoints = external_count(proof);
  if (endpoints == 0u || endpoints > DeviceVsmGraphResidentResourceCapacity ||
      stages.front().input == nullptr) {
    return {};
  }
  const std::string value_type = graph_resident_metal_type(proof);
  std::string source = "#include <metal_stdlib>\nusing namespace metal;\n";
  source += "// rund.compute.device_vsm.graph_resident\n";
  lowering::AppendKey(source, key, "// ");
  const auto helpers = merged_helpers(stages);
  if (!helpers.ok) {
    return {};
  }
  lowering::AppendMetalHelpers(source, helpers, key);
  lowering::AppendMetalIntegerDivHelpers(source, helpers, key);
  source +=
      "struct RundGraphTable { uint logical_elements; uint payload_elements; "
      "uint page_count; uint element_words; uint element_bytes; uint valid; "
      "uint owner_count; uint controller_version; uint controller_stage_count; "
      "uint controller_frame_capacity; uint controller_batch_count; "
      "uint controller_map_stage; uint controller_collective_stage; "
      "uint controller_same_dispatch[" +
      std::to_string(DeviceVsmGraphStageCapacity) +
      "]; uint controller_same_release[" +
      std::to_string(DeviceVsmGraphStageCapacity) +
      "]; uint controller_prior_dispatch[" +
      std::to_string(DeviceVsmGraphStageCapacity) +
      "]; uint controller_prior_release[" +
      std::to_string(DeviceVsmGraphStageCapacity) + "]; uint owner_valid[" +
      std::to_string(DeviceVsmGraphResidentPhysicalCapacity) +
      "]; ulong owner_generations[" +
      std::to_string(DeviceVsmGraphResidentPhysicalCapacity *
                     DeviceVsmGraphResidentBankCapacity) +
      "]; };\n#define config graph_table\n"
      "kernel void rund_compute_map_";
  lowering::AppendHex64Digits(source, key.op_hash_hi);
  source += "_";
  lowering::AppendHex64Digits(source, key.op_hash_lo);
  source += "_device_vsm(\n    constant uint* graph_page_map [[buffer(0)]],\n"
            "    constant RundGraphTable& graph_table [[buffer(1)]],\n"
            "    device atomic_uint* result [[buffer(2)]],\n";
  for (std::size_t owner = 0u; owner < proof.owner_binding_count; ++owner) {
    for (std::size_t bank = 0u; bank < DeviceVsmGraphResidentBankCapacity;
         ++bank) {
      source += "    device " + value_type + "* graph_owner_" +
                std::to_string(owner) + "_" + std::to_string(bank) +
                " [[buffer(" +
                std::to_string(3u + owner * DeviceVsmGraphResidentBankCapacity +
                               bank) +
                ")]],\n";
    }
  }
  for (std::size_t endpoint = 0u; endpoint < endpoints; ++endpoint) {
    const auto *row = resource(proof, [&] {
      for (std::size_t index = 0u; index < proof.resource_count; ++index) {
        if (proof.resources[index].external_slot == endpoint) {
          return proof.resources[index].resource;
        }
      }
      return std::uint32_t{};
    }());
    if (row == nullptr || (row->role != 0u && row->role != 2u)) {
      return {};
    }
    source += "    " + std::string{row->role == 0u ? "const " : ""} +
              "device " + value_type + "* graph_endpoint_" +
              std::to_string(endpoint) +
              " [[buffer(" +
              std::to_string(3u +
                             proof.owner_binding_count *
                                 DeviceVsmGraphResidentBankCapacity +
                             endpoint) +
              ")]],\n";
  }
  source +=
      "    uint local [[thread_index_in_threadgroup]],\n"
      "    uint group_width [[threads_per_threadgroup]]) {\n"
      "  threadgroup uint control_batch; threadgroup uint control_stage;\n"
      "  threadgroup uint control_completed; threadgroup uint "
      "control_previous;\n"
      "  threadgroup uint control_map; threadgroup uint control_collective;\n"
      "  threadgroup uint control_invalid;\n"
      "  if (local == 0u) { control_batch = 0u; control_stage = 0u; "
      "control_completed = 0u; control_previous = 0u; control_map = 0u; "
      "control_collective = 0u; control_invalid = (graph_table.valid == 0u || "
      "graph_table.element_bytes != " +
      std::to_string(proof.type.element_bytes) +
      "u || graph_table.element_words != " +
      std::to_string(proof.type.element_bytes / sizeof(std::uint32_t)) +
      "u || graph_table.controller_version != " +
      std::to_string(DeviceVsmGraphControllerVersion) +
      "u || graph_table.controller_stage_count != " +
      std::to_string(wavefront.stage_count) +
      "u || graph_table.controller_frame_capacity != " +
      std::to_string(wavefront.frame_capacity) +
      "u || graph_table.controller_batch_count != " +
      std::to_string(wavefront.batch_count) +
      "u || graph_table.controller_map_stage >= "
      "graph_table.controller_stage_count "
      "|| graph_table.controller_collective_stage >= "
      "graph_table.controller_stage_count "
      "|| graph_table.controller_map_stage == "
      "graph_table.controller_collective_stage "
      "|| graph_table.controller_batch_count > 0xffffffffu / "
      "graph_table.controller_stage_count) ? 1u : 0u; }\n"
      "  threadgroup_barrier(mem_flags::mem_device | "
      "mem_flags::mem_threadgroup);\n"
      "  for (uint batch = 0u; batch < graph_table.controller_batch_count; "
      "++batch) {\n"
      "    if (control_invalid != 0u) { break; }\n"
      "    if (local == 0u) { control_batch = batch; control_completed = 0u; "
      "control_map = 0u; control_collective = 0u; }\n"
      "    threadgroup_barrier(mem_flags::mem_device | "
      "mem_flags::mem_threadgroup);\n"
      "    for (uint ordinal = 0u; ordinal < "
      "graph_table.controller_stage_count; ++ordinal) {\n"
      "      if (local == 0u) { uint pick = "
      "graph_table.controller_stage_count; "
      "for (uint candidate = 0u; candidate < "
      "graph_table.controller_stage_count; ++candidate) { "
      "const uint bit = 1u << candidate; const uint same = "
      "graph_table.controller_same_dispatch[candidate] | "
      "graph_table.controller_same_release[candidate]; const uint prior = "
      "graph_table.controller_prior_dispatch[candidate] | "
      "graph_table.controller_prior_release[candidate]; if ((control_completed "
      "& bit) == 0u && "
      "(same & ~control_completed) == 0u && (batch == 0u || (prior & "
      "~control_previous) == 0u)) "
      "{ pick = candidate; break; } } if (pick == "
      "graph_table.controller_stage_count || "
      "(pick == graph_table.controller_collective_stage && control_map == 0u)) "
      "control_invalid = 1u; "
      "else control_stage = pick; }\n"
      "      threadgroup_barrier(mem_flags::mem_device | "
      "mem_flags::mem_threadgroup);\n"
      "      if (control_invalid != 0u) { break; }\n"
      "      const uint active_batch = control_batch;\n"
      "      const uint selected = control_stage;\n"
      "      {\n"
      "        const uint first_page = active_batch * " +
      std::to_string(wavefront.frame_capacity) +
      "u; const uint page_limit = min(graph_table.page_count, first_page + " +
      std::to_string(wavefront.frame_capacity) +
      "u);\n        for (uint page = first_page; page < page_limit; ++page) {\n"
      "          const ulong page_begin = ulong(page) * "
      "ulong(graph_table.payload_elements);\n"
      "          const uint page_elements = "
      "uint(min(ulong(graph_table.logical_elements) - page_begin, "
      "ulong(graph_table.payload_elements)));\n"
      "          const uint slot = page % " +
      std::to_string(wavefront.frame_capacity) +
      "u;\n          const uint bank = (page / " +
      std::to_string(wavefront.frame_capacity) + "u) % " +
      std::to_string(DeviceVsmGraphResidentBankCapacity) +
      "u;\n          for (uint page_local = local; page_local < page_elements; "
      "page_local += group_width) {\n"
      "            const uint gid = uint(page_begin + ulong(page_local));\n";
  for (std::size_t stage = 0u; stage < stages.size(); ++stage) {
    source += "        if (selected == " + std::to_string(stage) + "u) {\n";
    if (!append_metal_stage(source, proof, proof.stages[stage], stages[stage],
                            key, "page_local", "slot", "bank",
                            wavefront.frame_capacity)) {
      return {};
    }
    source += "        }\n";
  }
  source +=
      "          }\n"
      "      if (local == 0u && selected == "
      "graph_table.controller_collective_stage) { for (uint word = 0u; "
      "word < 6u; ++word) "
      "atomic_fetch_add_explicit(&result[word], 1u, memory_order_relaxed); }\n"
      "        }\n"
      "      }\n"
      "      if (local == 0u) { control_completed |= 1u << selected; if "
      "(selected == graph_table.controller_map_stage) control_map = 1u; "
      "if (selected == graph_table.controller_collective_stage) "
      "control_collective = 1u; "
      "const uint old_step = atomic_fetch_add_explicit(&result[8], 1u, "
      "memory_order_relaxed); "
      "const uint old_trace = old_step == 0u ? " +
      std::to_string(DeviceVsmGraphTraceOffset) +
      "u : atomic_load_explicit(&result[9], memory_order_relaxed); const uint "
      "coordinate = active_batch * " +
      std::to_string(wavefront.stage_count) +
      "u + selected + 1u; atomic_store_explicit(&result[9], (old_trace ^ "
      "coordinate) * " +
      std::to_string(DeviceVsmGraphTracePrime) +
      "u, memory_order_relaxed); }\n"
      "      threadgroup_barrier(mem_flags::mem_device | "
      "mem_flags::mem_threadgroup);\n"
      "    }\n"
      "    if (local == 0u) { const uint all = (1u << "
      "graph_table.controller_stage_count) - 1u; if (control_completed != all "
      "|| control_map == 0u || control_collective == 0u) control_invalid = 1u; "
      "control_previous = control_completed; }\n"
      "    threadgroup_barrier(mem_flags::mem_device | "
      "mem_flags::mem_threadgroup);\n"
      "  }\n"
      "  if (local == 0u && control_invalid != 0u) "
      "atomic_store_explicit(&result[6], 1u, memory_order_relaxed);\n"
      "  threadgroup_barrier(mem_flags::mem_device | "
      "mem_flags::mem_threadgroup);\n}\n";
  return source;
}

} // namespace rund::node::accel::detail::device_vsm_graph_resident_source::detail
