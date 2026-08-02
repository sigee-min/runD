#include "local.hpp"

#include "../../../../src/array.hpp"
#include "../../../../src/compute/cpu/arena.hpp"
#include "../../../../src/compute/cpu/prepared.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>

namespace rund_node_memory_contract {

int CheckCpuSealedArena() {
  using namespace rund::compute::detail;

  CpuArenaLayout layout{};
  CpuArenaSegment bytes{};
  CpuArenaSegment words{};
  if (!append_cpu_arena_segment<std::uint8_t>(layout, 3u, bytes) ||
      !append_cpu_arena_segment<std::uint64_t>(layout, 2u, words) ||
      bytes.offset_bytes != 0u || bytes.size_bytes != 3u ||
      words.offset_bytes != 8u || words.size_bytes != 16u ||
      layout.extent_bytes != 24u || layout.maximum_alignment != 8u ||
      !seal_cpu_arena_layout(layout, 4096u) ||
      layout.committed_bytes != 4096u || !layout.sealed) {
    return 1;
  }
  CpuArenaSegment rejected{};
  if (append_cpu_arena_segment<std::uint32_t>(layout, 1u, rejected) ||
      seal_cpu_arena_layout(layout, 4096u)) {
    return 2;
  }

  CpuArenaMapping mapping{};
  if (!mapping.allocate(layout) || !mapping.valid() ||
      mapping.extent_bytes() != layout.extent_bytes ||
      mapping.committed_bytes() != layout.committed_bytes) {
    return 3;
  }
  const std::span<std::uint8_t> byte_values =
      mapping.construct<std::uint8_t>(bytes);
  const std::span<std::uint64_t> word_values =
      mapping.construct<std::uint64_t>(words);
  if (byte_values.size() != 3u || word_values.size() != 2u ||
      reinterpret_cast<std::uintptr_t>(word_values.data()) %
              alignof(std::uint64_t) !=
          0u) {
    return 4;
  }
  byte_values[2u] = 7u;
  word_values[1u] = 11u;
  if (byte_values[2u] != 7u || word_values[1u] != 11u) {
    return 5;
  }
  mapping.destroy(word_values);
  mapping.destroy(byte_values);
  mapping.release();
  if (mapping.committed_bytes() != 0u || mapping.extent_bytes() != 0u ||
      !mapping.valid()) {
    return 6;
  }

  CpuArenaLayout empty{};
  CpuArenaMapping empty_mapping{};
  if (!seal_cpu_arena_layout(empty, 4096u) || !empty_mapping.allocate(empty) ||
      !empty_mapping.valid() || empty.committed_bytes != 0u) {
    return 7;
  }

  CpuArenaLayout invalid{};
  if (append_cpu_arena_segment(invalid, 1u, 1u, 3u, rejected) ||
      seal_cpu_arena_layout(invalid, 3u)) {
    return 8;
  }

  using ::rund::node::detail::PreparedArray;
  std::array<std::uint32_t, 2u> prepared_storage{};
  PreparedArray<std::uint32_t> prepared;
  if (!prepared.bind(prepared_storage, 0u) || !prepared.borrowed() ||
      !prepared.empty() || prepared.capacity() != prepared_storage.size()) {
    return 9;
  }
  prepared.resize(prepared_storage.size());
  prepared[0u] = 3u;
  prepared[1u] = 5u;
  if (prepared_storage != std::array<std::uint32_t, 2u>{3u, 5u}) {
    return 10;
  }
  try {
    prepared.resize(prepared_storage.size() + 1u);
    return 11;
  } catch (const std::length_error &) {
  }
  PreparedArray<std::uint32_t> replacement;
  replacement.push_back(7u);
  try {
    prepared = std::move(replacement);
    return 12;
  } catch (const std::logic_error &) {
  }

  PreparedArray<std::uint32_t> empty_prepared;
  if (!empty_prepared.bind(std::span<std::uint32_t>{}, 0u) ||
      !empty_prepared.borrowed() || !empty_prepared.empty()) {
    return 13;
  }
  try {
    empty_prepared.push_back(1u);
    return 14;
  } catch (const std::length_error &) {
  }

  PreparedArray<std::uint32_t> retained_owner;
  retained_owner.reserve(4u);
  retained_owner.clear();
  if (retained_owner.bind(std::span<std::uint32_t>{}, 0u) ||
      retained_owner.borrowed() || retained_owner.owned_bytes() == 0u) {
    return 15;
  }

  CpuPreparedArenaPlan prepared_plan{};
  CpuWorkspaceSlice populated_slice{};
  CpuWorkspaceSlice empty_slice{};
  if (!append_cpu_workspace_slice(prepared_plan, 2u, populated_slice) ||
      !append_cpu_workspace_slice(prepared_plan, 0u, empty_slice) ||
      !seal_cpu_prepared_arena_plan(prepared_plan, 4096u)) {
    return 16;
  }
  auto prepared_arena = make_cpu_prepared_arena(prepared_plan);
  if (!prepared_arena) {
    return 17;
  }
  std::shared_ptr<CpuPreparedArena> prepared_owner =
      std::move(prepared_arena).value();
  CpuWorkspaceStorage populated{};
  CpuWorkspaceStorage empty_storage{};
  if (!prepared_owner->view(populated_slice, populated) ||
      !prepared_owner->view(empty_slice, empty_storage) ||
      populated.workspace == nullptr || populated.buffers.size() != 2u ||
      populated.offsets.size() != 2u || empty_storage.workspace == nullptr ||
      !empty_storage.buffers.empty() || !empty_storage.offsets.empty() ||
      populated.workspace == empty_storage.workspace) {
    return 18;
  }
  CpuWorkspaceSlice outside = populated_slice;
  outside.buffer_begin = prepared_plan.buffer_owner_count;
  outside.buffer_count = 1u;
  outside.offset_count = 1u;
  CpuWorkspaceSlice mismatched = populated_slice;
  --mismatched.offset_count;
  CpuWorkspaceSlice missing_workspace = populated_slice;
  missing_workspace.workspace_count = 0u;
  CpuWorkspaceStorage rejected_storage{};
  if (prepared_owner->view(outside, rejected_storage) ||
      prepared_owner->view(mismatched, rejected_storage) ||
      prepared_owner->view(missing_workspace, rejected_storage)) {
    return 19;
  }

  const std::weak_ptr<CpuPreparedArena> lifetime = prepared_owner;
  std::shared_ptr<JobWorkspace> workspace_owner(prepared_owner,
                                                populated.workspace);
  prepared_owner.reset();
  if (lifetime.expired() || workspace_owner.get() != populated.workspace) {
    return 20;
  }
  workspace_owner.reset();
  if (!lifetime.expired()) {
    return 21;
  }
  return 0;
}

} // namespace rund_node_memory_contract
