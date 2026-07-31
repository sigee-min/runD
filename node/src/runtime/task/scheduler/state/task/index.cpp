#include "../../state/model/task.hpp"
#include "../../state/storage/check.hpp"
#include "../../state/storage.hpp"

#include <algorithm>
#include <limits>

namespace rund::node {
namespace {

constexpr std::uint32_t kIndexEmpty = 0u;
constexpr std::uint32_t kIndexDeleted =
    std::numeric_limits<std::uint32_t>::max();

[[nodiscard]] bool IndexOccupied(const std::uint32_t slot) noexcept {
  return slot != kIndexEmpty && slot != kIndexDeleted;
}

[[nodiscard]] bool IndexMatches(const SchedulerReadyState &ready,
                                const std::uint32_t entry,
                                const std::uint64_t id) noexcept {
  if (!IndexOccupied(entry)) {
    return false;
  }
  const std::size_t record_index = entry - 1u;
  return record_index < ready.records.size() &&
         ready.records[record_index].id == id;
}

void RebuildTaskIndex(SchedulerState &state,
                      const std::uint64_t retired_id) noexcept {
  auto &ready = state.ready;
  std::fill(ready.record_index_slots.begin(), ready.record_index_slots.end(),
            kIndexEmpty);
  ready.record_index_size = 0u;
  ready.record_index_deleted = 0u;
  for (std::size_t index = 0u; index < ready.records.size(); ++index) {
    const TaskRecord &record = ready.records[index];
    if (record.id != 0u && record.id != retired_id &&
        !state.IndexTask(record.id, index)) {
      std::abort();
    }
  }
}

} // namespace

TaskRecord *SchedulerState::Find(const std::uint64_t id) noexcept {
  RequireSequencer();
  const std::size_t index = IndexFor(id);
  if (index < ready.records.size() && ready.records[index].id == id) {
    return &ready.records[index];
  }
  return nullptr;
}

const TaskRecord *SchedulerState::Find(const std::uint64_t id) const noexcept {
  RequireSequencer();
  const std::size_t index = IndexFor(id);
  if (index < ready.records.size() && ready.records[index].id == id) {
    return &ready.records[index];
  }
  return nullptr;
}

TaskRecord *SchedulerState::FindAt(const std::size_t index,
                                   const std::uint64_t id) noexcept {
  RequireSequencer();
  if (index < ready.records.size() && ready.records[index].id == id) {
    return &ready.records[index];
  }
  return nullptr;
}

const TaskRecord *
SchedulerState::FindAt(const std::size_t index,
                       const std::uint64_t id) const noexcept {
  RequireSequencer();
  if (index < ready.records.size() && ready.records[index].id == id) {
    return &ready.records[index];
  }
  return nullptr;
}

bool SchedulerState::IndexTask(const std::uint64_t id,
                               const std::size_t index) noexcept {
  RequireSequencer();
  if (id == 0u || ready.record_index_slots.empty() ||
      index >= static_cast<std::size_t>(kIndexDeleted - 1u)) {
    return false;
  }
  const std::uint32_t encoded = static_cast<std::uint32_t>(index + 1u);
  const std::size_t capacity = ready.record_index_slots.size();
  std::size_t slot = static_cast<std::size_t>(id % capacity);
  std::size_t first_deleted = kInvalidTaskIndex;
  for (std::size_t probe = 0u; probe < capacity; ++probe) {
    const std::uint32_t entry = ready.record_index_slots[slot];
    if (IndexMatches(ready, entry, id)) {
      ready.record_index_slots[slot] = encoded;
      return true;
    }
    if (entry == kIndexDeleted && first_deleted == kInvalidTaskIndex) {
      first_deleted = slot;
    }
    if (entry == kIndexEmpty) {
      const std::size_t target =
          first_deleted == kInvalidTaskIndex ? slot : first_deleted;
      ready.record_index_slots[target] = encoded;
      ++ready.record_index_size;
      if (first_deleted != kInvalidTaskIndex) {
        --ready.record_index_deleted;
      }
      return true;
    }
    slot = (slot + 1u) % capacity;
  }
  if (first_deleted != kInvalidTaskIndex) {
    ready.record_index_slots[first_deleted] = encoded;
    ++ready.record_index_size;
    --ready.record_index_deleted;
    return true;
  }
  return false;
}

void SchedulerState::ForgetTask(const std::uint64_t id) noexcept {
  RequireSequencer();
  if (id == 0u || ready.record_index_slots.empty()) {
    return;
  }
  const std::size_t capacity = ready.record_index_slots.size();
  std::size_t slot = static_cast<std::size_t>(id % capacity);
  for (std::size_t probe = 0u; probe < capacity; ++probe) {
    const std::uint32_t entry = ready.record_index_slots[slot];
    if (entry == kIndexEmpty) {
      return;
    }
    if (IndexMatches(ready, entry, id)) {
      ready.record_index_slots[slot] = kIndexDeleted;
      --ready.record_index_size;
      ++ready.record_index_deleted;
      if (ready.record_index_size == 0u) {
        std::fill(ready.record_index_slots.begin(),
                  ready.record_index_slots.end(), kIndexEmpty);
        ready.record_index_deleted = 0u;
      } else if (ready.record_index_deleted > ready.record_index_size &&
                 ready.record_index_deleted >= (capacity + 3u) / 4u) {
        RebuildTaskIndex(*this, id);
      }
      return;
    }
    slot = (slot + 1u) % capacity;
  }
}

std::size_t SchedulerState::IndexFor(const std::uint64_t id) const noexcept {
  RequireSequencer();
  if (id == 0u || ready.record_index_slots.empty()) {
    return kInvalidTaskIndex;
  }
  const std::size_t capacity = ready.record_index_slots.size();
  std::size_t slot = static_cast<std::size_t>(id % capacity);
  for (std::size_t probe = 0u; probe < capacity; ++probe) {
    const std::uint32_t entry = ready.record_index_slots[slot];
    if (entry == kIndexEmpty) {
      return kInvalidTaskIndex;
    }
    if (IndexMatches(ready, entry, id)) {
      return static_cast<std::size_t>(entry - 1u);
    }
    slot = (slot + 1u) % capacity;
  }
  return kInvalidTaskIndex;
}

bool Scheduler::Matches(const task::Handle &handle,
                        const TaskRecord *record) const noexcept {
  state_->RequireSequencer();
  return record != nullptr &&
         ::rund::detail::task::HandleAccess::Scheduler(handle) ==
             state_->identity.scheduler_id &&
         handle.id() == record->id;
}

} // namespace rund::node
