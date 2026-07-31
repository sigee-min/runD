#pragma once

#include <kernel/program/compute/binding/model.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace rund::node::accel::detail {

template <std::size_t InlineCapacity>
struct ResidentRefs {
  std::array<rund::kernel::ResidentBufferRef, InlineCapacity> inline_refs{};
  std::array<std::shared_ptr<void>, InlineCapacity> inline_handles{};
  std::vector<rund::kernel::ResidentBufferRef> overflow_refs{};
  std::vector<std::shared_ptr<void>> overflow_handles{};
  std::uint64_t count = 0u;
  bool heap = false;
  bool ok = true;

  void clear() noexcept {
    count = 0u;
    overflow_refs.clear();
    overflow_handles.clear();
    heap = false;
    ok = true;
  }

  void reserve(const std::uint64_t expected) {
    if (!ok || expected >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
      ok = false;
      return;
    }
    if (expected > InlineCapacity) {
      if (count != 0u) {
        ok = false;
        return;
      }
      heap = true;
      overflow_refs.reserve(static_cast<std::size_t>(expected));
      overflow_handles.reserve(static_cast<std::size_t>(expected));
    }
  }

  [[nodiscard]] bool push(const rund::kernel::ResidentBufferRef ref,
                          std::shared_ptr<void> handle) {
    if (!ok || count == std::numeric_limits<std::uint64_t>::max()) {
      ok = false;
      return false;
    }
    if (!heap && count == InlineCapacity) {
      ok = false;
      return false;
    }
    if (!heap) {
      inline_refs[static_cast<std::size_t>(count)] = ref;
      inline_handles[static_cast<std::size_t>(count)] = std::move(handle);
    } else {
      overflow_refs.push_back(ref);
      overflow_handles.push_back(std::move(handle));
    }
    ++count;
    return true;
  }

  [[nodiscard]] bool valid() const noexcept {
    if (!ok || overflow_refs.empty() != overflow_handles.empty()) {
      return false;
    }
    return heap
               ? overflow_refs.size() == count &&
                     overflow_handles.size() == count
               : overflow_refs.empty() && count <= InlineCapacity;
  }

  [[nodiscard]] std::uint64_t size() const noexcept { return count; }

  [[nodiscard]] const rund::kernel::ResidentBufferRef* refs()
      const noexcept {
    if (count == 0u) {
      return nullptr;
    }
    return heap ? overflow_refs.data() : inline_refs.data();
  }

  [[nodiscard]] const std::shared_ptr<void>* handles() const noexcept {
    if (count == 0u) {
      return nullptr;
    }
    return heap ? overflow_handles.data() : inline_handles.data();
  }
};

static constexpr std::size_t kInlineBindingCapacity = 8u;
using RunBinds = ResidentRefs<kInlineBindingCapacity>;

template <std::size_t InlineCapacity>
struct ResidentViews {
  const rund::kernel::ResidentBufferRef *refs = nullptr;
  const std::shared_ptr<void> *handles = nullptr;
  std::uint64_t storage_count = 0u;
  std::array<std::uint64_t, InlineCapacity> inline_indices{};
  std::vector<std::uint64_t> overflow_indices{};
  std::uint64_t count = 0u;
  bool heap = false;
  bool ok = true;

  void bind(const RunBinds &source, const std::uint64_t expected) {
    refs = source.refs();
    handles = source.handles();
    storage_count = source.size();
    if (expected > InlineCapacity) {
      heap = true;
      overflow_indices.reserve(static_cast<std::size_t>(expected));
    }
  }

  [[nodiscard]] bool push(const std::uint64_t index) {
    if (!ok || refs == nullptr || handles == nullptr || index >= storage_count ||
        count == std::numeric_limits<std::uint64_t>::max()) {
      ok = false;
      return false;
    }
    if (!heap && count == InlineCapacity) {
      ok = false;
      return false;
    }
    if (heap) {
      overflow_indices.push_back(index);
    } else {
      inline_indices[static_cast<std::size_t>(count)] = index;
    }
    ++count;
    return true;
  }

  [[nodiscard]] bool valid() const noexcept {
    return ok && refs != nullptr && handles != nullptr &&
           (heap ? overflow_indices.size() == count
                 : overflow_indices.empty() && count <= InlineCapacity);
  }

  [[nodiscard]] std::uint64_t size() const noexcept { return count; }

  [[nodiscard]] rund::kernel::ResidentBindingRange range() const noexcept {
    if (count == 0u) {
      return {};
    }
    return rund::kernel::ResidentBindingRange{
        .refs = refs,
        .handles = handles,
        .indices = heap ? overflow_indices.data() : inline_indices.data(),
        .storage_count = storage_count,
        .count = count,
    };
  }
};

using StepViews = ResidentViews<kInlineBindingCapacity>;

}  // namespace rund::node::accel::detail
