#pragma once

#include "../exception.hpp"
#include "sink.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/retention.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>

namespace rund::node::accel::detail::backend_source_recipe {

// std::string may round reserve() above the requested character count. The
// planner therefore keeps text bytes and external storage bytes as separate
// dimensions. One cache-line of slack covers the supported libc++, libstdc++,
// and MSVC string-capacity rounding; runtime verifies the actual capacity and
// fails closed before publication on any implementation that exceeds it.
inline constexpr std::uint64_t StringExternalStorageSlackBytes = 64u;

[[nodiscard]] inline bool string_external_storage_upper_bytes(
    const std::uint64_t text_upper_bytes,
    std::uint64_t &storage_upper_bytes) noexcept {
  if (text_upper_bytes == 0u) {
    storage_upper_bytes = 0u;
    return true;
  }
  return rund::kernel::checked::add(
      text_upper_bytes, StringExternalStorageSlackBytes, storage_upper_bytes);
}

[[nodiscard]] inline bool string_external_storage_within(
    const std::string &text, const std::uint64_t storage_upper_bytes) noexcept {
  return rund::kernel::compute_retained_detail::StringExternalStorageBytes(
             text) <= storage_upper_bytes;
}

// Grow by constructing one empty final owner rather than relying on geometric
// growth from the raw string. When growth is needed, the caller's planned raw
// source transient is the only allocation that coexists with this retained
// owner. A successful return guarantees both frozen character capacity and
// the external-storage envelope.
[[nodiscard]] inline bool
reserve_string(std::string &text, const std::uint64_t reserve_bytes) noexcept {
  if (text.size() > reserve_bytes ||
      reserve_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  std::uint64_t storage_upper_bytes = 0u;
  if (!string_external_storage_upper_bytes(reserve_bytes,
                                           storage_upper_bytes)) {
    return false;
  }
  try {
    if (text.capacity() < reserve_bytes) {
      std::string expanded;
      expanded.reserve(static_cast<std::size_t>(reserve_bytes));
      const std::size_t frozen_capacity = expanded.capacity();
      if (!string_external_storage_within(expanded, storage_upper_bytes)) {
        return false;
      }
      expanded.append(text.data(), text.size());
      if (expanded.capacity() != frozen_capacity) {
        return false;
      }
      text = std::move(expanded);
    }
    return text.capacity() >= reserve_bytes &&
           string_external_storage_within(text, storage_upper_bytes);
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return false;
  }
}

template <typename Emit>
[[nodiscard]] bool bytes(Emit &&emit, std::uint64_t &upper) noexcept {
  // Counting emitters are a strict no-throw contract. String materialization
  // may still throw from std::string and is contained by materialize().
  static_assert(std::is_nothrow_invocable_r_v<bool, Emit &, CountSink &>);
  CountSink sink{};
  if (!emit(sink) || sink.bytes() == 0u) {
    return false;
  }
  const std::uint64_t candidate = sink.bytes();
  upper = candidate;
  return true;
}

// Materialize a previously frozen exact count with one final reserve. `Emit`
// must be allocation-free apart from the StringSink append operations;
// CountSink and StringSink consume the exact same branch/fragment recipe.
template <typename Emit>
[[nodiscard]] std::string materialize(Emit &&emit,
                                      const std::uint64_t exact_bytes,
                                      const std::uint64_t reserve_bytes) {
  if (exact_bytes == 0u || exact_bytes > reserve_bytes ||
      reserve_bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return {};
  }
  try {
    std::string text;
    if (!reserve_string(text, reserve_bytes)) {
      return {};
    }
    const std::size_t frozen_capacity = text.capacity();
    std::uint64_t storage_upper_bytes = 0u;
    if (!string_external_storage_upper_bytes(reserve_bytes,
                                             storage_upper_bytes) ||
        !string_external_storage_within(text, storage_upper_bytes)) {
      return {};
    }
    StringSink sink{text};
    if (!emit(sink) || text.size() != exact_bytes ||
        text.capacity() != frozen_capacity ||
        !string_external_storage_within(text, storage_upper_bytes)) {
      return {};
    }
    return text;
  } catch (...) {
    backend_exception::RethrowUnlessCapacityException();
    return {};
  }
}

template <typename Emit>
[[nodiscard]] std::string materialize(Emit &&emit,
                                      const std::uint64_t exact_bytes) {
  return materialize(std::forward<Emit>(emit), exact_bytes, exact_bytes);
}

// Convenience path for callers that have not frozen a count yet.
template <typename Emit> [[nodiscard]] std::string materialize(Emit &&emit) {
  std::uint64_t upper = 0u;
  const auto count_emit = [&](CountSink &sink) noexcept { return emit(sink); };
  if (!bytes(count_emit, upper)) {
    return {};
  }
  return materialize(std::forward<Emit>(emit), upper);
}

} // namespace rund::node::accel::detail::backend_source_recipe
