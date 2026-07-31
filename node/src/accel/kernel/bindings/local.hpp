#pragma once

#include "build.hpp"

namespace rund::node::accel::detail {

struct BindingSource {
  const rund::kernel::ResidentBufferRef* refs = nullptr;
  const std::shared_ptr<void>* handles = nullptr;
  std::uint64_t size = 0u;
};

[[nodiscard]] BindingSource BindingSourceFor(
    const RunBinds& run_binds) noexcept;

[[nodiscard]] bool BindingSourceValid(
    const BindingSource& source) noexcept;

[[nodiscard]] bool ReadBinding(
    const BindingSource& source,
    std::uint64_t index,
    const rund::kernel::ResidentBufferRef *&ref,
    const std::shared_ptr<void> *&handle) noexcept;

[[nodiscard]] bool ReadBindingPair(
    const BindingSource &source, std::uint64_t first_index,
    const rund::kernel::ResidentBufferRef *&first,
    const std::shared_ptr<void> *&first_handle, std::uint64_t second_index,
    const rund::kernel::ResidentBufferRef *&second,
    const std::shared_ptr<void> *&second_handle) noexcept;

[[nodiscard]] inline bool
BindingReady(const std::shared_ptr<void> *const handle) noexcept {
  return handle != nullptr && *handle != nullptr;
}

}  // namespace rund::node::accel::detail
