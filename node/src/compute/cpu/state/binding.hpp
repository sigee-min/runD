#pragma once

#include "../../../accel/kernel/view.hpp"
#include "../../job/binding.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace rund::compute::detail {

struct BufferState;

// Exact immutable slice for one private Pipeline Job. Buffer owners and Views
// are mutable only during cold preparation (dense-view staging may rewrite a
// binding); warm execution reads the frozen prefix without allocating.
struct CpuJobBindingSlice final {
  std::size_t input_begin{};
  std::size_t input_count{};
  std::size_t output_begin{};
  std::size_t output_count{};
  std::size_t input_view_begin{};
  std::size_t input_view_count{};
  std::size_t output_view_begin{};
  std::size_t output_view_count{};
  std::size_t kernel_view_begin{};
  std::size_t kernel_view_count{};
  std::size_t input_transfer_begin{};
  std::size_t input_transfer_count{};
  std::size_t output_transfer_begin{};
  std::size_t output_transfer_count{};

  [[nodiscard]] constexpr bool
  operator==(const CpuJobBindingSlice &) const noexcept = default;
};

struct CpuJobBindingCounts final {
  std::size_t inputs{};
  std::size_t outputs{};
  std::size_t kernel_views{};
  std::size_t input_transfers{};
  std::size_t output_transfers{};

  [[nodiscard]] constexpr bool
  operator==(const CpuJobBindingCounts &) const noexcept = default;
};

struct CpuJobBindingStorage final {
  std::span<std::shared_ptr<BufferState>> inputs{};
  std::span<std::shared_ptr<BufferState>> outputs{};
  std::span<JobBufferView> input_views{};
  std::span<JobBufferView> output_views{};
  std::span<node::accel::detail::KernelViewSlot> kernel_views{};
  std::span<CpuViewTransfer> input_transfers{};
  std::span<CpuViewTransfer> output_transfers{};
};

struct CpuWorkspaceSlice final {
  std::size_t workspace_begin{};
  std::size_t workspace_count{};
  std::size_t buffer_begin{};
  std::size_t buffer_count{};
  std::size_t offset_begin{};
  std::size_t offset_count{};

  [[nodiscard]] constexpr bool
  operator==(const CpuWorkspaceSlice &) const noexcept = default;
};

struct CpuWorkspaceStorage final {
  JobWorkspace *workspace{};
  std::span<std::shared_ptr<BufferState>> buffers{};
  std::span<std::size_t> offsets{};
};

} // namespace rund::compute::detail
