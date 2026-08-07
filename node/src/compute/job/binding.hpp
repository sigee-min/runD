#pragma once

#include "../../array.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail {

struct BufferState;
struct JobArena;
struct ProgramState;

struct JobBufferView final {
  std::size_t offset{};
  std::size_t count{};
  std::size_t stride{1u};
  std::size_t element_bytes{};
  std::size_t alignment{};
};

// CPU maps consume strided views directly. Reference collectives and
// primitives require dense ports, so only those external bindings receive a
// cold-prepared dense staging buffer. The original owner/view are retained
// here for allocation-free gather/publish on every execution.
struct CpuViewTransfer final {
  std::shared_ptr<BufferState> external;
  JobBufferView view{};
  std::uint32_t binding{};
};

// A recurrence owns one Program-internal value workspace for its complete
// lifetime. It carries the Program chunk table and, when present, the shared
// Pipeline JobArena used by accelerator View/scratch bindings. Its occurrences
// have distinct external binding routes, but execute serially behind the
// owning Pipeline gate and share this carrier without changing value or
// operation order. CPU dense-View transfers remain Job-owned and do not make a
// workspace present.
struct JobWorkspace final {
  JobWorkspace() noexcept = default;
  JobWorkspace(const JobWorkspace &) = delete;
  JobWorkspace &operator=(const JobWorkspace &) = delete;
  JobWorkspace(JobWorkspace &&) = delete;
  JobWorkspace &operator=(JobWorkspace &&) = delete;

  std::shared_ptr<ProgramState> program;
  ::rund::node::detail::PreparedArray<std::shared_ptr<BufferState>> buffers;
  ::rund::node::detail::PreparedArray<std::size_t> offsets;
  std::shared_ptr<JobArena> arena;
};

} // namespace rund::compute::detail
