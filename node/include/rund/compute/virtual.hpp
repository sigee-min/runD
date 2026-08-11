#pragma once

#include <rund/compute/abi/virtual.hpp>
#include <rund/compute/pipeline.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace rund::compute {

struct VirtualRead final {
  std::uint64_t offset{};
  std::span<std::byte> bytes;
};

struct VirtualWrite final {
  std::uint64_t offset{};
  std::span<const std::byte> bytes;
};

// Physical latency class of the logical backing. This selects bounded
// prefetch distance only; it never changes ordering, cache identity, or
// correctness. Host is the default for existing memory-backed owners.
enum class VirtualBackingTier : std::uint8_t { Host, Persistent };

template <class Signature> class VirtualPipeline;
namespace detail {
struct VirtualBackingState;
struct VirtualBackingAccess;
} // namespace detail

// VirtualBacking is the logical dataset authority. Implementations may use
// memory, a file, object storage, or deterministic generation; runD requests
// bounded byte ranges and never assumes that the complete dataset is mapped.
class VirtualBacking {
public:
  VirtualBacking();
  VirtualBacking(const VirtualBacking &) = delete;
  VirtualBacking &operator=(const VirtualBacking &) = delete;
  virtual ~VirtualBacking();

  [[nodiscard]] virtual std::uint64_t size_bytes() const noexcept = 0;
  [[nodiscard]] virtual VirtualBackingTier tier() const noexcept {
    return VirtualBackingTier::Host;
  }
  // One is the serialized default. A read-only backing may explicitly admit
  // bounded parallel range reads; runD never overlaps writes or exceeds two.
  [[nodiscard]] virtual std::uint32_t max_parallel_reads() const noexcept {
    return 1u;
  }
  [[nodiscard]] virtual Status read(std::uint64_t offset,
                                    std::span<std::byte> output) noexcept = 0;
  [[nodiscard]] virtual Status
  write(std::uint64_t offset, std::span<const std::byte> input) noexcept = 0;
  // Batch is the physical page-store interface. The default preserves custom
  // backings by issuing the exact ordered scalar callbacks; NVMe/object-store
  // implementations may override it with vectored I/O without changing page
  // order or partial-failure semantics.
  [[nodiscard]] virtual Status
  read_batch(std::span<const VirtualRead> ranges) noexcept;
  [[nodiscard]] virtual Status
  write_batch(std::span<const VirtualWrite> ranges) noexcept;
  // External mutation must be published through this boundary before the
  // backing is used again. It advances the cache generation while serialized
  // with every run using this backing; hidden mutation is a contract error.
  [[nodiscard]] Status invalidate() noexcept;

private:
  friend struct detail::VirtualBackingAccess;

  std::unique_ptr<detail::VirtualBackingState> state_;
};

template <detail::ComputeValue T> class VirtualBuffer final {
public:
  VirtualBuffer(const VirtualBuffer &) noexcept = default;
  VirtualBuffer(VirtualBuffer &&) noexcept = default;
  VirtualBuffer &operator=(const VirtualBuffer &) noexcept = default;
  VirtualBuffer &operator=(VirtualBuffer &&) noexcept = default;

  [[nodiscard]] bool valid() const noexcept {
    return detail::valid_virtual_buffer(state_);
  }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] std::uint64_t size() const noexcept {
    return detail::virtual_buffer_size(state_);
  }

private:
  template <class> friend class VirtualPipeline;
  template <detail::ComputeValue U>
  friend Result<VirtualBuffer<U>>
  virtual_buffer(std::uint64_t, std::shared_ptr<VirtualBacking>) noexcept;
  template <detail::ComputeValue R, detail::ComputeValue A>
  friend Result<VirtualPipeline<R(A)>>
  virtual_pipeline(const Program<R(A)> &, const VirtualBuffer<A> &,
                   VirtualBuffer<R> &, struct ResidencyConfig) noexcept;

  explicit VirtualBuffer(std::shared_ptr<detail::VirtualBufferState> state)
      : state_(std::move(state)) {}

  std::shared_ptr<detail::VirtualBufferState> state_;
};

template <detail::ComputeValue T>
[[nodiscard]] Result<VirtualBuffer<T>>
virtual_buffer(const std::uint64_t count,
               std::shared_ptr<VirtualBacking> backing) noexcept {
  auto state = detail::make_virtual_buffer(count, sizeof(T), detail::type<T>(),
                                           detail::storage_format<T>(),
                                           std::move(backing));
  if (!state) {
    return Result<VirtualBuffer<T>>::fail(state.reason());
  }
  return Result<VirtualBuffer<T>>::success(
      VirtualBuffer<T>{std::move(state).value()});
}

template <detail::ComputeValue R, detail::ComputeValue A>
class VirtualPipeline<R(A)> final {
public:
  VirtualPipeline(const VirtualPipeline &) = delete;
  VirtualPipeline &operator=(const VirtualPipeline &) = delete;
  VirtualPipeline(VirtualPipeline &&) noexcept = default;
  VirtualPipeline &operator=(VirtualPipeline &&) noexcept = default;

  [[nodiscard]] bool valid() const noexcept {
    return detail::valid_virtual_pipeline(state_);
  }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] Status begin_samples() noexcept {
    return detail::begin_virtual_pipeline_samples(state_);
  }
  [[nodiscard]] Status end_samples() noexcept {
    return detail::end_virtual_pipeline_samples(state_);
  }
  [[nodiscard]] Status run() noexcept {
    return detail::run_virtual_pipeline(state_);
  }
  [[nodiscard]] Status run(const std::uint64_t active_count) noexcept {
    return detail::run_virtual_pipeline(state_, active_count);
  }
  [[nodiscard]] Stats stats() const noexcept {
    return detail::virtual_pipeline_stats(state_);
  }
  [[nodiscard]] MemoryStats memory() const noexcept {
    return detail::virtual_pipeline_memory(state_);
  }
  [[nodiscard]] PipelinePlan plan() const noexcept {
    return detail::virtual_pipeline_plan(state_);
  }
  [[nodiscard]] Result<telemetry::Profile> profile() const noexcept {
    return detail::virtual_pipeline_profile(state_);
  }

private:
  template <detail::ComputeValue U, detail::ComputeValue T>
  friend Result<VirtualPipeline<U(T)>>
  virtual_pipeline(const Program<U(T)> &, const VirtualBuffer<T> &,
                   VirtualBuffer<U> &, ResidencyConfig) noexcept;

  explicit VirtualPipeline(
      std::shared_ptr<detail::VirtualPipelineState> state) noexcept
      : state_(std::move(state)) {}

  std::shared_ptr<detail::VirtualPipelineState> state_;
};

template <detail::ComputeValue R, detail::ComputeValue A>
[[nodiscard]] Result<VirtualPipeline<R(A)>>
virtual_pipeline(const Program<R(A)> &program, const VirtualBuffer<A> &input,
                 VirtualBuffer<R> &output,
                 const ResidencyConfig config = {}) noexcept {
  auto prepared =
      detail::prepare_virtual_pipeline(detail::ProgramAccess::state(program),
                                       input.state_, output.state_, config);
  if (!prepared) {
    return Result<VirtualPipeline<R(A)>>::fail(prepared.reason(),
                                               prepared.location());
  }
  return Result<VirtualPipeline<R(A)>>::success(
      VirtualPipeline<R(A)>{std::move(prepared).value()});
}

static_assert(std::is_trivially_copyable_v<ResidencyConfig>);
static_assert(std::is_trivially_copyable_v<VirtualRead>);
static_assert(std::is_trivially_copyable_v<VirtualWrite>);

} // namespace rund::compute
