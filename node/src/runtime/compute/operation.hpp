#pragma once

#include "../../compute/host.hpp"
#include "../../compute/pipeline/local.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::compute_detail {

struct TaskState;
struct Operation;

enum class DispatchDisposition : std::uint8_t {
  Failed,
  AcceptedNoBackend,
  BackendSubmitted,
};

class Dispatch final {
public:
  [[nodiscard]] static constexpr Dispatch
  failed(compute::Status failure) noexcept {
    if (failure) {
      failure = compute::Status::fail(compute::Reason::CompletionInvalid);
    }
    return Dispatch{failure};
  }

  [[nodiscard]] static constexpr Dispatch accepted_without_backend() noexcept {
    return Dispatch{DispatchDisposition::AcceptedNoBackend};
  }

  [[nodiscard]] static constexpr Dispatch backend_submitted() noexcept {
    return Dispatch{DispatchDisposition::BackendSubmitted};
  }

  [[nodiscard]] constexpr DispatchDisposition disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr compute::Status status() const noexcept {
    return disposition_ == DispatchDisposition::Failed
               ? failure_
               : compute::Status::success();
  }

private:
  explicit constexpr Dispatch(const compute::Status failure) noexcept
      : disposition_(DispatchDisposition::Failed), failure_(failure) {}

  explicit constexpr Dispatch(const DispatchDisposition disposition) noexcept
      : disposition_(disposition) {}

  DispatchDisposition disposition_ = DispatchDisposition::Failed;
  compute::Status failure_ =
      compute::Status::fail(compute::Reason::CompletionInvalid);
};

enum class AdvanceDisposition : std::uint8_t {
  Failed,
  Pending,
  BackendSubmitted,
  Complete,
};

class Advance final {
public:
  [[nodiscard]] static constexpr Advance
  failed(compute::Status failure) noexcept {
    if (failure) {
      failure = compute::Status::fail(compute::Reason::CompletionInvalid);
    }
    return Advance{failure};
  }

  [[nodiscard]] static constexpr Advance pending() noexcept {
    return Advance{AdvanceDisposition::Pending};
  }

  [[nodiscard]] static constexpr Advance backend_submitted() noexcept {
    return Advance{AdvanceDisposition::BackendSubmitted};
  }

  [[nodiscard]] static constexpr Advance complete() noexcept {
    return Advance{AdvanceDisposition::Complete};
  }

  [[nodiscard]] constexpr AdvanceDisposition disposition() const noexcept {
    return disposition_;
  }

  [[nodiscard]] constexpr compute::Status status() const noexcept {
    return disposition_ == AdvanceDisposition::Failed
               ? failure_
               : compute::Status::success();
  }

private:
  explicit constexpr Advance(const compute::Status failure) noexcept
      : disposition_(AdvanceDisposition::Failed), failure_(failure) {}

  explicit constexpr Advance(const AdvanceDisposition disposition) noexcept
      : disposition_(disposition) {}

  AdvanceDisposition disposition_ = AdvanceDisposition::Failed;
  compute::Status failure_ =
      compute::Status::fail(compute::Reason::CompletionInvalid);
};

// Session erases Job and Pipeline once, at admission. This is the only
// operation discriminator: coordinator, cancellation, publication, evidence,
// and frame accounting all dispatch through this immutable table.
struct OperationTable final {
  compute::Result<compute::Backend> (*backend)(
      const std::shared_ptr<void> &) noexcept = nullptr;
  kernel::u32 (*workers)(const std::shared_ptr<void> &) noexcept = nullptr;
  compute::Status (*reserve)(const std::shared_ptr<void> &) noexcept = nullptr;
  Dispatch (*submit_cpu)(const Operation &, TaskState &) noexcept =
      nullptr;
  Advance (*advance_cpu)(const Operation &, TaskState &) noexcept = nullptr;
  compute::Status (*result_cpu)(const Operation &, TaskState &) noexcept =
      nullptr;
  Dispatch (*submit_accel)(const Operation &, TaskState &) noexcept =
      nullptr;
  compute::Status (*result_accel)(const Operation &, TaskState &) noexcept =
      nullptr;
  compute::Status (*fail)(const std::shared_ptr<void> &,
                          compute::Status) noexcept = nullptr;
  compute::Status (*cancel)(const std::shared_ptr<void> &) noexcept = nullptr;
  compute::Stats (*evidence)(const std::shared_ptr<void> &) noexcept = nullptr;
  void (*record_frame)(const std::shared_ptr<void> &, std::uint64_t, bool,
                       std::uint64_t) noexcept = nullptr;
  void (*release_frame)(const std::shared_ptr<void> &,
                        std::uint64_t) noexcept = nullptr;
  void (*release)(std::shared_ptr<void> &) noexcept = nullptr;
};

struct Operation final {
  const OperationTable *table = nullptr;
  std::shared_ptr<void> owner{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return table != nullptr && owner != nullptr;
  }
};

[[nodiscard]] Operation
make_job(std::shared_ptr<compute::detail::JobState> state) noexcept;
[[nodiscard]] Operation
make_pipeline(std::shared_ptr<compute::detail::PipelineState> state) noexcept;
[[nodiscard]] Operation make_operation(std::shared_ptr<void> owner,
                                       const void *table) noexcept;

} // namespace rund::node::compute_detail
