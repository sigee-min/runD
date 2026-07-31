#pragma once

#include "../../compute/host.hpp"
#include "../../compute/pipeline/local.hpp"

#include <cstdint>
#include <memory>

namespace rund::node::compute_detail {

struct TaskState;
struct Operation;

struct Advance final {
  compute::Status status{compute::Status::success()};
  bool complete = false;
  bool backend_submitted = false;

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status);
  }
};

struct Dispatch final {
  compute::Status status{compute::Status::success()};
  bool backend_submitted = false;

  [[nodiscard]] explicit operator bool() const noexcept {
    return static_cast<bool>(status);
  }
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
