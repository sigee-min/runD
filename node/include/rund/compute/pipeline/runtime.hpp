#pragma once

#include <rund/compute/pipeline/bind.hpp>
#include <rund/compute/pipeline/memory.hpp>
#include <rund/compute/pipeline/profile.hpp>
#include <rund/compute/pipeline/snapshot.hpp>

#include <rund/compute/graph/info.hpp>
#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>
#include <rund/compute/telemetry.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace rund {
class Session;
}

namespace rund::compute {

class Pipeline;
class PipelineBuilder;
class HostIteration;

namespace detail {

struct HostFeedbackAccess;
struct PipelineStateAccess;

[[nodiscard]] bool
valid_pipeline(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] bool
poisoned_pipeline(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] Status
begin_pipeline_samples(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] Status
end_pipeline_samples(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] Status
run_pipeline(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] Stats
pipeline_stats(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] CheckpointStats
pipeline_checkpoint_stats(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] MemoryStats
pipeline_memory(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] MemorySnapshot
pipeline_memory_snapshot(const std::shared_ptr<PipelineState> &state,
                         std::span<MemoryEntry> entries) noexcept;
[[nodiscard]] Result<PipelineProfileSnapshot>
pipeline_profile(const std::shared_ptr<PipelineState> &state,
                 std::span<PipelineStepProfile> steps) noexcept;
[[nodiscard]] Result<telemetry::Profile>
pipeline_profile(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] graph::Fingerprint
pipeline_fingerprint(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] std::uint64_t
pipeline_generation(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] Status
read_pipeline_raw(const std::shared_ptr<PipelineState> &state,
                  const std::shared_ptr<BufferState> &buffer, Type type,
                  FixedFormat format, void *data, std::size_t bytes,
                  std::size_t count) noexcept;
[[nodiscard]] Status
write_pipeline_raw(const std::shared_ptr<PipelineState> &state,
                   const std::shared_ptr<BufferState> &buffer, HostView input,
                   WriteStats &writes) noexcept;
[[nodiscard]] PipelinePlan
pipeline_plan(const std::shared_ptr<PipelineState> &state) noexcept;

} // namespace detail

class Pipeline final {
public:
  Pipeline(const Pipeline &) = delete;
  Pipeline &operator=(const Pipeline &) = delete;
  Pipeline(Pipeline &&) noexcept = default;
  Pipeline &operator=(Pipeline &&) noexcept = default;

  [[nodiscard]] bool valid() const noexcept {
    return detail::valid_pipeline(state_);
  }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] bool poisoned() const noexcept {
    return detail::poisoned_pipeline(state_);
  }
  [[nodiscard]] Status begin_samples() noexcept {
    return detail::begin_pipeline_samples(state_);
  }
  [[nodiscard]] Status end_samples() noexcept {
    return detail::end_pipeline_samples(state_);
  }
  [[nodiscard]] Status run() noexcept { return detail::run_pipeline(state_); }
  [[nodiscard]] Stats stats() const noexcept {
    return detail::pipeline_stats(state_);
  }
  [[nodiscard]] CheckpointStats checkpoint_stats() const noexcept {
    return detail::pipeline_checkpoint_stats(state_);
  }
  [[nodiscard]] MemoryStats memory() const noexcept {
    return detail::pipeline_memory(state_);
  }
  [[nodiscard]] PipelinePlan plan() const noexcept {
    return detail::pipeline_plan(state_);
  }
  [[nodiscard]] MemorySnapshot
  memory_snapshot(const std::span<MemoryEntry> entries) const noexcept {
    return detail::pipeline_memory_snapshot(state_, entries);
  }
  [[nodiscard]] Result<PipelineProfileSnapshot>
  profile(const std::span<PipelineStepProfile> steps) const noexcept {
    return detail::pipeline_profile(state_, steps);
  }
  [[nodiscard]] Result<telemetry::Profile> profile() const noexcept {
    return detail::pipeline_profile(state_);
  }
  [[nodiscard]] graph::Fingerprint fingerprint() const noexcept {
    return detail::pipeline_fingerprint(state_);
  }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return detail::pipeline_generation(state_);
  }
  [[nodiscard]] Result<StateSnapshot> snapshot() const noexcept {
    auto saved = detail::snapshot_pipeline_state(state_);
    if (!saved) {
      return Result<StateSnapshot>::fail(saved.reason());
    }
    return Result<StateSnapshot>::success(
        StateSnapshot{std::move(saved).value()});
  }
  [[nodiscard]] Status restore(const StateSnapshot &snapshot) noexcept {
    return detail::restore_pipeline_state(state_, snapshot.state_);
  }
  [[nodiscard]] Result<LatestDeviceState> latest_device_state() const noexcept {
    auto latest = detail::latest_pipeline_state(state_);
    if (!latest) {
      return Result<LatestDeviceState>::fail(latest.reason());
    }
    return Result<LatestDeviceState>::success(
        LatestDeviceState{std::move(latest).value()});
  }
  [[nodiscard]] Result<SnapshotStorage> snapshot_storage() const noexcept {
    auto storage = detail::make_snapshot_storage(state_, 0u, true);
    if (!storage) {
      return Result<SnapshotStorage>::fail(storage.reason());
    }
    return Result<SnapshotStorage>::success(
        SnapshotStorage{std::move(storage).value()});
  }
  [[nodiscard]] Result<SnapshotStorage>
  snapshot_storage(const std::size_t byte_capacity) const noexcept {
    auto storage = detail::make_snapshot_storage(state_, byte_capacity, false);
    if (!storage) {
      return Result<SnapshotStorage>::fail(storage.reason());
    }
    return Result<SnapshotStorage>::success(
        SnapshotStorage{std::move(storage).value()});
  }
  [[nodiscard]] Status snapshot_into(SnapshotStorage &storage) const noexcept {
    return detail::snapshot_pipeline_into(state_, storage.state_);
  }
  [[nodiscard]] Status restore(const LatestDeviceState &latest) noexcept {
    return detail::restore_pipeline_state(state_, latest.state_);
  }
  [[nodiscard]] Status restore(const SnapshotStorage &storage) noexcept {
    return detail::restore_pipeline_state(state_, storage.state_);
  }

  template <class T>
  [[nodiscard]] Status
  read(const Buffer<T> &buffer,
       const std::span<std::type_identity_t<T>> output) const noexcept {
    return detail::read_pipeline_raw(
        state_, detail::BufferAccess::state(buffer), detail::type<T>(),
        detail::storage_format<T>(), output.data(), output.size_bytes(),
        output.size());
  }

private:
  friend class PipelineBuilder;
  friend class HostIteration;
  friend class ::rund::Session;
  friend struct detail::PipelineStateAccess;
  explicit Pipeline(std::shared_ptr<detail::PipelineState> state) noexcept
      : state_(std::move(state)) {}

  std::shared_ptr<detail::PipelineState> state_{};
};

class HostIteration final {
public:
  HostIteration(const HostIteration &) = delete;
  HostIteration &operator=(const HostIteration &) = delete;
  HostIteration(HostIteration &&) = delete;
  HostIteration &operator=(HostIteration &&) = delete;

  [[nodiscard]] std::size_t completed() const noexcept { return completed_; }
  [[nodiscard]] std::size_t total() const noexcept { return total_; }
  [[nodiscard]] std::size_t remaining() const noexcept {
    return total_ - completed_;
  }
  [[nodiscard]] bool has_next() const noexcept { return completed_ < total_; }
  [[nodiscard]] Stats stats() const noexcept { return pipeline_->stats(); }
  [[nodiscard]] Result<telemetry::Profile> profile() const noexcept {
    return pipeline_->profile();
  }
  [[nodiscard]] WriteStats write_stats() const noexcept { return writes_; }

  template <class T>
  [[nodiscard]] Status
  read(const Buffer<T> &buffer,
       const std::span<std::type_identity_t<T>> output) const noexcept {
    return pipeline_->read(buffer, output);
  }

  template <class T>
  [[nodiscard]] Status
  write(Buffer<T> &buffer,
        const std::span<const std::type_identity_t<T>> input) noexcept {
    if (!has_next()) {
      return Status::fail(Reason::AlreadyCompleted);
    }
    return detail::write_pipeline_raw(
        pipeline_->state_, detail::BufferAccess::state(buffer),
        detail::HostView{input.data(), input.size(), detail::type<T>()},
        writes_);
  }

private:
  friend struct detail::HostFeedbackAccess;
  HostIteration(Pipeline &pipeline, const std::size_t completed,
                const std::size_t total) noexcept
      : pipeline_(&pipeline), completed_(completed), total_(total) {}

  Pipeline *pipeline_{};
  std::size_t completed_{};
  std::size_t total_{};
  WriteStats writes_{};
};

} // namespace rund::compute
