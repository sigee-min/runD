#pragma once

#include <rund/compute/graph/info.hpp>
#include <rund/compute/status.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace rund::compute {

class Pipeline;
class PipelineBuilder;
class LatestDeviceState;
class SnapshotStorage;
class StateSnapshot;

namespace detail {

struct PipelineState;
struct PipelinePublicationState;
struct SnapshotStorageState;
struct StateSnapshotState;

[[nodiscard]] Result<std::shared_ptr<StateSnapshotState>>
snapshot_pipeline_state(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] Result<std::shared_ptr<PipelinePublicationState>>
latest_pipeline_state(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] Result<std::shared_ptr<SnapshotStorageState>>
make_snapshot_storage(const std::shared_ptr<PipelineState> &state,
                      std::size_t byte_capacity, bool exact) noexcept;
[[nodiscard]] Status snapshot_pipeline_into(
    const std::shared_ptr<PipelineState> &state,
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept;
[[nodiscard]] Status restore_pipeline_state(
    const std::shared_ptr<PipelineState> &state,
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept;
[[nodiscard]] Status restore_pipeline_state(
    const std::shared_ptr<PipelineState> &state,
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept;
[[nodiscard]] Status restore_pipeline_state(
    const std::shared_ptr<PipelineState> &state,
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept;
[[nodiscard]] bool
snapshot_valid(const std::shared_ptr<StateSnapshotState> &snapshot) noexcept;
[[nodiscard]] std::uint64_t snapshot_generation(
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept;
[[nodiscard]] graph::Fingerprint snapshot_fingerprint(
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept;
[[nodiscard]] std::uint64_t
snapshot_hash(const std::shared_ptr<StateSnapshotState> &snapshot) noexcept;
[[nodiscard]] bool latest_state_valid(
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept;
[[nodiscard]] std::uint64_t latest_state_generation(
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept;
[[nodiscard]] graph::Fingerprint latest_state_fingerprint(
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept;
[[nodiscard]] bool snapshot_storage_valid(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept;
[[nodiscard]] bool snapshot_storage_has_snapshot(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept;
[[nodiscard]] std::uint64_t snapshot_storage_generation(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept;
[[nodiscard]] graph::Fingerprint snapshot_storage_fingerprint(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept;
[[nodiscard]] std::uint64_t snapshot_storage_hash(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept;
[[nodiscard]] std::size_t snapshot_storage_capacity(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept;
[[nodiscard]] std::size_t snapshot_storage_field_capacity(
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept;

} // namespace detail

class StateSnapshot final {
public:
  StateSnapshot(const StateSnapshot &) noexcept = default;
  StateSnapshot &operator=(const StateSnapshot &) noexcept = default;
  StateSnapshot(StateSnapshot &&) noexcept = default;
  StateSnapshot &operator=(StateSnapshot &&) noexcept = default;

  [[nodiscard]] bool valid() const noexcept {
    return detail::snapshot_valid(state_);
  }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return detail::snapshot_generation(state_);
  }
  [[nodiscard]] graph::Fingerprint fingerprint() const noexcept {
    return detail::snapshot_fingerprint(state_);
  }
  [[nodiscard]] std::uint64_t hash() const noexcept {
    return detail::snapshot_hash(state_);
  }

private:
  friend class Pipeline;
  friend class PipelineBuilder;
  explicit StateSnapshot(
      std::shared_ptr<detail::StateSnapshotState> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<detail::StateSnapshotState> state_;
};

class LatestDeviceState final {
public:
  LatestDeviceState(const LatestDeviceState &) noexcept = default;
  LatestDeviceState &operator=(const LatestDeviceState &) noexcept = default;
  LatestDeviceState(LatestDeviceState &&) noexcept = default;
  LatestDeviceState &operator=(LatestDeviceState &&) noexcept = default;

  [[nodiscard]] bool valid() const noexcept {
    return detail::latest_state_valid(state_);
  }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return detail::latest_state_generation(state_);
  }
  [[nodiscard]] graph::Fingerprint fingerprint() const noexcept {
    return detail::latest_state_fingerprint(state_);
  }

private:
  friend class Pipeline;
  friend class PipelineBuilder;
  explicit LatestDeviceState(
      std::shared_ptr<detail::PipelinePublicationState> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<detail::PipelinePublicationState> state_;
};

class SnapshotStorage final {
public:
  SnapshotStorage(const SnapshotStorage &) = delete;
  SnapshotStorage &operator=(const SnapshotStorage &) = delete;
  SnapshotStorage(SnapshotStorage &&) noexcept = default;
  SnapshotStorage &operator=(SnapshotStorage &&) noexcept = default;

  [[nodiscard]] bool valid() const noexcept {
    return detail::snapshot_storage_valid(state_);
  }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
  [[nodiscard]] bool has_snapshot() const noexcept {
    return detail::snapshot_storage_has_snapshot(state_);
  }
  [[nodiscard]] std::uint64_t generation() const noexcept {
    return detail::snapshot_storage_generation(state_);
  }
  [[nodiscard]] graph::Fingerprint fingerprint() const noexcept {
    return detail::snapshot_storage_fingerprint(state_);
  }
  [[nodiscard]] std::uint64_t hash() const noexcept {
    return detail::snapshot_storage_hash(state_);
  }
  [[nodiscard]] std::size_t capacity() const noexcept {
    return detail::snapshot_storage_capacity(state_);
  }
  [[nodiscard]] std::size_t field_capacity() const noexcept {
    return detail::snapshot_storage_field_capacity(state_);
  }

private:
  friend class Pipeline;
  friend class PipelineBuilder;
  explicit SnapshotStorage(
      std::shared_ptr<detail::SnapshotStorageState> state) noexcept
      : state_(std::move(state)) {}
  std::shared_ptr<detail::SnapshotStorageState> state_;
};

} // namespace rund::compute
