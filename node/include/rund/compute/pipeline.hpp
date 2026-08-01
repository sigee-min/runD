#pragma once

#include <rund/compute/device.hpp>
#include <rund/compute/graph/info.hpp>
#include <rund/compute/pipeline/bind.hpp>
#include <rund/compute/pipeline/memory.hpp>
#include <rund/compute/pipeline/profile.hpp>
#include <rund/compute/pipeline/shape.hpp>
#include <rund/compute/pipeline/tile.hpp>
#include <rund/compute/pipeline/window.hpp>
#include <rund/compute/program.hpp>
#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
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
class LatestDeviceState;
class SnapshotStorage;
class StateSnapshot;

namespace detail {

struct HostFeedbackAccess;
struct PipelinePublicationState;
struct SnapshotStorageState;
struct PipelineStateAccess;

struct DeviceAccess final {
  [[nodiscard]] static const std::shared_ptr<DeviceState> &
  state(const Device &device) noexcept {
    return device.state_;
  }
};

struct ProgramAccess final {
  template <class Signature>
  [[nodiscard]] static const std::shared_ptr<ProgramState> &
  state(const Program<Signature> &program) noexcept {
    return program.state_;
  }
};

template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
          std::size_t N, class SeedSignature, class ActionSignature,
          class FoldSignature, class Inputs, class Finals, class Windows>
struct TileRepeatWindowBindingContract : std::false_type {
  static constexpr bool valid = false;
};

template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
          std::size_t N, class SeedSignature, class ActionSignature,
          class FoldSignature, class... I, class... O, class... W>
  requires(N != 0u && !std::is_void_v<ActionSignature>)
struct TileRepeatWindowBindingContract<
    Max, Tile, Terminal, N, SeedSignature, ActionSignature, FoldSignature,
    TypeList<I...>, TypeList<O...>, TypeList<W...>>
    final {
  static constexpr bool valid =
      Max != 0u && Tile != 0u && Tile <= Max &&
      Max / Tile + (Max % Tile == 0u ? 0u : 1u) <= PipelineIterationCapacity &&
      N != 0u && N <= PipelineInnerIterationCapacity &&
      TileWindowContract<SeedSignature, ActionSignature, FoldSignature,
                         TypeList<O...>, TypeList<W...>>::valid &&
      std::is_same_v<
          TypeList<I...>,
          typename Join<TypeList<O...>,
                        typename TileRepeatContract<
                            SeedSignature, ActionSignature,
                            FoldSignature>::SeedExternalInputs>::type> &&
      (Terminal == NoWindowTerminal ||
       (Terminal < sizeof...(O) && U32At<Terminal, TypeList<O...>>::value));
};

template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
          class SeedSignature, class FoldSignature, class Inputs, class Finals,
          class Windows>
struct TileFoldWindowBindingContract : std::false_type {};

template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
          class SeedSignature, class FoldSignature, class... I, class... O,
          class... W>
struct TileFoldWindowBindingContract<Max, Tile, Terminal, SeedSignature,
                                     FoldSignature, TypeList<I...>,
                                     TypeList<O...>, TypeList<W...>>
    final {
  static constexpr bool valid =
      Max != 0u && Tile != 0u && Tile <= Max &&
      Max / Tile + (Max % Tile == 0u ? 0u : 1u) <= PipelineIterationCapacity &&
      TileWindowFoldContract<SeedSignature, FoldSignature, TypeList<O...>,
                             TypeList<W...>>::valid &&
      std::is_same_v<
          TypeList<I...>,
          typename Join<TypeList<O...>, typename TileFoldContract<
                                            SeedSignature, FoldSignature>::
                                            SeedExternalInputs>::type> &&
      (Terminal == NoWindowTerminal ||
       (Terminal < sizeof...(O) && U32At<Terminal, TypeList<O...>>::value));
};

[[nodiscard]] std::shared_ptr<PipelineBuildState>
make_pipeline(const std::shared_ptr<DeviceState> &device) noexcept;
void append_pipeline(const std::shared_ptr<PipelineBuildState> &build,
                     const std::shared_ptr<ProgramState> &program,
                     std::span<const ResourceView> inputs,
                     std::span<const ResourceView> outputs) noexcept;
void append_pipeline_repeat(const std::shared_ptr<PipelineBuildState> &build,
                            const std::shared_ptr<ProgramState> &program,
                            std::span<const ResourceView> inputs,
                            std::span<const ResourceView> outputs,
                            std::size_t iterations) noexcept;
void append_pipeline_repeat_each(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &program,
    std::span<const ResourceView> inputs, std::span<const ResourceView> outputs,
    std::size_t iterations) noexcept;
void append_pipeline_windows(const std::shared_ptr<PipelineBuildState> &build,
                             const std::shared_ptr<ProgramState> &program,
                             const ResourceView &resident,
                             std::span<const ResourceView> inputs,
                             std::span<const ResourceView> outputs,
                             std::size_t maximum, std::size_t tile,
                             std::size_t terminal,
                             std::uint32_t expected) noexcept;
void append_pipeline_window_repeat(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<ProgramState> &seed,
    const std::shared_ptr<ProgramState> &action,
    const std::shared_ptr<ProgramState> &fold, const ResourceView &resident,
    std::span<const ResourceView> inputs,
    std::span<const ResourceView> final_outputs,
    std::span<const ResourceView> window_outputs, std::size_t maximum,
    std::size_t tile, std::size_t inner, std::size_t terminal,
    std::uint32_t expected) noexcept;
void append_pipeline_state(const std::shared_ptr<PipelineBuildState> &build,
                           const std::shared_ptr<BufferState> &published,
                           const std::shared_ptr<BufferState> &pending,
                           Type type, FixedFormat format) noexcept;
void configure_pipeline_profile(
    const std::shared_ptr<PipelineBuildState> &build,
    PipelineProfile profile) noexcept;
void configure_pipeline_sealed_repetitions(
    const std::shared_ptr<PipelineBuildState> &build,
    std::size_t repetitions) noexcept;
void commit_pipeline(const std::shared_ptr<PipelineBuildState> &build) noexcept;
void seed_pipeline(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept;
void seed_pipeline(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<PipelinePublicationState> &publication) noexcept;
void seed_pipeline(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<SnapshotStorageState> &storage) noexcept;
[[nodiscard]] Result<std::shared_ptr<PipelineState>>
prepare_pipeline(std::shared_ptr<PipelineBuildState> build) noexcept;
[[nodiscard]] Result<PipelinePlan>
plan_pipeline(const std::shared_ptr<PipelineBuildState> &build) noexcept;
[[nodiscard]] PipelinePlan
pipeline_plan(const std::shared_ptr<PipelineState> &state) noexcept;
void configure_pipeline_budget(const std::shared_ptr<PipelineBuildState> &build,
                               MemoryBudget budget) noexcept;
[[nodiscard]] bool
valid_pipeline(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] bool
poisoned_pipeline(const std::shared_ptr<PipelineState> &state) noexcept;
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
[[nodiscard]] graph::Fingerprint
pipeline_fingerprint(const std::shared_ptr<PipelineState> &state) noexcept;
[[nodiscard]] std::uint64_t
pipeline_generation(const std::shared_ptr<PipelineState> &state) noexcept;
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
[[nodiscard]] Status
read_pipeline_raw(const std::shared_ptr<PipelineState> &state,
                  const std::shared_ptr<BufferState> &buffer, Type type,
                  FixedFormat format, void *data, std::size_t bytes,
                  std::size_t count) noexcept;

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
    return detail::write_buffer(
        detail::BufferAccess::state(buffer),
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

namespace detail {

struct HostFeedbackAccess final {
  [[nodiscard]] static HostIteration
  iteration(Pipeline &pipeline, const std::size_t completed,
            const std::size_t total) noexcept {
    return HostIteration{pipeline, completed, total};
  }
};

} // namespace detail

class PipelineBuilder final {
public:
  PipelineBuilder(const PipelineBuilder &) = delete;
  PipelineBuilder &operator=(const PipelineBuilder &) = delete;
  PipelineBuilder(PipelineBuilder &&) noexcept = default;
  PipelineBuilder &operator=(PipelineBuilder &&) noexcept = default;

  PipelineBuilder &profile(const PipelineProfile profile) & noexcept {
    detail::configure_pipeline_profile(state_, profile);
    return *this;
  }

  PipelineBuilder &&profile(const PipelineProfile profile) && noexcept {
    static_cast<PipelineBuilder &>(*this).profile(profile);
    return std::move(*this);
  }

  template <std::size_t N>
    requires(N != 0u && N <= PipelineSealedRepetitionCapacity)
  PipelineBuilder &sealed_repetitions() & noexcept {
    detail::configure_pipeline_sealed_repetitions(state_, N);
    return *this;
  }

  template <std::size_t N>
    requires(N != 0u && N <= PipelineSealedRepetitionCapacity)
  PipelineBuilder &&sealed_repetitions() && noexcept {
    static_cast<PipelineBuilder &>(*this).template sealed_repetitions<N>();
    return std::move(*this);
  }

  PipelineBuilder &budget(const MemoryBudget limit) & noexcept {
    detail::configure_pipeline_budget(state_, limit);
    return *this;
  }

  PipelineBuilder &&budget(const MemoryBudget limit) && noexcept {
    static_cast<PipelineBuilder &>(*this).budget(limit);
    return std::move(*this);
  }

  [[nodiscard]] Result<PipelinePlan> plan() const noexcept {
    return detail::plan_pipeline(state_);
  }

  template <class T>
  PipelineBuilder &state(Buffer<T> &published, Buffer<T> &pending) & noexcept {
    detail::append_pipeline_state(
        state_, detail::BufferAccess::state(published),
        detail::BufferAccess::state(pending), detail::type<T>(),
        detail::storage_format<T>());
    return *this;
  }

  template <class T>
  PipelineBuilder &&state(Buffer<T> &published,
                          Buffer<T> &pending) && noexcept {
    static_cast<PipelineBuilder &>(*this).state(published, pending);
    return std::move(*this);
  }

  template <class Signature, class... I, class... O>
    requires(std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                            detail::TypeList<I...>> &&
             std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                            detail::TypeList<O...>>)
  PipelineBuilder &then(const Program<Signature> &program,
                        detail::ReadPack<I...> inputs,
                        detail::WritePack<O...> outputs) & noexcept {
    detail::append_pipeline(state_, detail::ProgramAccess::state(program),
                            inputs.views_, outputs.views_);
    return *this;
  }

  template <class Signature, class... I, class... O>
    requires(std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                            detail::TypeList<I...>> &&
             std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                            detail::TypeList<O...>>)
  PipelineBuilder &&then(const Program<Signature> &program,
                         detail::ReadPack<I...> inputs,
                         detail::WritePack<O...> outputs) && noexcept {
    static_cast<PipelineBuilder &>(*this).then(program, std::move(inputs),
                                               std::move(outputs));
    return std::move(*this);
  }

  template <std::size_t N, class Signature, class... I, class... O>
    requires(N != 0u &&
             std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                            detail::TypeList<I...>> &&
             std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                            detail::TypeList<O...>> &&
             detail::StartsWith<detail::TypeList<O...>,
                                detail::TypeList<I...>>::value)
  PipelineBuilder &repeat(const Program<Signature> &program,
                          detail::ReadPack<I...> inputs,
                          detail::WriteFinalPack<O...> outputs) & noexcept {
    detail::append_pipeline_repeat(state_,
                                   detail::ProgramAccess::state(program),
                                   inputs.views_, outputs.views_, N);
    return *this;
  }

  template <std::size_t N, class Signature, class... I, class... O>
    requires(N != 0u &&
             std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                            detail::TypeList<I...>> &&
             std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                            detail::TypeList<O...>> &&
             detail::StartsWith<detail::TypeList<O...>,
                                detail::TypeList<I...>>::value)
  PipelineBuilder &&repeat(const Program<Signature> &program,
                           detail::ReadPack<I...> inputs,
                           detail::WriteFinalPack<O...> outputs) && noexcept {
    static_cast<PipelineBuilder &>(*this).template repeat<N>(
        program, std::move(inputs), std::move(outputs));
    return std::move(*this);
  }

  template <std::size_t N, class Signature, class... I, class... O>
    requires(N != 0u &&
             std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                            detail::TypeList<I...>> &&
             std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                            detail::TypeList<O...>> &&
             detail::StartsWith<detail::TypeList<O...>,
                                detail::TypeList<I...>>::value)
  PipelineBuilder &repeat(const Program<Signature> &program,
                          detail::ReadPack<I...> inputs,
                          detail::WriteEachPack<O...> outputs) & noexcept {
    detail::append_pipeline_repeat_each(state_,
                                        detail::ProgramAccess::state(program),
                                        inputs.views_, outputs.views_, N);
    return *this;
  }

  template <std::size_t N, class Signature, class... I, class... O>
    requires(N != 0u &&
             std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                            detail::TypeList<I...>> &&
             std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                            detail::TypeList<O...>> &&
             detail::StartsWith<detail::TypeList<O...>,
                                detail::TypeList<I...>>::value)
  PipelineBuilder &&repeat(const Program<Signature> &program,
                           detail::ReadPack<I...> inputs,
                           detail::WriteEachPack<O...> outputs) && noexcept {
    static_cast<PipelineBuilder &>(*this).template repeat<N>(
        program, std::move(inputs), std::move(outputs));
    return std::move(*this);
  }

  template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
            class Signature, class... I, class... O>
    requires(
        Max != 0u && Tile != 0u && Tile <= Max &&
        (Max + Tile - 1u) / Tile <= PipelineIterationCapacity &&
        std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                       detail::TypeList<I..., std::uint32_t, std::uint32_t>> &&
        std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                       detail::TypeList<O...>> &&
        detail::StartsWith<detail::TypeList<O...>,
                           detail::TypeList<I...>>::value &&
        (Terminal == NoWindowTerminal ||
         (Terminal < sizeof...(O) &&
          detail::U32At<Terminal, detail::TypeList<O...>>::value)))
  PipelineBuilder &windows(const Program<Signature> &program,
                           const WindowInput<Terminal> &resident,
                           detail::ReadPack<I...> inputs,
                           detail::WriteFinalPack<O...> outputs) & noexcept {
    detail::append_pipeline_windows(
        state_, detail::ProgramAccess::state(program), resident.count_,
        inputs.views_, outputs.views_, Max, Tile, Terminal, resident.expected_);
    return *this;
  }

  template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
            class Signature, class... I, class... O>
    requires(
        Max != 0u && Tile != 0u && Tile <= Max &&
        (Max + Tile - 1u) / Tile <= PipelineIterationCapacity &&
        std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                       detail::TypeList<I..., std::uint32_t, std::uint32_t>> &&
        std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                       detail::TypeList<O...>> &&
        detail::StartsWith<detail::TypeList<O...>,
                           detail::TypeList<I...>>::value &&
        (Terminal == NoWindowTerminal ||
         (Terminal < sizeof...(O) &&
          detail::U32At<Terminal, detail::TypeList<O...>>::value)))
  PipelineBuilder &&windows(const Program<Signature> &program,
                            const WindowInput<Terminal> &resident,
                            detail::ReadPack<I...> inputs,
                            detail::WriteFinalPack<O...> outputs) && noexcept {
    static_cast<PipelineBuilder &>(*this).template windows<Max, Tile>(
        program, resident, std::move(inputs), std::move(outputs));
    return std::move(*this);
  }

  template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
            std::size_t N, class SeedSignature, class ActionSignature,
            class FoldSignature, class... I, class... O>
    requires detail::TileRepeatWindowBindingContract<
        Max, Tile, Terminal, N, SeedSignature, ActionSignature, FoldSignature,
        detail::TypeList<I...>, detail::TypeList<O...>,
        detail::TypeList<>>::valid
  PipelineBuilder &windows(
      const TileRepeat<N, SeedSignature, ActionSignature, FoldSignature> &body,
      const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
      detail::WriteFinalPack<O...> outputs) & noexcept {
    detail::append_pipeline_window_repeat(
        state_, detail::ProgramAccess::state(body.seed_),
        detail::ProgramAccess::state(body.action_),
        detail::ProgramAccess::state(body.fold_), resident.count_,
        inputs.views_, outputs.views_, std::span<const detail::ResourceView>{},
        Max, Tile, N, Terminal, resident.expected_);
    return *this;
  }

  template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
            std::size_t N, class SeedSignature, class ActionSignature,
            class FoldSignature, class... I, class... O, class... W>
    requires(sizeof...(W) != 0u &&
             detail::TileRepeatWindowBindingContract<
                 Max, Tile, Terminal, N, SeedSignature, ActionSignature,
                 FoldSignature, detail::TypeList<I...>, detail::TypeList<O...>,
                 detail::TypeList<W...>>::valid)
  PipelineBuilder &windows(
      const TileRepeat<N, SeedSignature, ActionSignature, FoldSignature> &body,
      const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
      detail::WriteFinalPack<O...> outputs,
      detail::WriteWindowPack<W...> window_outputs) & noexcept {
    detail::append_pipeline_window_repeat(
        state_, detail::ProgramAccess::state(body.seed_),
        detail::ProgramAccess::state(body.action_),
        detail::ProgramAccess::state(body.fold_), resident.count_,
        inputs.views_, outputs.views_, window_outputs.views_, Max, Tile, N,
        Terminal, resident.expected_);
    return *this;
  }

  template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
            std::size_t N, class SeedSignature, class ActionSignature,
            class FoldSignature, class... I, class... O>
    requires detail::TileRepeatWindowBindingContract<
        Max, Tile, Terminal, N, SeedSignature, ActionSignature, FoldSignature,
        detail::TypeList<I...>, detail::TypeList<O...>,
        detail::TypeList<>>::valid
  PipelineBuilder &&windows(
      const TileRepeat<N, SeedSignature, ActionSignature, FoldSignature> &body,
      const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
      detail::WriteFinalPack<O...> outputs) && noexcept {
    static_cast<PipelineBuilder &>(*this).template windows<Max, Tile>(
        body, resident, std::move(inputs), std::move(outputs));
    return std::move(*this);
  }

  template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
            std::size_t N, class SeedSignature, class ActionSignature,
            class FoldSignature, class... I, class... O, class... W>
    requires(sizeof...(W) != 0u &&
             detail::TileRepeatWindowBindingContract<
                 Max, Tile, Terminal, N, SeedSignature, ActionSignature,
                 FoldSignature, detail::TypeList<I...>, detail::TypeList<O...>,
                 detail::TypeList<W...>>::valid)
  PipelineBuilder &&windows(
      const TileRepeat<N, SeedSignature, ActionSignature, FoldSignature> &body,
      const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
      detail::WriteFinalPack<O...> outputs,
      detail::WriteWindowPack<W...> window_outputs) && noexcept {
    static_cast<PipelineBuilder &>(*this).template windows<Max, Tile>(
        body, resident, std::move(inputs), std::move(outputs),
        std::move(window_outputs));
    return std::move(*this);
  }

  template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
            class SeedSignature, class FoldSignature, class... I, class... O>
    requires detail::TileFoldWindowBindingContract<
        Max, Tile, Terminal, SeedSignature, FoldSignature,
        detail::TypeList<I...>, detail::TypeList<O...>,
        detail::TypeList<>>::valid
  PipelineBuilder &
  windows(const TileRepeat<0u, SeedSignature, void, FoldSignature> &body,
          const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
          detail::WriteFinalPack<O...> outputs) & noexcept {
    detail::append_pipeline_window_repeat(
        state_, detail::ProgramAccess::state(body.seed_), {},
        detail::ProgramAccess::state(body.fold_), resident.count_,
        inputs.views_, outputs.views_, std::span<const detail::ResourceView>{},
        Max, Tile, 0u, Terminal, resident.expected_);
    return *this;
  }

  template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
            class SeedSignature, class FoldSignature, class... I, class... O,
            class... W>
    requires(sizeof...(W) != 0u &&
             detail::TileFoldWindowBindingContract<
                 Max, Tile, Terminal, SeedSignature, FoldSignature,
                 detail::TypeList<I...>, detail::TypeList<O...>,
                 detail::TypeList<W...>>::valid)
  PipelineBuilder &
  windows(const TileRepeat<0u, SeedSignature, void, FoldSignature> &body,
          const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
          detail::WriteFinalPack<O...> outputs,
          detail::WriteWindowPack<W...> window_outputs) & noexcept {
    detail::append_pipeline_window_repeat(
        state_, detail::ProgramAccess::state(body.seed_), {},
        detail::ProgramAccess::state(body.fold_), resident.count_,
        inputs.views_, outputs.views_, window_outputs.views_, Max, Tile, 0u,
        Terminal, resident.expected_);
    return *this;
  }

  template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
            class SeedSignature, class FoldSignature, class... I, class... O>
    requires detail::TileFoldWindowBindingContract<
        Max, Tile, Terminal, SeedSignature, FoldSignature,
        detail::TypeList<I...>, detail::TypeList<O...>,
        detail::TypeList<>>::valid
  PipelineBuilder &&
  windows(const TileRepeat<0u, SeedSignature, void, FoldSignature> &body,
          const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
          detail::WriteFinalPack<O...> outputs) && noexcept {
    static_cast<PipelineBuilder &>(*this).template windows<Max, Tile>(
        body, resident, std::move(inputs), std::move(outputs));
    return std::move(*this);
  }

  template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
            class SeedSignature, class FoldSignature, class... I, class... O,
            class... W>
    requires(sizeof...(W) != 0u &&
             detail::TileFoldWindowBindingContract<
                 Max, Tile, Terminal, SeedSignature, FoldSignature,
                 detail::TypeList<I...>, detail::TypeList<O...>,
                 detail::TypeList<W...>>::valid)
  PipelineBuilder &&
  windows(const TileRepeat<0u, SeedSignature, void, FoldSignature> &body,
          const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
          detail::WriteFinalPack<O...> outputs,
          detail::WriteWindowPack<W...> window_outputs) && noexcept {
    static_cast<PipelineBuilder &>(*this).template windows<Max, Tile>(
        body, resident, std::move(inputs), std::move(outputs),
        std::move(window_outputs));
    return std::move(*this);
  }

  PipelineBuilder &restore(const StateSnapshot &snapshot) & noexcept {
    detail::seed_pipeline(state_, snapshot.state_);
    return *this;
  }

  PipelineBuilder &&restore(const StateSnapshot &snapshot) && noexcept {
    static_cast<PipelineBuilder &>(*this).restore(snapshot);
    return std::move(*this);
  }

  PipelineBuilder &restore(const LatestDeviceState &latest) & noexcept {
    detail::seed_pipeline(state_, latest.state_);
    return *this;
  }

  PipelineBuilder &&restore(const LatestDeviceState &latest) && noexcept {
    static_cast<PipelineBuilder &>(*this).restore(latest);
    return std::move(*this);
  }

  PipelineBuilder &restore(const SnapshotStorage &storage) & noexcept {
    detail::seed_pipeline(state_, storage.state_);
    return *this;
  }

  PipelineBuilder &&restore(const SnapshotStorage &storage) && noexcept {
    static_cast<PipelineBuilder &>(*this).restore(storage);
    return std::move(*this);
  }

  PipelineBuilder &commit() & noexcept {
    detail::commit_pipeline(state_);
    return *this;
  }

  PipelineBuilder &&commit() && noexcept {
    static_cast<PipelineBuilder &>(*this).commit();
    return std::move(*this);
  }

  [[nodiscard]] Result<Pipeline> prepare() && noexcept {
    auto prepared = detail::prepare_pipeline(std::move(state_));
    if (!prepared) {
      return Result<Pipeline>::fail(prepared.reason(), prepared.location());
    }
    return Result<Pipeline>::success(Pipeline{std::move(prepared).value()});
  }

private:
  friend PipelineBuilder pipeline(const Device &) noexcept;
  explicit PipelineBuilder(
      std::shared_ptr<detail::PipelineBuildState> state) noexcept
      : state_(std::move(state)) {}

  std::shared_ptr<detail::PipelineBuildState> state_{};
};

[[nodiscard]] inline PipelineBuilder pipeline(const Device &device) noexcept {
  return PipelineBuilder{
      detail::make_pipeline(detail::DeviceAccess::state(device))};
}

template <class Callback>
  requires(std::is_nothrow_invocable_r_v<Status, Callback &, HostIteration &>)
[[nodiscard]] Status host_feedback(Pipeline &pipeline,
                                   const std::size_t iterations,
                                   Callback &&callback) noexcept {
  if (!pipeline.valid()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t index = 0u; index < iterations; ++index) {
    const Status ran = pipeline.run();
    if (!ran) {
      return ran;
    }
    auto iteration =
        detail::HostFeedbackAccess::iteration(pipeline, index + 1u, iterations);
    const Status feedback = std::invoke(callback, iteration);
    if (!feedback) {
      return feedback;
    }
  }
  return Status::success();
}

} // namespace rund::compute
