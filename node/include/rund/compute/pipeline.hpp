#pragma once

#include <rund/compute/device.hpp>
#include <rund/compute/graph/info.hpp>
#include <rund/compute/pipeline/bind.hpp>
#include <rund/compute/pipeline/memory.hpp>
#include <rund/compute/pipeline/profile.hpp>
#include <rund/compute/pipeline/shape.hpp>
#include <rund/compute/pipeline/window.hpp>
#include <rund/compute/program.hpp>
#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>

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
class StateSnapshot;

namespace detail {

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
void append_pipeline_windows(const std::shared_ptr<PipelineBuildState> &build,
                             const std::shared_ptr<ProgramState> &program,
                             const ResourceView &resident,
                             std::span<const ResourceView> inputs,
                             std::span<const ResourceView> outputs,
                             std::size_t maximum, std::size_t tile,
                             std::size_t terminal,
                             std::uint32_t expected) noexcept;
void append_pipeline_state(const std::shared_ptr<PipelineBuildState> &build,
                           const std::shared_ptr<BufferState> &published,
                           const std::shared_ptr<BufferState> &pending,
                           Type type, FixedFormat format) noexcept;
void configure_pipeline_profile(
    const std::shared_ptr<PipelineBuildState> &build,
    PipelineProfile profile) noexcept;
void commit_pipeline(const std::shared_ptr<PipelineBuildState> &build) noexcept;
void seed_pipeline(
    const std::shared_ptr<PipelineBuildState> &build,
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept;
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
[[nodiscard]] Status restore_pipeline_state(
    const std::shared_ptr<PipelineState> &state,
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept;
[[nodiscard]] bool
snapshot_valid(const std::shared_ptr<StateSnapshotState> &snapshot) noexcept;
[[nodiscard]] std::uint64_t snapshot_generation(
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept;
[[nodiscard]] graph::Fingerprint snapshot_fingerprint(
    const std::shared_ptr<StateSnapshotState> &snapshot) noexcept;
[[nodiscard]] std::uint64_t
snapshot_hash(const std::shared_ptr<StateSnapshotState> &snapshot) noexcept;
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
                          detail::WritePack<O...> outputs) & noexcept {
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
                           detail::WritePack<O...> outputs) && noexcept {
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
                           detail::WritePack<O...> outputs) & noexcept {
    detail::append_pipeline_windows(
        state_, detail::ProgramAccess::state(program), resident.count_,
        inputs.views_, outputs.views_, Max, Tile, Terminal,
        resident.expected_);
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
                            detail::WritePack<O...> outputs) && noexcept {
    static_cast<PipelineBuilder &>(*this).template windows<Max, Tile>(
        program, resident, std::move(inputs), std::move(outputs));
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

} // namespace rund::compute
