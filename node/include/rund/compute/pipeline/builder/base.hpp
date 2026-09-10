#pragma once

#include <rund/compute/pipeline/access.hpp>
#include <rund/compute/pipeline/runtime.hpp>

#include <rund/compute/device.hpp>
#include <rund/compute/pipeline/capacity.hpp>
#include <rund/compute/pipeline/shape.hpp>
#include <rund/compute/pipeline/tile.hpp>
#include <rund/compute/pipeline/window.hpp>
#include <rund/compute/program.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

namespace rund::compute {

namespace detail {

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
void configure_pipeline_budget(const std::shared_ptr<PipelineBuildState> &build,
                               MemoryBudget budget) noexcept;

} // namespace detail

} // namespace rund::compute
