#pragma once

#include <rund/compute/pipeline/builder/base.hpp>

namespace rund::compute {

class PipelineBuilder final {
public:
  PipelineBuilder(const PipelineBuilder &) = delete;
  PipelineBuilder &operator=(const PipelineBuilder &) = delete;
  PipelineBuilder(PipelineBuilder &&) noexcept = default;
  PipelineBuilder &operator=(PipelineBuilder &&) noexcept = default;

  PipelineBuilder &profile(const PipelineProfile profile) & noexcept;

  PipelineBuilder &&profile(const PipelineProfile profile) && noexcept;

  template <std::size_t N>
    requires(N != 0u && N <= PipelineSealedRepetitionCapacity)
  PipelineBuilder &sealed_repetitions() & noexcept;

  template <std::size_t N>
    requires(N != 0u && N <= PipelineSealedRepetitionCapacity)
  PipelineBuilder &&sealed_repetitions() && noexcept;

  PipelineBuilder &budget(const MemoryBudget limit) & noexcept;

  PipelineBuilder &&budget(const MemoryBudget limit) && noexcept;

  [[nodiscard]] Result<PipelinePlan> plan() const noexcept;

  template <class T>
  PipelineBuilder &state(Buffer<T> &published, Buffer<T> &pending) & noexcept;

  template <class T>
  PipelineBuilder &&state(Buffer<T> &published, Buffer<T> &pending) && noexcept;

  template <class Signature, class... I, class... O>
    requires(std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                            detail::TypeList<I...>> &&
             std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                            detail::TypeList<O...>>)
  PipelineBuilder &then(const Program<Signature> &program,
                        detail::ReadPack<I...> inputs,
                        detail::WritePack<O...> outputs) & noexcept;

  template <class Signature, class... I, class... O>
    requires(std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                            detail::TypeList<I...>> &&
             std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                            detail::TypeList<O...>>)
  PipelineBuilder &&then(const Program<Signature> &program,
                         detail::ReadPack<I...> inputs,
                         detail::WritePack<O...> outputs) && noexcept;

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
                          detail::WriteFinalPack<O...> outputs) & noexcept;

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
                           detail::WriteFinalPack<O...> outputs) && noexcept;

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
                          detail::WriteEachPack<O...> outputs) & noexcept;

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
                           detail::WriteEachPack<O...> outputs) && noexcept;

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
                           detail::WriteFinalPack<O...> outputs) & noexcept;

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
                            detail::WriteFinalPack<O...> outputs) && noexcept;

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
      detail::WriteFinalPack<O...> outputs) & noexcept;

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
      detail::WriteWindowPack<W...> window_outputs) & noexcept;

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
      detail::WriteFinalPack<O...> outputs) && noexcept;

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
      detail::WriteWindowPack<W...> window_outputs) && noexcept;

  template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
            class SeedSignature, class FoldSignature, class... I, class... O>
    requires detail::TileFoldWindowBindingContract<
        Max, Tile, Terminal, SeedSignature, FoldSignature,
        detail::TypeList<I...>, detail::TypeList<O...>,
        detail::TypeList<>>::valid
  PipelineBuilder &
  windows(const TileRepeat<0u, SeedSignature, void, FoldSignature> &body,
          const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
          detail::WriteFinalPack<O...> outputs) & noexcept;

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
          detail::WriteWindowPack<W...> window_outputs) & noexcept;

  template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
            class SeedSignature, class FoldSignature, class... I, class... O>
    requires detail::TileFoldWindowBindingContract<
        Max, Tile, Terminal, SeedSignature, FoldSignature,
        detail::TypeList<I...>, detail::TypeList<O...>,
        detail::TypeList<>>::valid
  PipelineBuilder &&
  windows(const TileRepeat<0u, SeedSignature, void, FoldSignature> &body,
          const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
          detail::WriteFinalPack<O...> outputs) && noexcept;

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
          detail::WriteWindowPack<W...> window_outputs) && noexcept;

  PipelineBuilder &restore(const StateSnapshot &snapshot) & noexcept;

  PipelineBuilder &&restore(const StateSnapshot &snapshot) && noexcept;

  PipelineBuilder &restore(const LatestDeviceState &latest) & noexcept;

  PipelineBuilder &&restore(const LatestDeviceState &latest) && noexcept;

  PipelineBuilder &restore(const SnapshotStorage &storage) & noexcept;

  PipelineBuilder &&restore(const SnapshotStorage &storage) && noexcept;

  PipelineBuilder &commit() & noexcept;

  PipelineBuilder &&commit() && noexcept;

  [[nodiscard]] Result<Pipeline> prepare() && noexcept;

private:
  friend PipelineBuilder pipeline(const Device &) noexcept;
  explicit PipelineBuilder(
      std::shared_ptr<detail::PipelineBuildState> state) noexcept
      : state_(std::move(state)) {}

  std::shared_ptr<detail::PipelineBuildState> state_{};
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

#include <rund/compute/pipeline/builder/configuration.hpp>
#include <rund/compute/pipeline/builder/windows.hpp>
