#pragma once

namespace rund::compute {

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
PipelineBuilder &
PipelineBuilder::windows(const Program<Signature> &program,
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
PipelineBuilder &&
PipelineBuilder::windows(const Program<Signature> &program,
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
      detail::TypeList<I...>, detail::TypeList<O...>, detail::TypeList<>>::valid
PipelineBuilder &PipelineBuilder::windows(
    const TileRepeat<N, SeedSignature, ActionSignature, FoldSignature> &body,
    const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
    detail::WriteFinalPack<O...> outputs) & noexcept {
  detail::append_pipeline_window_repeat(
      state_, detail::ProgramAccess::state(body.seed_),
      detail::ProgramAccess::state(body.action_),
      detail::ProgramAccess::state(body.fold_), resident.count_, inputs.views_,
      outputs.views_, std::span<const detail::ResourceView>{}, Max, Tile, N,
      Terminal, resident.expected_);
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
PipelineBuilder &PipelineBuilder::windows(
    const TileRepeat<N, SeedSignature, ActionSignature, FoldSignature> &body,
    const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
    detail::WriteFinalPack<O...> outputs,
    detail::WriteWindowPack<W...> window_outputs) & noexcept {
  detail::append_pipeline_window_repeat(
      state_, detail::ProgramAccess::state(body.seed_),
      detail::ProgramAccess::state(body.action_),
      detail::ProgramAccess::state(body.fold_), resident.count_, inputs.views_,
      outputs.views_, window_outputs.views_, Max, Tile, N, Terminal,
      resident.expected_);
  return *this;
}

template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
          std::size_t N, class SeedSignature, class ActionSignature,
          class FoldSignature, class... I, class... O>
  requires detail::TileRepeatWindowBindingContract<
      Max, Tile, Terminal, N, SeedSignature, ActionSignature, FoldSignature,
      detail::TypeList<I...>, detail::TypeList<O...>, detail::TypeList<>>::valid
PipelineBuilder &&PipelineBuilder::windows(
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
PipelineBuilder &&PipelineBuilder::windows(
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
      Max, Tile, Terminal, SeedSignature, FoldSignature, detail::TypeList<I...>,
      detail::TypeList<O...>, detail::TypeList<>>::valid
PipelineBuilder &PipelineBuilder::windows(
    const TileRepeat<0u, SeedSignature, void, FoldSignature> &body,
    const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
    detail::WriteFinalPack<O...> outputs) & noexcept {
  detail::append_pipeline_window_repeat(
      state_, detail::ProgramAccess::state(body.seed_), {},
      detail::ProgramAccess::state(body.fold_), resident.count_, inputs.views_,
      outputs.views_, std::span<const detail::ResourceView>{}, Max, Tile, 0u,
      Terminal, resident.expected_);
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
PipelineBuilder &PipelineBuilder::windows(
    const TileRepeat<0u, SeedSignature, void, FoldSignature> &body,
    const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
    detail::WriteFinalPack<O...> outputs,
    detail::WriteWindowPack<W...> window_outputs) & noexcept {
  detail::append_pipeline_window_repeat(
      state_, detail::ProgramAccess::state(body.seed_), {},
      detail::ProgramAccess::state(body.fold_), resident.count_, inputs.views_,
      outputs.views_, window_outputs.views_, Max, Tile, 0u, Terminal,
      resident.expected_);
  return *this;
}

template <std::size_t Max, std::size_t Tile, std::size_t Terminal,
          class SeedSignature, class FoldSignature, class... I, class... O>
  requires detail::TileFoldWindowBindingContract<
      Max, Tile, Terminal, SeedSignature, FoldSignature, detail::TypeList<I...>,
      detail::TypeList<O...>, detail::TypeList<>>::valid
PipelineBuilder &&PipelineBuilder::windows(
    const TileRepeat<0u, SeedSignature, void, FoldSignature> &body,
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
PipelineBuilder &&PipelineBuilder::windows(
    const TileRepeat<0u, SeedSignature, void, FoldSignature> &body,
    const WindowInput<Terminal> &resident, detail::ReadPack<I...> inputs,
    detail::WriteFinalPack<O...> outputs,
    detail::WriteWindowPack<W...> window_outputs) && noexcept {
  static_cast<PipelineBuilder &>(*this).template windows<Max, Tile>(
      body, resident, std::move(inputs), std::move(outputs),
      std::move(window_outputs));
  return std::move(*this);
}

} // namespace rund::compute
