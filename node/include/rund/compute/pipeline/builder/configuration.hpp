#pragma once

namespace rund::compute {

template <std::size_t N>
  requires(N != 0u && N <= PipelineSealedRepetitionCapacity)
PipelineBuilder &PipelineBuilder::sealed_repetitions() & noexcept {
  detail::configure_pipeline_sealed_repetitions(state_, N);
  return *this;
}

template <std::size_t N>
  requires(N != 0u && N <= PipelineSealedRepetitionCapacity)
PipelineBuilder &&PipelineBuilder::sealed_repetitions() && noexcept {
  static_cast<PipelineBuilder &>(*this).template sealed_repetitions<N>();
  return std::move(*this);
}

template <class T>
PipelineBuilder &PipelineBuilder::state(Buffer<T> &published,
                                        Buffer<T> &pending) & noexcept {
  detail::append_pipeline_state(state_, detail::BufferAccess::state(published),
                                detail::BufferAccess::state(pending),
                                detail::type<T>(), detail::storage_format<T>());
  return *this;
}

template <class T>
PipelineBuilder &&PipelineBuilder::state(Buffer<T> &published,
                                         Buffer<T> &pending) && noexcept {
  static_cast<PipelineBuilder &>(*this).state(published, pending);
  return std::move(*this);
}

template <class Signature, class... I, class... O>
  requires(std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                          detail::TypeList<I...>> &&
           std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                          detail::TypeList<O...>>)
PipelineBuilder &
PipelineBuilder::then(const Program<Signature> &program,
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
PipelineBuilder &&
PipelineBuilder::then(const Program<Signature> &program,
                      detail::ReadPack<I...> inputs,
                      detail::WritePack<O...> outputs) && noexcept {
  static_cast<PipelineBuilder &>(*this).then(program, std::move(inputs),
                                             std::move(outputs));
  return std::move(*this);
}

template <std::size_t N, class Signature, class... I, class... O>
  requires(
      N != 0u &&
      std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                     detail::TypeList<I...>> &&
      std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                     detail::TypeList<O...>> &&
      detail::StartsWith<detail::TypeList<O...>, detail::TypeList<I...>>::value)
PipelineBuilder &
PipelineBuilder::repeat(const Program<Signature> &program,
                        detail::ReadPack<I...> inputs,
                        detail::WriteFinalPack<O...> outputs) & noexcept {
  detail::append_pipeline_repeat(state_, detail::ProgramAccess::state(program),
                                 inputs.views_, outputs.views_, N);
  return *this;
}

template <std::size_t N, class Signature, class... I, class... O>
  requires(
      N != 0u &&
      std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                     detail::TypeList<I...>> &&
      std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                     detail::TypeList<O...>> &&
      detail::StartsWith<detail::TypeList<O...>, detail::TypeList<I...>>::value)
PipelineBuilder &&
PipelineBuilder::repeat(const Program<Signature> &program,
                        detail::ReadPack<I...> inputs,
                        detail::WriteFinalPack<O...> outputs) && noexcept {
  static_cast<PipelineBuilder &>(*this).template repeat<N>(
      program, std::move(inputs), std::move(outputs));
  return std::move(*this);
}

template <std::size_t N, class Signature, class... I, class... O>
  requires(
      N != 0u &&
      std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                     detail::TypeList<I...>> &&
      std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                     detail::TypeList<O...>> &&
      detail::StartsWith<detail::TypeList<O...>, detail::TypeList<I...>>::value)
PipelineBuilder &
PipelineBuilder::repeat(const Program<Signature> &program,
                        detail::ReadPack<I...> inputs,
                        detail::WriteEachPack<O...> outputs) & noexcept {
  detail::append_pipeline_repeat_each(state_,
                                      detail::ProgramAccess::state(program),
                                      inputs.views_, outputs.views_, N);
  return *this;
}

template <std::size_t N, class Signature, class... I, class... O>
  requires(
      N != 0u &&
      std::is_same_v<typename detail::SignatureTypes<Signature>::Inputs,
                     detail::TypeList<I...>> &&
      std::is_same_v<typename detail::SignatureTypes<Signature>::Outputs,
                     detail::TypeList<O...>> &&
      detail::StartsWith<detail::TypeList<O...>, detail::TypeList<I...>>::value)
PipelineBuilder &&
PipelineBuilder::repeat(const Program<Signature> &program,
                        detail::ReadPack<I...> inputs,
                        detail::WriteEachPack<O...> outputs) && noexcept {
  static_cast<PipelineBuilder &>(*this).template repeat<N>(
      program, std::move(inputs), std::move(outputs));
  return std::move(*this);
}

} // namespace rund::compute
