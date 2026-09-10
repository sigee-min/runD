#pragma once

#include <rund/compute/pipeline/access.hpp>
#include <rund/compute/virtual/buffer.hpp>
#include <rund/compute/virtual/pipeline.hpp>

#include <array>

namespace rund::compute {
namespace detail {

template <class Signature, ComputeValue R, std::size_t N>
[[nodiscard]] Result<VirtualPipeline<Signature>>
prepare_public_virtual_pipeline(
    const Program<Signature> &program,
    const std::array<std::shared_ptr<VirtualBufferState>, N> &inputs,
    VirtualBuffer<R> &output, const ResidencyConfig config) noexcept {
  auto prepared =
      prepare_virtual_pipeline(ProgramAccess::state(program), inputs,
                               VirtualBufferAccess::state(output), config);
  if (!prepared) {
    return Result<VirtualPipeline<Signature>>::fail(prepared.reason(),
                                                    prepared.location());
  }
  return Result<VirtualPipeline<Signature>>::success(
      VirtualPipelineAccess::make<Signature>(std::move(prepared).value()));
}

template <class Signature, ComputeValue R, std::size_t N>
[[nodiscard]] Result<VirtualPipeline<Signature>>
prepare_public_virtual_pipeline(
    const Program<Signature> &program,
    const std::array<std::shared_ptr<VirtualBufferState>, N> &inputs,
    VirtualBuffer<R> &output, const ResidencyConfig config,
    const GraphPageMap page_map) noexcept {
  auto prepared = prepare_virtual_pipeline(
      ProgramAccess::state(program), inputs, VirtualBufferAccess::state(output),
      config, page_map);
  if (!prepared) {
    return Result<VirtualPipeline<Signature>>::fail(prepared.reason(),
                                                    prepared.location());
  }
  return Result<VirtualPipeline<Signature>>::success(
      VirtualPipelineAccess::make<Signature>(std::move(prepared).value()));
}

} // namespace detail

template <detail::ComputeValue R, detail::ComputeValue A>
[[nodiscard]] Result<VirtualPipeline<R(A)>>
virtual_pipeline(const Program<R(A)> &program, const VirtualBuffer<A> &input,
                 VirtualBuffer<R> &output,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{detail::VirtualBufferAccess::state(input)};
  return detail::prepare_public_virtual_pipeline<R(A)>(program, inputs, output,
                                                       config);
}

template <detail::ComputeValue R, detail::ComputeValue A>
[[nodiscard]] Result<VirtualPipeline<R(A)>>
virtual_pipeline(const Program<R(A)> &program, const VirtualBuffer<A> &input,
                 VirtualBuffer<R> &output, const GraphPageMap page_map,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{detail::VirtualBufferAccess::state(input)};
  return detail::prepare_public_virtual_pipeline<R(A)>(program, inputs, output,
                                                       config, page_map);
}

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B>
[[nodiscard]] Result<VirtualPipeline<R(A, B)>>
virtual_pipeline(const Program<R(A, B)> &program, const VirtualBuffer<A> &a,
                 const VirtualBuffer<B> &b, VirtualBuffer<R> &output,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{detail::VirtualBufferAccess::state(a),
                          detail::VirtualBufferAccess::state(b)};
  return detail::prepare_public_virtual_pipeline<R(A, B)>(program, inputs,
                                                          output, config);
}

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B>
[[nodiscard]] Result<VirtualPipeline<R(A, B)>>
virtual_pipeline(const Program<R(A, B)> &program, const VirtualBuffer<A> &a,
                 const VirtualBuffer<B> &b, VirtualBuffer<R> &output,
                 const GraphPageMap page_map,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{detail::VirtualBufferAccess::state(a),
                          detail::VirtualBufferAccess::state(b)};
  return detail::prepare_public_virtual_pipeline<R(A, B)>(
      program, inputs, output, config, page_map);
}

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B, detail::ComputeValue C>
[[nodiscard]] Result<VirtualPipeline<R(A, B, C)>>
virtual_pipeline(const Program<R(A, B, C)> &program, const VirtualBuffer<A> &a,
                 const VirtualBuffer<B> &b, const VirtualBuffer<C> &c,
                 VirtualBuffer<R> &output,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{detail::VirtualBufferAccess::state(a),
                          detail::VirtualBufferAccess::state(b),
                          detail::VirtualBufferAccess::state(c)};
  return detail::prepare_public_virtual_pipeline<R(A, B, C)>(program, inputs,
                                                             output, config);
}

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B, detail::ComputeValue C>
[[nodiscard]] Result<VirtualPipeline<R(A, B, C)>>
virtual_pipeline(const Program<R(A, B, C)> &program, const VirtualBuffer<A> &a,
                 const VirtualBuffer<B> &b, const VirtualBuffer<C> &c,
                 VirtualBuffer<R> &output, const GraphPageMap page_map,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{detail::VirtualBufferAccess::state(a),
                          detail::VirtualBufferAccess::state(b),
                          detail::VirtualBufferAccess::state(c)};
  return detail::prepare_public_virtual_pipeline<R(A, B, C)>(
      program, inputs, output, config, page_map);
}

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B, detail::ComputeValue C,
          detail::ComputeValue D>
[[nodiscard]] Result<VirtualPipeline<R(A, B, C, D)>>
virtual_pipeline(const Program<R(A, B, C, D)> &program,
                 const VirtualBuffer<A> &a, const VirtualBuffer<B> &b,
                 const VirtualBuffer<C> &c, const VirtualBuffer<D> &d,
                 VirtualBuffer<R> &output,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{
      detail::VirtualBufferAccess::state(a),
      detail::VirtualBufferAccess::state(b),
      detail::VirtualBufferAccess::state(c),
      detail::VirtualBufferAccess::state(d),
  };
  return detail::prepare_public_virtual_pipeline<R(A, B, C, D)>(program, inputs,
                                                                output, config);
}

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B, detail::ComputeValue C,
          detail::ComputeValue D>
[[nodiscard]] Result<VirtualPipeline<R(A, B, C, D)>>
virtual_pipeline(const Program<R(A, B, C, D)> &program,
                 const VirtualBuffer<A> &a, const VirtualBuffer<B> &b,
                 const VirtualBuffer<C> &c, const VirtualBuffer<D> &d,
                 VirtualBuffer<R> &output, const GraphPageMap page_map,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{
      detail::VirtualBufferAccess::state(a),
      detail::VirtualBufferAccess::state(b),
      detail::VirtualBufferAccess::state(c),
      detail::VirtualBufferAccess::state(d),
  };
  return detail::prepare_public_virtual_pipeline<R(A, B, C, D)>(
      program, inputs, output, config, page_map);
}

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B, detail::ComputeValue C,
          detail::ComputeValue D, detail::ComputeValue E>
[[nodiscard]] Result<VirtualPipeline<R(A, B, C, D, E)>>
virtual_pipeline(const Program<R(A, B, C, D, E)> &program,
                 const VirtualBuffer<A> &a, const VirtualBuffer<B> &b,
                 const VirtualBuffer<C> &c, const VirtualBuffer<D> &d,
                 const VirtualBuffer<E> &e, VirtualBuffer<R> &output,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{
      detail::VirtualBufferAccess::state(a),
      detail::VirtualBufferAccess::state(b),
      detail::VirtualBufferAccess::state(c),
      detail::VirtualBufferAccess::state(d),
      detail::VirtualBufferAccess::state(e),
  };
  return detail::prepare_public_virtual_pipeline<R(A, B, C, D, E)>(
      program, inputs, output, config);
}

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B, detail::ComputeValue C,
          detail::ComputeValue D, detail::ComputeValue E>
[[nodiscard]] Result<VirtualPipeline<R(A, B, C, D, E)>>
virtual_pipeline(const Program<R(A, B, C, D, E)> &program,
                 const VirtualBuffer<A> &a, const VirtualBuffer<B> &b,
                 const VirtualBuffer<C> &c, const VirtualBuffer<D> &d,
                 const VirtualBuffer<E> &e, VirtualBuffer<R> &output,
                 const GraphPageMap page_map,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{
      detail::VirtualBufferAccess::state(a),
      detail::VirtualBufferAccess::state(b),
      detail::VirtualBufferAccess::state(c),
      detail::VirtualBufferAccess::state(d),
      detail::VirtualBufferAccess::state(e),
  };
  return detail::prepare_public_virtual_pipeline<R(A, B, C, D, E)>(
      program, inputs, output, config, page_map);
}

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B, detail::ComputeValue C,
          detail::ComputeValue D, detail::ComputeValue E,
          detail::ComputeValue F>
[[nodiscard]] Result<VirtualPipeline<R(A, B, C, D, E, F)>>
virtual_pipeline(const Program<R(A, B, C, D, E, F)> &program,
                 const VirtualBuffer<A> &a, const VirtualBuffer<B> &b,
                 const VirtualBuffer<C> &c, const VirtualBuffer<D> &d,
                 const VirtualBuffer<E> &e, const VirtualBuffer<F> &f,
                 VirtualBuffer<R> &output,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{
      detail::VirtualBufferAccess::state(a),
      detail::VirtualBufferAccess::state(b),
      detail::VirtualBufferAccess::state(c),
      detail::VirtualBufferAccess::state(d),
      detail::VirtualBufferAccess::state(e),
      detail::VirtualBufferAccess::state(f),
  };
  return detail::prepare_public_virtual_pipeline<R(A, B, C, D, E, F)>(
      program, inputs, output, config);
}

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B, detail::ComputeValue C,
          detail::ComputeValue D, detail::ComputeValue E,
          detail::ComputeValue F>
[[nodiscard]] Result<VirtualPipeline<R(A, B, C, D, E, F)>>
virtual_pipeline(const Program<R(A, B, C, D, E, F)> &program,
                 const VirtualBuffer<A> &a, const VirtualBuffer<B> &b,
                 const VirtualBuffer<C> &c, const VirtualBuffer<D> &d,
                 const VirtualBuffer<E> &e, const VirtualBuffer<F> &f,
                 VirtualBuffer<R> &output, const GraphPageMap page_map,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{
      detail::VirtualBufferAccess::state(a),
      detail::VirtualBufferAccess::state(b),
      detail::VirtualBufferAccess::state(c),
      detail::VirtualBufferAccess::state(d),
      detail::VirtualBufferAccess::state(e),
      detail::VirtualBufferAccess::state(f),
  };
  return detail::prepare_public_virtual_pipeline<R(A, B, C, D, E, F)>(
      program, inputs, output, config, page_map);
}

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B, detail::ComputeValue C,
          detail::ComputeValue D, detail::ComputeValue E,
          detail::ComputeValue F, detail::ComputeValue G>
[[nodiscard]] Result<VirtualPipeline<R(A, B, C, D, E, F, G)>>
virtual_pipeline(const Program<R(A, B, C, D, E, F, G)> &program,
                 const VirtualBuffer<A> &a, const VirtualBuffer<B> &b,
                 const VirtualBuffer<C> &c, const VirtualBuffer<D> &d,
                 const VirtualBuffer<E> &e, const VirtualBuffer<F> &f,
                 const VirtualBuffer<G> &g, VirtualBuffer<R> &output,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{
      detail::VirtualBufferAccess::state(a),
      detail::VirtualBufferAccess::state(b),
      detail::VirtualBufferAccess::state(c),
      detail::VirtualBufferAccess::state(d),
      detail::VirtualBufferAccess::state(e),
      detail::VirtualBufferAccess::state(f),
      detail::VirtualBufferAccess::state(g),
  };
  return detail::prepare_public_virtual_pipeline<R(A, B, C, D, E, F, G)>(
      program, inputs, output, config);
}

template <detail::ComputeValue R, detail::ComputeValue A,
          detail::ComputeValue B, detail::ComputeValue C,
          detail::ComputeValue D, detail::ComputeValue E,
          detail::ComputeValue F, detail::ComputeValue G>
[[nodiscard]] Result<VirtualPipeline<R(A, B, C, D, E, F, G)>>
virtual_pipeline(const Program<R(A, B, C, D, E, F, G)> &program,
                 const VirtualBuffer<A> &a, const VirtualBuffer<B> &b,
                 const VirtualBuffer<C> &c, const VirtualBuffer<D> &d,
                 const VirtualBuffer<E> &e, const VirtualBuffer<F> &f,
                 const VirtualBuffer<G> &g, VirtualBuffer<R> &output,
                 const GraphPageMap page_map,
                 const ResidencyConfig config = {}) noexcept {
  const std::array inputs{
      detail::VirtualBufferAccess::state(a),
      detail::VirtualBufferAccess::state(b),
      detail::VirtualBufferAccess::state(c),
      detail::VirtualBufferAccess::state(d),
      detail::VirtualBufferAccess::state(e),
      detail::VirtualBufferAccess::state(f),
      detail::VirtualBufferAccess::state(g),
  };
  return detail::prepare_public_virtual_pipeline<R(A, B, C, D, E, F, G)>(
      program, inputs, output, config, page_map);
}

} // namespace rund::compute
