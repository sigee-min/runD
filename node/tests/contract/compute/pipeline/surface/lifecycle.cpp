#include "../local.hpp"

#include "../../../../../src/compute/pipeline/claim.hpp"
#include "../../../../../src/compute/pipeline/state.hpp"

#include <concepts>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace rund_node_test_pipeline {

template <class T>
concept ReadsTemporary =
    requires(T value) { rund::compute::read(std::move(value)); };

template <class T>
concept WritesConst = requires(const T value) { rund::compute::write(value); };

template <class T>
concept WritesTemporary =
    requires(T value) { rund::compute::write(std::move(value)); };

template <class T>
concept WritesFinalConst =
    requires(const T value) { rund::compute::write_final(value); };

template <class T>
concept WritesFinalTemporary =
    requires(T value) { rund::compute::write_final(std::move(value)); };

template <class T>
concept WritesWindowConst =
    requires(const T value) { rund::compute::write_window(value); };

template <class T>
concept WritesWindowTemporary =
    requires(T value) { rund::compute::write_window(std::move(value)); };

template <class T>
concept WritesEachConst =
    requires(const T value) { rund::compute::write_each(value); };

template <class T>
concept WritesEachTemporary =
    requires(T value) { rund::compute::write_each(std::move(value)); };

template <class T>
concept ReadsValue = requires(T value) { rund::compute::read(value); };

template <class T>
concept WritesValue = requires(T value) { rund::compute::write(value); };

template <class T>
concept WritesFinalValue =
    requires(T value) { rund::compute::write_final(value); };

template <class T>
concept WritesWindowValue =
    requires(T value) { rund::compute::write_window(value); };

template <class T>
concept WritesEachValue =
    requires(T value) { rund::compute::write_each(value); };

template <class T>
concept ReadsVolatile =
    requires(volatile T &value) { rund::compute::read(value); };

template <class T>
concept WritesVolatile =
    requires(volatile T &value) { rund::compute::write(value); };

template <class T>
concept WritesFinalVolatile =
    requires(volatile T &value) { rund::compute::write_final(value); };

template <class T>
concept WritesWindowVolatile =
    requires(volatile T &value) { rund::compute::write_window(value); };

template <class T>
concept WritesEachVolatile =
    requires(volatile T &value) { rund::compute::write_each(value); };

template <class T>
concept HasDependsOnSurface = requires(T value) { value.depends_on(0u, 1u); };

template <class T>
concept HasAfterSurface = requires(T value) { value.after(0u, 1u); };

template <class T>
concept PreparesLvalue = requires(T value) { value.prepare(); };

template <class T>
concept PreparesRvalue = requires(T value) { std::move(value).prepare(); };

template <class T>
concept HasRunSurface = requires(T value) { value.run(); };

template <std::size_t N, class Builder>
concept CanSealRepetitions = requires(Builder builder) {
  builder.template sealed_repetitions<N>();
  std::move(builder).template sealed_repetitions<N>();
};

template <class T>
concept RetainsColdPipelineBindings = requires(T value) {
  value.inputs;
  value.outputs;
  value.input_bindings;
  value.output_bindings;
};

using ReadPack =
    decltype(rund::compute::read(std::declval<Buffer<std::int32_t> &>()));
using WritePack =
    decltype(rund::compute::write(std::declval<Buffer<std::int32_t> &>()));
using WriteFinalPack = decltype(rund::compute::write_final(
    std::declval<Buffer<std::int32_t> &>()));
using WriteWindowPack = decltype(rund::compute::write_window(
    std::declval<Buffer<std::int32_t> &>()));
using WriteEachPack = decltype(
    rund::compute::write_each(std::declval<Buffer<std::int32_t> &>()));

static_assert(!std::is_copy_constructible_v<Pipeline>);
static_assert(std::is_nothrow_move_constructible_v<Pipeline>);
static_assert(!std::is_copy_constructible_v<PipelineBuilder>);
static_assert(std::is_nothrow_move_constructible_v<PipelineBuilder>);
static_assert(!std::is_copy_constructible_v<rund::compute::HostIteration>);
static_assert(!std::is_move_constructible_v<rund::compute::HostIteration>);
static_assert(std::is_nothrow_copy_constructible_v<StateSnapshot>);
static_assert(std::is_nothrow_copy_assignable_v<StateSnapshot>);
static_assert(
    std::is_nothrow_copy_constructible_v<rund::compute::LatestDeviceState>);
static_assert(
    std::is_nothrow_copy_assignable_v<rund::compute::LatestDeviceState>);
static_assert(!std::is_copy_constructible_v<rund::compute::SnapshotStorage>);
static_assert(
    std::is_nothrow_move_constructible_v<rund::compute::SnapshotStorage>);
static_assert(!std::is_copy_constructible_v<ReadPack>);
static_assert(std::is_nothrow_move_constructible_v<ReadPack>);
static_assert(!std::is_copy_constructible_v<WritePack>);
static_assert(std::is_nothrow_move_constructible_v<WritePack>);
static_assert(!std::is_copy_constructible_v<WriteFinalPack>);
static_assert(std::is_nothrow_move_constructible_v<WriteFinalPack>);
static_assert(!std::is_copy_constructible_v<WriteWindowPack>);
static_assert(std::is_nothrow_move_constructible_v<WriteWindowPack>);
static_assert(!std::is_copy_constructible_v<WriteEachPack>);
static_assert(std::is_nothrow_move_constructible_v<WriteEachPack>);
static_assert(!ReadsTemporary<Buffer<std::int32_t>>);
static_assert(!WritesConst<Buffer<std::int32_t>>);
static_assert(!WritesTemporary<Buffer<std::int32_t>>);
static_assert(!WritesFinalConst<Buffer<std::int32_t>>);
static_assert(!WritesFinalTemporary<Buffer<std::int32_t>>);
static_assert(!WritesWindowConst<Buffer<std::int32_t>>);
static_assert(!WritesWindowTemporary<Buffer<std::int32_t>>);
static_assert(!WritesEachConst<Buffer<std::int32_t>>);
static_assert(!WritesEachTemporary<Buffer<std::int32_t>>);
static_assert(!ReadsValue<std::int32_t>);
static_assert(!WritesValue<std::int32_t>);
static_assert(!WritesFinalValue<std::int32_t>);
static_assert(!WritesWindowValue<std::int32_t>);
static_assert(!WritesEachValue<std::int32_t>);
static_assert(!ReadsVolatile<Buffer<std::int32_t>>);
static_assert(!WritesVolatile<Buffer<std::int32_t>>);
static_assert(!WritesFinalVolatile<Buffer<std::int32_t>>);
static_assert(!WritesWindowVolatile<Buffer<std::int32_t>>);
static_assert(!WritesEachVolatile<Buffer<std::int32_t>>);
static_assert(!HasDependsOnSurface<PipelineBuilder>);
static_assert(!HasAfterSurface<PipelineBuilder>);
static_assert(!PreparesLvalue<PipelineBuilder>);
static_assert(PreparesRvalue<PipelineBuilder>);
static_assert(rund::compute::PipelineStepCapacity == 64u);
static_assert(rund::compute::PipelineIterationCapacity == 1024u);
static_assert(rund::compute::PipelineInnerIterationCapacity == 1024u);
static_assert(rund::compute::PipelineSealedRepetitionCapacity == 1024u);
static_assert(rund::compute::PipelineGenerationCapacity ==
              std::numeric_limits<std::uint32_t>::max());
static_assert(CanSealRepetitions<1u, PipelineBuilder>);
static_assert(
    CanSealRepetitions<rund::compute::PipelineSealedRepetitionCapacity,
                       PipelineBuilder>);
static_assert(!CanSealRepetitions<0u, PipelineBuilder>);
static_assert(
    !CanSealRepetitions<rund::compute::PipelineSealedRepetitionCapacity + 1u,
                        PipelineBuilder>);
static_assert(rund::compute::detail::PipelineBindingCapacity ==
              rund::compute::PipelineIterationCapacity *
                  rund::compute::detail::PipelineLeafCapacity);
static_assert(
    !RetainsColdPipelineBindings<rund::compute::detail::PipelineStep>);
static_assert(sizeof(rund::compute::detail::PipelineStep) <=
              sizeof(std::shared_ptr<rund::compute::detail::ProgramState>) *
                  4u);
static_assert(sizeof(rund::compute::detail::BufferClaim) ==
              sizeof(void *) * 2u);

} // namespace rund_node_test_pipeline
