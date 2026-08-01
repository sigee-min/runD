#include "local.hpp"

#include "src/compute/pipeline/claim.hpp"
#include "src/compute/pipeline/state.hpp"

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

struct ValidHostFeedback final {
  [[nodiscard]] rund::compute::Status
  operator()(rund::compute::HostIteration &) noexcept {
    return rund::compute::Status::success();
  }
};

struct ThrowingHostFeedback final {
  [[nodiscard]] rund::compute::Status
  operator()(rund::compute::HostIteration &) {
    return rund::compute::Status::success();
  }
};

struct InvalidHostFeedback final {
  void operator()(rund::compute::HostIteration &) noexcept {}
};

template <class Callback>
concept CanHostFeedback = requires(Pipeline &prepared, Callback callback) {
  rund::compute::host_feedback(prepared, 4u, callback);
};

template <class T>
concept RetainsColdPipelineBindings = requires(T value) {
  value.inputs;
  value.outputs;
  value.input_bindings;
  value.output_bindings;
};

template <class Builder, class Program, class Input, class Output>
concept CanThenWithWrite =
    requires(Builder builder, Program program, Input input, Output output) {
      builder.then(program, rund::compute::read(input),
                   rund::compute::write(output));
    };

template <class Builder, class Program, class Input, class Output>
concept CanThenWithFinal =
    requires(Builder builder, Program program, Input input, Output output) {
      builder.then(program, rund::compute::read(input),
                   rund::compute::write_final(output));
    };

template <class Builder, class Program, class Input, class Output>
concept CanThenWithEach =
    requires(Builder builder, Program program, Input input, Output output) {
      builder.then(program, rund::compute::read(input),
                   rund::compute::write_each(output));
    };

template <class Builder, class Program, class Input, class Output>
concept CanRepeatWithWrite =
    requires(Builder builder, Program program, Input input, Output output) {
      builder.template repeat<8u>(program, rund::compute::read(input),
                                  rund::compute::write(output));
    };

template <class Builder, class Program, class Input, class Output>
concept CanRepeatWithFinal =
    requires(Builder builder, Program program, Input input, Output output) {
      builder.template repeat<8u>(program, rund::compute::read(input),
                                  rund::compute::write_final(output));
    };

template <class Builder, class Program, class Input, class Output>
concept CanRepeatWithEach =
    requires(Builder builder, Program program, Input input, Output output) {
      builder.template repeat<8u>(program, rund::compute::read(input),
                                  rund::compute::write_each(output));
    };

template <class Builder, class Program, class Count, class Input, class Output>
concept CanWindowsWithWrite = requires(
    Builder builder, Program program, Count count, Input input, Output output) {
  builder.template windows<64u, 8u>(program, rund::compute::window(count),
                                    rund::compute::read(input),
                                    rund::compute::write(output));
};

template <class Builder, class Program, class Count, class Input, class Output>
concept CanWindowsWithFinal = requires(
    Builder builder, Program program, Count count, Input input, Output output) {
  builder.template windows<64u, 8u>(program, rund::compute::window(count),
                                    rund::compute::read(input),
                                    rund::compute::write_final(output));
};

template <class Builder, class Program, class Count, class Input, class Output>
concept CanWindowsWithEach = requires(Builder builder, Program program,
                                      Count count, Input input, Output output) {
  builder.template windows<64u, 8u>(program, rund::compute::window(count),
                                    rund::compute::read(input),
                                    rund::compute::write_each(output));
};

template <std::size_t N, class Builder>
concept CanSealRepetitions = requires(Builder builder) {
  builder.template sealed_repetitions<N>();
  std::move(builder).template sealed_repetitions<N>();
};

template <std::size_t N, class Seed, class Action, class Fold>
concept CanMakeTileRepeat = requires(Seed seed, Action action, Fold fold) {
  rund::compute::tile_repeat<N>(seed, action, fold);
};

template <std::size_t N, class Seed, class Fold>
concept CanMakeTileFold = requires(Seed seed, Fold fold) {
  rund::compute::tile_repeat<N>(seed, fold);
};

template <class Builder, class Body, class Count, class Outer, class Seed,
          class Output>
concept CanTileWindowsLvalue = requires(Builder builder, Body body, Count count,
                                        Outer outer, Seed seed, Output output) {
  builder.template windows<64u, 8u>(body, rund::compute::window(count),
                                    rund::compute::read(outer, seed),
                                    rund::compute::write_final(output));
};

template <class Builder, class Body, class Count, class Outer, class Seed,
          class Output>
concept CanTileWindowsWithWrite =
    requires(Builder builder, Body body, Count count, Outer outer, Seed seed,
             Output output) {
      builder.template windows<64u, 8u>(body, rund::compute::window(count),
                                        rund::compute::read(outer, seed),
                                        rund::compute::write(output));
    };

template <class Builder, class Body, class Count, class Outer, class Seed,
          class Output>
concept CanTileWindowsWithEach =
    requires(Builder builder, Body body, Count count, Outer outer, Seed seed,
             Output output) {
      builder.template windows<64u, 8u>(body, rund::compute::window(count),
                                        rund::compute::read(outer, seed),
                                        rund::compute::write_each(output));
    };

template <class Builder, class Body, class Count, class Outer, class Seed,
          class Output>
concept CanTileWindowsRvalue = requires(Builder builder, Body body, Count count,
                                        Outer outer, Seed seed, Output output) {
  std::move(builder).template windows<64u, 8u>(
      body, rund::compute::window(count), rund::compute::read(outer, seed),
      rund::compute::write_final(output));
};

template <class Builder, class Body, class Count, class Outer, class Seed,
          class Output, class WindowOutput>
concept CanTileWindowsWithWindowLvalue =
    requires(Builder builder, Body body, Count count, Outer outer, Seed seed,
             Output output, WindowOutput window_output) {
      builder.template windows<64u, 8u>(
          body, rund::compute::window(count), rund::compute::read(outer, seed),
          rund::compute::write_final(output),
          rund::compute::write_window(window_output));
    };

template <class Builder, class Body, class Count, class Outer, class Seed,
          class Output, class WindowOutput>
concept CanTileWindowsWithWindowRvalue =
    requires(Builder builder, Body body, Count count, Outer outer, Seed seed,
             Output output, WindowOutput window_output) {
      std::move(builder).template windows<64u, 8u>(
          body, rund::compute::window(count), rund::compute::read(outer, seed),
          rund::compute::write_final(output),
          rund::compute::write_window(window_output));
    };

template <class Builder, class Body, class Count, class Outer, class Output>
concept CanTileWindowsWithoutSeed = requires(
    Builder builder, Body body, Count count, Outer outer, Output output) {
  builder.template windows<64u, 8u>(body, rund::compute::window(count),
                                    rund::compute::read(outer),
                                    rund::compute::write_final(output));
};

template <std::size_t Terminal, class Builder, class Body, class Count,
          class Outer, class Seed, class Output>
concept CanTerminalTileWindows =
    requires(Builder builder, Body body, Count count, Outer outer, Seed seed,
             Output output) {
      builder.template windows<64u, 8u>(
          body, rund::compute::window(count).template until<Terminal>(),
          rund::compute::read(outer, seed), rund::compute::write_final(output));
    };

using ReadPack =
    decltype(rund::compute::read(std::declval<Buffer<std::int32_t> &>()));
using WritePack =
    decltype(rund::compute::write(std::declval<Buffer<std::int32_t> &>()));
using WriteFinalPack = decltype(rund::compute::write_final(
    std::declval<Buffer<std::int32_t> &>()));
using WriteWindowPack = decltype(rund::compute::write_window(
    std::declval<Buffer<std::int32_t> &>()));
using WriteEachPack =
    decltype(rund::compute::write_each(std::declval<Buffer<std::int32_t> &>()));

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
using IntProgram = rund::compute::Program<std::int32_t(std::int32_t)>;
static_assert(CanThenWithWrite<PipelineBuilder, IntProgram,
                               Buffer<std::int32_t>, Buffer<std::int32_t>>);
static_assert(!CanThenWithFinal<PipelineBuilder, IntProgram,
                                Buffer<std::int32_t>, Buffer<std::int32_t>>);
static_assert(!CanThenWithEach<PipelineBuilder, IntProgram,
                               Buffer<std::int32_t>, Buffer<std::int32_t>>);
static_assert(!CanThenWithWrite<PipelineBuilder, IntProgram,
                                Buffer<std::uint32_t>, Buffer<std::int32_t>>);
static_assert(!CanThenWithWrite<PipelineBuilder, IntProgram,
                                Buffer<std::int32_t>, Buffer<std::uint32_t>>);
static_assert(!CanRepeatWithWrite<PipelineBuilder, IntProgram,
                                  Buffer<std::int32_t>, Buffer<std::int32_t>>);
static_assert(CanRepeatWithFinal<PipelineBuilder, IntProgram,
                                 Buffer<std::int32_t>, Buffer<std::int32_t>>);
static_assert(CanRepeatWithEach<PipelineBuilder, IntProgram,
                                Buffer<std::int32_t>, Buffer<std::int32_t>>);
static_assert(!CanRepeatWithFinal<PipelineBuilder, IntProgram,
                                  Buffer<std::uint32_t>, Buffer<std::int32_t>>);
static_assert(!CanRepeatWithEach<PipelineBuilder, IntProgram,
                                 Buffer<std::int32_t>, Buffer<std::uint32_t>>);

using WindowProgram = rund::compute::Program<std::int32_t(
    std::int32_t, std::uint32_t, std::uint32_t)>;
static_assert(
    !CanWindowsWithWrite<PipelineBuilder, WindowProgram, Buffer<std::uint32_t>,
                         Buffer<std::int32_t>, Buffer<std::int32_t>>);
static_assert(
    CanWindowsWithFinal<PipelineBuilder, WindowProgram, Buffer<std::uint32_t>,
                        Buffer<std::int32_t>, Buffer<std::int32_t>>);
static_assert(
    !CanWindowsWithEach<PipelineBuilder, WindowProgram, Buffer<std::uint32_t>,
                        Buffer<std::int32_t>, Buffer<std::int32_t>>);

using TileSeedProgram =
    rund::compute::Program<rund::compute::Outputs<std::int32_t, std::int16_t>(
        std::int64_t, std::uint32_t, std::uint32_t)>;
using TileActionProgram =
    rund::compute::Program<std::int32_t(std::int32_t, std::int16_t)>;
using TileFoldProgram = rund::compute::Program<std::uint32_t(
    std::uint32_t, std::int32_t, std::int16_t)>;
using TileWindowFoldProgram =
    rund::compute::Program<rund::compute::Outputs<std::uint32_t, std::int32_t>(
        std::uint32_t, std::int32_t, std::int16_t)>;
using BadTileWindowFoldProgram =
    rund::compute::Program<rund::compute::Outputs<std::uint32_t, std::uint32_t>(
        std::uint32_t, std::int32_t, std::int16_t)>;
using TileSeedNoExternalProgram =
    rund::compute::Program<rund::compute::Outputs<std::int32_t, std::int16_t>(
        std::uint32_t, std::uint32_t)>;
using TileBody = decltype(rund::compute::tile_repeat<8u>(
    std::declval<const TileSeedProgram &>(),
    std::declval<const TileActionProgram &>(),
    std::declval<const TileFoldProgram &>()));
using TileBodyNoExternal = decltype(rund::compute::tile_repeat<8u>(
    std::declval<const TileSeedNoExternalProgram &>(),
    std::declval<const TileActionProgram &>(),
    std::declval<const TileFoldProgram &>()));
using TileWindowBody = decltype(rund::compute::tile_repeat<8u>(
    std::declval<const TileSeedProgram &>(),
    std::declval<const TileActionProgram &>(),
    std::declval<const TileWindowFoldProgram &>()));
using TileWindowFoldBody = decltype(rund::compute::tile_repeat<0u>(
    std::declval<const TileSeedProgram &>(),
    std::declval<const TileWindowFoldProgram &>()));
using BadTileSeedCoordinateProgram =
    rund::compute::Program<rund::compute::Outputs<std::int32_t, std::int16_t>(
        std::int64_t, std::uint32_t, std::int32_t)>;
using BadTileActionInputProgram =
    rund::compute::Program<std::int32_t(std::int16_t, std::int16_t)>;
using BadTileActionOutputProgram =
    rund::compute::Program<std::int16_t(std::int32_t, std::int16_t)>;
using BadTileFoldProgram = rund::compute::Program<std::uint32_t(
    std::uint32_t, std::int16_t, std::int32_t)>;

static_assert(
    CanMakeTileRepeat<8u, TileSeedProgram, TileActionProgram, TileFoldProgram>);
static_assert(CanMakeTileRepeat<8u, TileSeedNoExternalProgram,
                                TileActionProgram, TileFoldProgram>);
static_assert(!CanMakeTileRepeat<0u, TileSeedProgram, TileActionProgram,
                                 TileFoldProgram>);
static_assert(
    !CanMakeTileRepeat<rund::compute::PipelineInnerIterationCapacity + 1u,
                       TileSeedProgram, TileActionProgram, TileFoldProgram>);
static_assert(!CanMakeTileRepeat<8u, BadTileSeedCoordinateProgram,
                                 TileActionProgram, TileFoldProgram>);
static_assert(!CanMakeTileRepeat<8u, TileSeedProgram, BadTileActionInputProgram,
                                 TileFoldProgram>);
static_assert(!CanMakeTileRepeat<8u, TileSeedProgram,
                                 BadTileActionOutputProgram, TileFoldProgram>);
static_assert(!CanMakeTileRepeat<8u, TileSeedProgram, TileActionProgram,
                                 BadTileFoldProgram>);
static_assert(CanMakeTileFold<0u, TileSeedProgram, TileFoldProgram>);
static_assert(!CanMakeTileFold<1u, TileSeedProgram, TileFoldProgram>);
static_assert(
    !CanMakeTileFold<0u, BadTileSeedCoordinateProgram, TileFoldProgram>);
static_assert(std::is_nothrow_copy_constructible_v<TileBody>);
static_assert(std::is_nothrow_move_constructible_v<TileBody>);
static_assert(!std::is_copy_assignable_v<TileBody>);
static_assert(!std::is_move_assignable_v<TileBody>);
static_assert(!std::is_constructible_v<TileBody, TileSeedProgram,
                                       TileActionProgram, TileFoldProgram>);
static_assert(!PreparesLvalue<TileBody>);
static_assert(!PreparesRvalue<TileBody>);
static_assert(!HasRunSurface<TileBody>);
static_assert(CanHostFeedback<ValidHostFeedback>);
static_assert(!CanHostFeedback<ThrowingHostFeedback>);
static_assert(!CanHostFeedback<InvalidHostFeedback>);
static_assert(
    CanTileWindowsLvalue<PipelineBuilder, TileBody, Buffer<std::uint32_t>,
                         Buffer<std::uint32_t>, Buffer<std::int64_t>,
                         Buffer<std::uint32_t>>);
static_assert(
    !CanTileWindowsWithWrite<PipelineBuilder, TileBody, Buffer<std::uint32_t>,
                             Buffer<std::uint32_t>, Buffer<std::int64_t>,
                             Buffer<std::uint32_t>>);
static_assert(
    !CanTileWindowsWithEach<PipelineBuilder, TileBody, Buffer<std::uint32_t>,
                            Buffer<std::uint32_t>, Buffer<std::int64_t>,
                            Buffer<std::uint32_t>>);
static_assert(CanTileWindowsWithWindowLvalue<
              PipelineBuilder, TileWindowBody, Buffer<std::uint32_t>,
              Buffer<std::uint32_t>, Buffer<std::int64_t>,
              Buffer<std::uint32_t>, Buffer<std::int32_t>>);
static_assert(CanTileWindowsWithWindowRvalue<
              PipelineBuilder, TileWindowBody, Buffer<std::uint32_t>,
              Buffer<std::uint32_t>, Buffer<std::int64_t>,
              Buffer<std::uint32_t>, Buffer<std::int32_t>>);
static_assert(CanTileWindowsWithWindowLvalue<
              PipelineBuilder, TileWindowFoldBody, Buffer<std::uint32_t>,
              Buffer<std::uint32_t>, Buffer<std::int64_t>,
              Buffer<std::uint32_t>, Buffer<std::int32_t>>);
static_assert(CanTileWindowsWithWindowRvalue<
              PipelineBuilder, TileWindowFoldBody, Buffer<std::uint32_t>,
              Buffer<std::uint32_t>, Buffer<std::int64_t>,
              Buffer<std::uint32_t>, Buffer<std::int32_t>>);
static_assert(!CanTileWindowsWithWindowLvalue<
              PipelineBuilder, TileWindowBody, Buffer<std::uint32_t>,
              Buffer<std::uint32_t>, Buffer<std::int64_t>,
              Buffer<std::uint32_t>, Buffer<std::uint32_t>>);
static_assert(
    CanTileWindowsRvalue<PipelineBuilder, TileBody, Buffer<std::uint32_t>,
                         Buffer<std::uint32_t>, Buffer<std::int64_t>,
                         Buffer<std::uint32_t>>);
static_assert(
    !CanTileWindowsLvalue<PipelineBuilder, TileBody, Buffer<std::uint32_t>,
                          Buffer<std::int64_t>, Buffer<std::uint32_t>,
                          Buffer<std::uint32_t>>);
static_assert(
    !CanTileWindowsLvalue<PipelineBuilder, TileBody, Buffer<std::uint32_t>,
                          Buffer<std::uint32_t>, Buffer<std::int64_t>,
                          Buffer<std::int32_t>>);
static_assert(
    !CanTileWindowsWithoutSeed<PipelineBuilder, TileBody, Buffer<std::uint32_t>,
                               Buffer<std::uint32_t>, Buffer<std::uint32_t>>);
static_assert(CanTileWindowsWithoutSeed<
              PipelineBuilder, TileBodyNoExternal, Buffer<std::uint32_t>,
              Buffer<std::uint32_t>, Buffer<std::uint32_t>>);
static_assert(
    CanTerminalTileWindows<0u, PipelineBuilder, TileBody, Buffer<std::uint32_t>,
                           Buffer<std::uint32_t>, Buffer<std::int64_t>,
                           Buffer<std::uint32_t>>);
static_assert(
    !CanTerminalTileWindows<1u, PipelineBuilder, TileBody,
                            Buffer<std::uint32_t>, Buffer<std::uint32_t>,
                            Buffer<std::int64_t>, Buffer<std::uint32_t>>);

struct ValueField final {};
struct WeightField final {};

[[nodiscard]] int CheckSurface(rund::compute::Device &device) {
  using namespace rund::compute;
  constexpr std::array<std::int32_t, 4u> input_values{1, 2, 3, 4};
  auto input = Upload(device, input_values);
  if (!input) {
    return 1;
  }

  auto record_program =
      on(device)
          .map<std::int32_t>("pipeline-record", input_values.size(),
                             [](auto value) {
                               return record(field<ValueField>(value * 2),
                                             field<WeightField>(value + 7));
                             })
          .compile();
  auto record_values = device.buffer<std::int32_t>(input_values.size());
  auto record_weights = device.buffer<std::int32_t>(input_values.size());
  if (!record_program || !record_values || !record_weights) {
    return 2;
  }
  auto record_pipeline = pipeline(device)
                             .then(*record_program, read(*input),
                                   write(*record_values, *record_weights))
                             .prepare();
  if (!record_pipeline || !record_pipeline->run()) {
    return 3;
  }
  std::array<std::int32_t, 4u> values{};
  std::array<std::int32_t, 4u> weights{};
  if (!ReadExact(*record_pipeline, *record_values, values) ||
      !ReadExact(*record_pipeline, *record_weights, weights) ||
      values != std::array<std::int32_t, 4u>{2, 4, 6, 8} ||
      weights != std::array<std::int32_t, 4u>{8, 9, 10, 11} ||
      record_pipeline->stats().output_hash == 0u) {
    return 4;
  }

  auto bounded_program =
      on(device)
          .map<std::int32_t>("pipeline-bounded", input_values.size(),
                             [](auto value) { return value; })
          .filter([](auto value) { return value > 2; })
          .compile();
  auto bounded_values = device.buffer<std::int32_t>(input_values.size());
  auto bounded_count = device.buffer<std::uint32_t>(1u);
  if (!bounded_program || !bounded_values || !bounded_count) {
    return 5;
  }
  auto bounded_pipeline = pipeline(device)
                              .then(*bounded_program, read(*input),
                                    write(*bounded_values, *bounded_count))
                              .prepare();
  if (!bounded_pipeline || !bounded_pipeline->run()) {
    return 6;
  }
  std::array<std::int32_t, 4u> compacted{};
  std::array<std::uint32_t, 1u> count{};
  if (!ReadExact(*bounded_pipeline, *bounded_values, compacted) ||
      !ReadExact(*bounded_pipeline, *bounded_count, count) || count[0] != 2u ||
      compacted[0] != 3 || compacted[1] != 4) {
    return 7;
  }

  auto aliases_program =
      on(device)
          .map<std::int32_t>("pipeline-alias-source", input_values.size(),
                             [](auto value) { return value * 3; })
          .branch([](auto source) {
            auto next = source.map("pipeline-alias-next",
                                   [](auto value) { return value + 1; });
            return outputs(source, next, source);
          })
          .compile();
  auto alias_source = device.buffer<std::int32_t>(input_values.size());
  auto alias_next = device.buffer<std::int32_t>(input_values.size());
  if (!aliases_program || !alias_source || !alias_next) {
    return 8;
  }
  auto alias_pipeline =
      pipeline(device)
          .then(*aliases_program, read(*input),
                write(*alias_source, *alias_next, *alias_source))
          .prepare();
  if (!alias_pipeline || !alias_pipeline->run()) {
    return 9;
  }
  if (!ReadExact(*alias_pipeline, *alias_source, values) ||
      !ReadExact(*alias_pipeline, *alias_next, weights) ||
      values != std::array<std::int32_t, 4u>{3, 6, 9, 12} ||
      weights != std::array<std::int32_t, 4u>{4, 7, 10, 13} ||
      alias_pipeline->stats().pipeline.resource_count != 3u) {
    return 10;
  }

  auto bad_alias_source = device.buffer<std::int32_t>(input_values.size());
  if (!bad_alias_source) {
    return 11;
  }
  auto rejected_projection =
      pipeline(device)
          .then(*aliases_program, read(*input),
                write(*alias_source, *alias_next, *bad_alias_source))
          .prepare();
  if (rejected_projection ||
      rejected_projection.reason() != Reason::BindingAliasUnsupported) {
    return 12;
  }

  auto shared_count_program =
      on(device)
          .map<std::uint32_t>("pipeline-shared-count", input_values.size(),
                              [](auto value) { return value; })
          .filter([](auto value) { return value > 1u; })
          .branch([](auto values) {
            auto next = values.map("pipeline-shared-next",
                                   [](auto value) { return value + 10u; });
            auto last = values.map("pipeline-shared-last",
                                   [](auto value) { return value + 20u; });
            return outputs(values, next, last);
          })
          .compile();
  constexpr std::array<std::uint32_t, 17u> shared_initial{
      1u, 2u, 3u, 4u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
  auto shared_storage = Upload(device, shared_initial);
  if (!shared_count_program || !shared_storage) {
    return 13;
  }
  auto shared_input = shared_storage->view(0u, input_values.size());
  auto first = shared_storage->view(input_values.size(), input_values.size());
  auto shared_count = shared_storage->view(input_values.size() * 2u, 1u);
  auto second =
      shared_storage->view(input_values.size() * 2u + 1u, input_values.size());
  auto last =
      shared_storage->view(input_values.size() * 3u + 1u, input_values.size());
  if (!shared_input || !first || !shared_count || !second || !last) {
    return 14;
  }
  auto shared_count_pipeline =
      pipeline(device)
          .then(*shared_count_program, read(*shared_input),
                write(*first, *shared_count, *second, *shared_count, *last,
                      *shared_count))
          .prepare();
  const Status shared_run = shared_count_pipeline
                                ? shared_count_pipeline->run()
                                : Status::fail(shared_count_pipeline.reason());
  std::array<std::uint32_t, shared_initial.size()> shared_values{};
  if (!shared_run ||
      !ReadExact(*shared_count_pipeline, *shared_storage, shared_values) ||
      shared_values[4] != 2u || shared_values[5] != 3u ||
      shared_values[6] != 4u || shared_values[8] != 3u ||
      shared_values[9] != 12u || shared_values[10] != 13u ||
      shared_values[11] != 14u || shared_values[13] != 22u ||
      shared_values[14] != 23u || shared_values[15] != 24u) {
    return 15;
  }
  return 0;
}

} // namespace rund_node_test_pipeline
