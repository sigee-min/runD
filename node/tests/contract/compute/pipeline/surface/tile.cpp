#include "../local.hpp"

#include <type_traits>
#include <utility>

namespace rund_node_test_pipeline {

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
concept TilePreparesLvalue = requires(T value) { value.prepare(); };

template <class T>
concept TilePreparesRvalue = requires(T value) {
  std::move(value).prepare();
};

template <class T>
concept TileHasRunSurface = requires(T value) { value.run(); };

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
          rund::compute::read(outer, seed),
          rund::compute::write_final(output));
    };

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
static_assert(!TilePreparesLvalue<TileBody>);
static_assert(!TilePreparesRvalue<TileBody>);
static_assert(!TileHasRunSurface<TileBody>);
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

} // namespace rund_node_test_pipeline
