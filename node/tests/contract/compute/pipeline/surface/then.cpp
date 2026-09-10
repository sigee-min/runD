#include "../local.hpp"

#include <utility>

namespace rund_node_test_pipeline {

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

} // namespace rund_node_test_pipeline
