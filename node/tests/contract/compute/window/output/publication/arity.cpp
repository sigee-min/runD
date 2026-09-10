#include "../../../pipeline/local.hpp"
#include "../../local.hpp"
#include "../local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/output.hpp"
#include "src/compute/pipeline/plan/contract.hpp"
#include "src/compute/pipeline/plan/publication.hpp"
#include "src/compute/pipeline/state.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace rund::node::test_contract::window {

template <std::size_t WindowCount>
  requires(WindowCount >= 4u && WindowCount <= 6u)
[[nodiscard]] auto PublicationArityFoldProgram(Device &device) {
  using namespace rund::compute;
  return on(device)
      .input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(kOutputTile)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .zip_input<std::uint32_t>(1u)
      .branch([](auto outer, auto tile, auto count, auto ordinal,
                 auto witness) {
        (void)count;
        (void)ordinal;
        (void)witness;
        auto next = outer.combine(
            "window-publication-arity-fold", tile.reduce(Reduce::Sum),
            [](auto left, auto right) { return left + right; });
        auto first = tile.map("window-publication-arity-0",
                              [](auto value) { return value; });
        auto second = tile.map("window-publication-arity-1",
                               [](auto value) { return value + 1u; });
        auto third = tile.map("window-publication-arity-2",
                              [](auto value) { return value + 2u; });
        auto fourth = tile.map("window-publication-arity-3",
                               [](auto value) { return value + 3u; });
        if constexpr (WindowCount == 4u) {
          return outputs(next, first, second, third, fourth);
        } else {
          auto fifth = tile.map("window-publication-arity-4",
                                [](auto value) { return value + 4u; });
          if constexpr (WindowCount == 5u) {
            return outputs(next, first, second, third, fourth, fifth);
          } else {
            auto sixth = tile.map("window-publication-arity-5",
                                  [](auto value) { return value + 5u; });
            return outputs(next, first, second, third, fourth, fifth, sixth);
          }
        }
      })
      .compile();
}
template <std::size_t WindowCount, std::size_t Inner, class Seed, class Action,
          class Fold>
[[nodiscard]] int CheckPublicationArityCase(Device &device, const Seed &seed,
                                            const Action &action,
                                            const Fold &fold) {
  using namespace rund::compute;
  static_assert(WindowCount >= 4u && WindowCount <= 6u);
  static_assert(Inner <= 1u);
  constexpr std::array<std::uint32_t, 1u> initial{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> count_value{kOutputMaximum};
  constexpr std::array<std::uint32_t, 1u> witness_value{0u};

  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kOutputValues);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(witness_value);
  auto count = device.upload<std::uint32_t>(count_value);
  auto final = device.buffer<std::uint32_t>(1u);
  auto first = device.buffer<std::uint32_t>(kOutputMaximum);
  auto second = device.buffer<std::uint32_t>(kOutputMaximum);
  auto third = device.buffer<std::uint32_t>(kOutputMaximum);
  auto fourth = device.buffer<std::uint32_t>(kOutputMaximum);
  auto fifth = device.buffer<std::uint32_t>(kOutputMaximum);
  auto sixth = device.buffer<std::uint32_t>(kOutputMaximum);
  if (!outer || !values || !lanes || !witness || !count || !final || !first ||
      !second || !third || !fourth || !fifth || !sixth) {
    return 1;
  }

  const auto body = [&] {
    if constexpr (Inner == 0u) {
      return tile_repeat<0u>(seed, fold);
    } else {
      return tile_repeat<Inner>(seed, action, fold);
    }
  }();
  auto builder = pipeline(device);
  if constexpr (WindowCount == 4u) {
    builder.windows<kOutputMaximum, kOutputTile>(
        body, rund::compute::window(*count),
        read(*outer, *values, *lanes, *witness), write_final(*final),
        write_window(*first, *second, *third, *fourth));
  } else if constexpr (WindowCount == 5u) {
    builder.windows<kOutputMaximum, kOutputTile>(
        body, rund::compute::window(*count),
        read(*outer, *values, *lanes, *witness), write_final(*final),
        write_window(*first, *second, *third, *fourth, *fifth));
  } else {
    builder.windows<kOutputMaximum, kOutputTile>(
        body, rund::compute::window(*count),
        read(*outer, *values, *lanes, *witness), write_final(*final),
        write_window(*first, *second, *third, *fourth, *fifth, *sixth));
  }
  const auto plan = builder.plan();
  if (!plan || plan->outer_window_count != kOutputOuter ||
      plan->inner_iteration_count != Inner ||
      plan->publish_count != 1u + WindowCount * kOutputOuter) {
    if (!plan) {
      std::fprintf(stderr,
                   "window publication arity W=%zu N=%zu reason=%u "
                   "location=%u/%u/%u\n",
                   WindowCount, Inner, static_cast<unsigned>(plan.reason()),
                   plan.location().step, plan.location().iteration,
                   static_cast<unsigned>(plan.location().nested_phase));
    }
    return 2;
  }
  auto prepared = std::move(builder).prepare();
  if (!prepared) {
    std::fprintf(stderr,
                 "window publication arity prepare W=%zu N=%zu reason=%u "
                 "location=%u/%u/%u\n",
                 WindowCount, Inner, static_cast<unsigned>(prepared.reason()),
                 prepared.location().step, prepared.location().iteration,
                 static_cast<unsigned>(prepared.location().nested_phase));
    return 3;
  }
  const Status executed = prepared->run();
  if (!executed) {
    std::fprintf(stderr, "window publication arity run W=%zu N=%zu reason=%u\n",
                 WindowCount, Inner, static_cast<unsigned>(executed.reason()));
    return 3;
  }
  std::array<std::uint32_t, kOutputMaximum> observed{};
  const auto &last = [&]() -> const rund::compute::Buffer<std::uint32_t> & {
    if constexpr (WindowCount == 4u) {
      return *fourth;
    } else if constexpr (WindowCount == 5u) {
      return *fifth;
    } else {
      return *sixth;
    }
  }();
  if (!prepared->read(last, observed)) {
    return 4;
  }
  for (std::size_t index = 0u; index < observed.size(); ++index) {
    if (observed[index] !=
        kOutputValues[index] + static_cast<std::uint32_t>(WindowCount - 1u)) {
      return 5;
    }
  }
  return 0;
}

template <class Seed, class Action>
[[nodiscard]] int CheckPublicationArity(Device &device, const Seed &seed,
                                        const Action &action) {
  auto four = PublicationArityFoldProgram<4u>(device);
  auto five = PublicationArityFoldProgram<5u>(device);
  auto six = PublicationArityFoldProgram<6u>(device);
  if (!four || !five || !six) {
    return 1;
  }
  if (const int result =
          CheckPublicationArityCase<4u, 0u>(device, seed, action, *four);
      result != 0) {
    return 10 + result;
  }
  if (const int result =
          CheckPublicationArityCase<5u, 0u>(device, seed, action, *five);
      result != 0) {
    return 20 + result;
  }
  if (const int result =
          CheckPublicationArityCase<6u, 0u>(device, seed, action, *six);
      result != 0) {
    return 30 + result;
  }
  if (const int result =
          CheckPublicationArityCase<4u, 1u>(device, seed, action, *four);
      result != 0) {
    return 40 + result;
  }
  if (const int result =
          CheckPublicationArityCase<5u, 1u>(device, seed, action, *five);
      result != 0) {
    return 50 + result;
  }
  if (const int result =
          CheckPublicationArityCase<6u, 1u>(device, seed, action, *six);
      result != 0) {
    return 60 + result;
  }

  constexpr std::array<std::uint32_t, 1u> initial{kOutputInitial};
  constexpr std::array<std::uint32_t, 1u> count_value{kOutputMaximum};
  constexpr std::array<std::uint32_t, 1u> witness_value{0u};
  auto outer = device.upload<std::uint32_t>(initial);
  auto values = device.upload<std::uint32_t>(kOutputValues);
  auto lanes = device.upload<std::uint32_t>(kOutputLanes);
  auto witness = device.upload<std::uint32_t>(witness_value);
  auto count = device.upload<std::uint32_t>(count_value);
  auto final = device.buffer<std::uint32_t>(1u);
  auto first = device.buffer<std::uint32_t>(kOutputMaximum);
  auto second = device.buffer<std::uint32_t>(kOutputMaximum);
  auto third = device.buffer<std::uint32_t>(kOutputMaximum);
  auto fourth = device.buffer<std::uint32_t>(kOutputMaximum);
  auto fifth = device.buffer<std::uint32_t>(kOutputMaximum);
  if (!outer || !values || !lanes || !witness || !count || !final || !first ||
      !second || !third || !fourth || !fifth) {
    return 71;
  }
  using namespace rund::compute::detail;
  const std::array<ResourceView, 4u> inputs{
      BufferAccess::view(*outer, ResourceAccess::Read),
      BufferAccess::view(*values, ResourceAccess::Read),
      BufferAccess::view(*lanes, ResourceAccess::Read),
      BufferAccess::view(*witness, ResourceAccess::Read),
  };
  const std::array<ResourceView, 1u> finals{
      BufferAccess::view(*final, ResourceAccess::Write),
  };
  const std::array<ResourceView, 5u> windows{
      BufferAccess::view(*first, ResourceAccess::Write),
      BufferAccess::view(*second, ResourceAccess::Write),
      BufferAccess::view(*third, ResourceAccess::Write),
      BufferAccess::view(*fourth, ResourceAccess::Write),
      BufferAccess::view(*fifth, ResourceAccess::Write),
  };
  auto build = make_pipeline(DeviceAccess::state(device));
  append_pipeline_window_repeat(
      build, ProgramAccess::state(seed), {}, ProgramAccess::state(*five),
      BufferAccess::view(*count, ResourceAccess::Read), inputs, finals, windows,
      kOutputMaximum, kOutputTile, 0u, rund::compute::NoWindowTerminal, 1u);
  if (build == nullptr || build->failure != Reason::Ok ||
      build->publications.empty()) {
    return 72;
  }
  auto *terminal =
      std::get_if<PipelineBuildTerminalPublication>(&build->publications[0u]);
  if (terminal == nullptr) {
    return 73;
  }
  terminal->edge.output.value = 5u;
  const auto invalid = plan_pipeline(build);
  const rund::compute::Location location = invalid.location();
  if (invalid || invalid.reason() != Reason::PipelineInvalid ||
      location.step != 0u || location.iteration != 0u ||
      location.nested_phase != rund::compute::PipelineNestedPhase::Fold) {
    return 74;
  }
  return 0;
}

int CheckOutputPublicationArity(Device &device, const OutputSeed &seed,
                                const OutputAction &action) {
  return CheckPublicationArity(device, seed, action);
}
} // namespace rund::node::test_contract::window
