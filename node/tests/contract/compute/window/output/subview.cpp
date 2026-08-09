#include "../../allocation.hpp"
#include "../../pipeline/local.hpp"
#include "../local.hpp"
#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/output.hpp"
#include "src/compute/pipeline/plan/contract.hpp"
#include "src/compute/pipeline/plan/publication.hpp"
#include "src/compute/pipeline/state.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <utility>

namespace rund::node::test_contract::window {
[[nodiscard]] int CheckAliasSubviewRouting(Device &device) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::size_t width = 4u;
  constexpr std::array<std::uint32_t, width> first_seed{1u, 2u, 3u, 4u};
  constexpr std::array<std::uint32_t, width> second_seed{10u, 20u, 30u, 40u};
  constexpr std::array<std::uint32_t, 1u> count_value{3u};
  constexpr std::array<std::uint32_t, 9u> backing_seed{
      kOutputSentinel, kOutputSentinel, kOutputSentinel,
      kOutputSentinel, kOutputSentinel, kOutputSentinel,
      kOutputSentinel, kOutputSentinel, kOutputSentinel,
  };
  auto program = on(device)
                     .input<std::uint32_t>(width)
                     .zip_input<std::uint32_t>(width)
                     .zip_input<std::uint32_t>(width)
                     .zip_input<std::uint32_t>(1u)
                     .zip_input<std::uint32_t>(1u)
                     .branch([](auto first, auto second, auto alias, auto count,
                                auto ordinal) {
                       (void)alias;
                       (void)count;
                       (void)ordinal;
                       auto next_first =
                           first.map("window-alias-subview-first",
                                     [](auto value) { return value + 1u; });
                       auto next_second =
                           second.map("window-alias-subview-second",
                                      [](auto value) { return value + 2u; });
                       return outputs(next_first, next_second, next_first);
                     })
                     .compile();
  auto consumer =
      on(device)
          .input<std::uint32_t>(2u)
          .branch([](auto values) {
            return values.map("window-alias-subview-consumer",
                              [](auto value) { return value + 100u; });
          })
          .compile();
  auto first = device.upload<std::uint32_t>(first_seed);
  auto second = device.upload<std::uint32_t>(second_seed);
  auto count = device.upload<std::uint32_t>(count_value);
  auto backing = device.upload<std::uint32_t>(backing_seed);
  auto second_target = device.buffer<std::uint32_t>(width);
  auto sink = device.buffer<std::uint32_t>(2u);
  if (!program || !consumer || !first || !second || !count || !backing ||
      !second_target || !sink) {
    return 1;
  }
  auto target = backing->view(1u, width, 2u);
  auto downstream = backing->view(3u, 2u, 4u);
  if (!target || !downstream) {
    return 2;
  }
  const std::array<ResourceView, 3u> inputs{
      BufferAccess::view(*first, ResourceAccess::Read),
      BufferAccess::view(*second, ResourceAccess::Read),
      BufferAccess::view(*first, ResourceAccess::Read),
  };
  const std::array<ResourceView, 3u> outputs{
      BufferAccess::view(*target, ResourceAccess::Write),
      BufferAccess::view(*second_target, ResourceAccess::Write),
      BufferAccess::view(*target, ResourceAccess::Write),
  };
  auto build = make_pipeline(DeviceAccess::state(device));
  append_pipeline_windows(build, ProgramAccess::state(*program),
                          BufferAccess::view(*count, ResourceAccess::Read),
                          inputs, outputs, 4u, 1u, NoWindowTerminal, 1u);
  const std::array<ResourceView, 1u> consume_input{
      BufferAccess::view(*downstream, ResourceAccess::Read),
  };
  const std::array<ResourceView, 1u> consume_output{
      BufferAccess::view(*sink, ResourceAccess::Write),
  };
  append_pipeline(build, ProgramAccess::state(*consumer), consume_input,
                  consume_output);
  if (build == nullptr || build->failure != Reason::Ok ||
      build->steps.size() != 5u || build->internals.size() != 5u ||
      build->publications.size() != 2u) {
    return 3;
  }
  const auto plan = plan_pipeline(build);
  auto prepared = prepare_pipeline(build);
  if (!plan || !prepared || !run_pipeline(*prepared)) {
    return 4;
  }
  std::array<std::uint32_t, backing_seed.size()> backing_actual{};
  std::array<std::uint32_t, width> second_actual{};
  std::array<std::uint32_t, 2u> sink_actual{};
  if (!read_pipeline_raw(*prepared, BufferAccess::state(*backing), Type::U32,
                         FixedFormat{}, backing_actual.data(),
                         sizeof(backing_actual), backing_actual.size()) ||
      !read_pipeline_raw(*prepared, BufferAccess::state(*second_target),
                         Type::U32, FixedFormat{}, second_actual.data(),
                         sizeof(second_actual), second_actual.size()) ||
      !read_pipeline_raw(*prepared, BufferAccess::state(*sink), Type::U32,
                         FixedFormat{}, sink_actual.data(), sizeof(sink_actual),
                         sink_actual.size())) {
    return 5;
  }
  auto expected_backing = backing_seed;
  for (std::size_t index = 0u; index < width; ++index) {
    expected_backing[1u + 2u * index] = first_seed[index] + count_value[0u];
  }
  const std::array<std::uint32_t, width> expected_second{16u, 26u, 36u, 46u};
  const std::array<std::uint32_t, 2u> expected_sink{105u, 107u};
  return backing_actual == expected_backing &&
                 second_actual == expected_second &&
                 sink_actual == expected_sink
             ? 0
             : 6;
}

} // namespace rund::node::test_contract::window
