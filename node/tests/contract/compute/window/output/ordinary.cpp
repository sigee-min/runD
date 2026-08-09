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
[[nodiscard]] int CheckOrdinaryAliasAuthority(Device &device) {
  using namespace rund::compute;
  using namespace rund::compute::detail;
  constexpr std::array<std::uint32_t, 1u> first_seed{10u};
  constexpr std::array<std::uint32_t, 1u> second_seed{20u};
  constexpr std::array<std::uint32_t, 1u> count_value{4u};
  auto program = on(device)
                     .input<std::uint32_t>(1u)
                     .zip_input<std::uint32_t>(1u)
                     .zip_input<std::uint32_t>(1u)
                     .zip_input<std::uint32_t>(1u)
                     .zip_input<std::uint32_t>(1u)
                     .branch([](auto first, auto second, auto alias, auto count,
                                auto ordinal) {
                       (void)alias;
                       (void)count;
                       (void)ordinal;
                       auto next_first =
                           first.map("window-ordinary-alias-first",
                                     [](auto value) { return value + 1u; });
                       auto next_second =
                           second.map("window-ordinary-alias-second",
                                      [](auto value) { return value + 2u; });
                       return outputs(next_first, next_second, next_first);
                     })
                     .compile();
  auto first = device.upload<std::uint32_t>(first_seed);
  auto second = device.upload<std::uint32_t>(second_seed);
  auto count = device.upload<std::uint32_t>(count_value);
  auto first_target = device.buffer<std::uint32_t>(1u);
  auto second_target = device.buffer<std::uint32_t>(1u);
  auto bad_target = device.buffer<std::uint32_t>(1u);
  if (!program || !first || !second || !count || !first_target ||
      !second_target || !bad_target) {
    return 1;
  }
  const std::array<ResourceView, 3u> inputs{
      BufferAccess::view(*first, ResourceAccess::Read),
      BufferAccess::view(*second, ResourceAccess::Read),
      BufferAccess::view(*first, ResourceAccess::Read),
  };
  const std::array<ResourceView, 3u> outputs{
      BufferAccess::view(*first_target, ResourceAccess::Write),
      BufferAccess::view(*second_target, ResourceAccess::Write),
      BufferAccess::view(*first_target, ResourceAccess::Write),
  };
  auto build = make_pipeline(DeviceAccess::state(device));
  append_pipeline_windows(build, ProgramAccess::state(*program),
                          BufferAccess::view(*count, ResourceAccess::Read),
                          inputs, outputs, 4u, 1u, NoWindowTerminal, 1u);
  if (build == nullptr || build->failure != Reason::Ok ||
      build->steps.size() != 4u || build->internals.size() != 5u ||
      build->publications.size() != 2u ||
      std::any_of(build->steps.begin(), build->steps.end(),
                  [](const auto &step) {
                    return step.outputs.size() != 3u ||
                           step.outputs[0u].owner != step.outputs[2u].owner ||
                           step.outputs[0u].owner == step.outputs[1u].owner;
                  })) {
    return 2;
  }
  for (const PipelineBuildPublication &authored : build->publications) {
    const auto *terminal =
        std::get_if<PipelineBuildTerminalPublication>(&authored);
    if (terminal == nullptr) {
      return 3;
    }
    const auto source = resolve_publication_source(*build, *terminal);
    const auto output =
        source ? resolve_build_output(*build, *source)
               : Result<PipelineBuildOutputProjection>::fail(source.reason());
    if (!source || !output || source->step.value != 3u ||
        output->source.value != terminal->edge.output.value) {
      return 3;
    }
  }
  const auto plan = plan_pipeline(build);
  auto prepared = prepare_pipeline(build);
  if (!plan || !prepared) {
    return 4;
  }
  constexpr std::array<std::uint32_t, 4u> counts{0u, 1u, 3u, 4u};
  for (const std::uint32_t active : counts) {
    const std::array<std::uint32_t, 1u> next_count{active};
    if (!rund_node_test_pipeline::Overwrite(*count, next_count) ||
        !run_pipeline(*prepared)) {
      return 5;
    }
    std::array<std::uint32_t, 1u> first_actual{};
    std::array<std::uint32_t, 1u> second_actual{};
    if (!read_pipeline_raw(*prepared, BufferAccess::state(*first_target),
                           Type::U32, FixedFormat{}, first_actual.data(),
                           sizeof(first_actual), first_actual.size()) ||
        !read_pipeline_raw(*prepared, BufferAccess::state(*second_target),
                           Type::U32, FixedFormat{}, second_actual.data(),
                           sizeof(second_actual), second_actual.size()) ||
        first_actual[0u] != first_seed[0u] + active ||
        second_actual[0u] != second_seed[0u] + 2u * active) {
      return 5;
    }
  }

  const std::array<ResourceView, 3u> mismatched{
      outputs[0u],
      outputs[1u],
      BufferAccess::view(*bad_target, ResourceAccess::Write),
  };
  auto rejected = make_pipeline(DeviceAccess::state(device));
  append_pipeline_windows(rejected, ProgramAccess::state(*program),
                          BufferAccess::view(*count, ResourceAccess::Read),
                          inputs, mismatched, 4u, 1u, NoWindowTerminal, 1u);
  const auto rejected_plan = plan_pipeline(rejected);
  if (rejected_plan ||
      rejected_plan.reason() != Reason::BindingAliasUnsupported) {
    return 6;
  }
  ResourceView invalid_count =
      BufferAccess::view(*count, ResourceAccess::Write);
  auto invalid_resident = make_pipeline(DeviceAccess::state(device));
  append_pipeline_windows(invalid_resident, ProgramAccess::state(*program),
                          invalid_count, inputs, mismatched, 4u, 1u,
                          NoWindowTerminal, 1u);
  const auto invalid_resident_plan = plan_pipeline(invalid_resident);
  if (invalid_resident_plan ||
      invalid_resident_plan.reason() != Reason::BindingInvalid) {
    return 7;
  }
  auto invalid_access_outputs = outputs;
  invalid_access_outputs[2u].access = ResourceAccess::Read;
  auto invalid_access = make_pipeline(DeviceAccess::state(device));
  append_pipeline_windows(invalid_access, ProgramAccess::state(*program),
                          BufferAccess::view(*count, ResourceAccess::Read),
                          inputs, invalid_access_outputs, 4u, 1u,
                          NoWindowTerminal, 1u);
  const auto invalid_access_plan = plan_pipeline(invalid_access);
  return !invalid_access_plan &&
                 invalid_access_plan.reason() == Reason::BindingInvalid
             ? 0
             : 8;
}

} // namespace rund::node::test_contract::window
