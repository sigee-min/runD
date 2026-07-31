#include "local.hpp"

#include "../../../../../src/compute/flow/state.hpp"
#include "../../../../../src/compute/program/state.hpp"

#include <array>
#include <vector>

namespace rund_node_flow_contract {

int CheckShape() {
  using namespace rund::compute;
  constexpr std::array<std::uint32_t, 4u> input{1u, 2u, 3u, 4u};
  auto device = open(Target::cpu(1u));
  if (!device) {
    return 1;
  }
  auto program = on(*device)
                     .map<std::uint32_t>("input-shape", input.size(),
                                         [](auto value) { return value + 1u; })
                     .compile();
  if (!program) {
    return 2;
  }
  auto source = device->upload(std::span<const std::uint32_t>{input});
  auto target = device->buffer<std::uint32_t>(input.size());
  auto job = program->resident(std::span<const std::uint32_t>{input});
  const auto &state = detail::FlowAccess::state(*program);
  if (!source || !target || !job || state == nullptr ||
      !detail::valid_input_shape(*state) || state->input_types.size() != 1u ||
      state->input_sizes != std::vector<std::size_t>{input.size()}) {
    return 3;
  }

  const auto input_types = state->input_types;
  state->input_types.clear();
  if (detail::valid_input_shape(*state)) {
    return 4;
  }
  const auto host_run = program->run(std::span<const std::uint32_t>{input});
  const auto buffer_run = program->run(*source, *target);
  const auto resident =
      program->resident(std::span<const std::uint32_t>{input});
  const Status written = job->write(std::span<const std::uint32_t>{input});
  if (host_run || host_run.code() != Code::Binding ||
      host_run.error() != "compute_program_input_shape_invalid" || buffer_run ||
      buffer_run.code() != Code::Binding ||
      buffer_run.error() != "compute_program_input_shape_invalid" || resident ||
      resident.code() != Code::Binding ||
      resident.error() != "compute_program_input_shape_invalid" || written ||
      written.code() != Code::Binding ||
      written.error() != "compute_program_input_shape_invalid") {
    return 5;
  }
  state->input_types = input_types;
  return detail::valid_input_shape(*state) ? 0 : 6;
}

} // namespace rund_node_flow_contract
