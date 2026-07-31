#include "output.hpp"
#include "state.hpp"

#include <rund/compute/graph/info.hpp>

#include <cstring>

namespace rund::compute::detail {

Status validate_host_bounded_input(const ProgramState &program,
                                   const std::size_t index,
                                   const HostView input) noexcept {
  if (!valid_input_shape(program) || index >= program.input_types.size()) {
    return Status::fail(Reason::ProgramInputShapeInvalid);
  }
  const std::size_t capacity = program.bounded_input_capacities[index];
  if (capacity == 0u)
    return Status::success();
  if (input.data == nullptr || input.count != 1u ||
      (input.type != Type::U32 && input.type != Type::U64)) {
    return Status::fail(Reason::ProgramInputShapeInvalid);
  }
  std::uint64_t logical = 0u;
  if (input.type == Type::U64) {
    std::memcpy(&logical, input.data, sizeof(logical));
  } else {
    std::uint32_t narrow = 0u;
    std::memcpy(&narrow, input.data, sizeof(narrow));
    logical = narrow;
  }
  return logical > capacity ? Status::fail(Reason::WorksetOverflow)
                            : Status::success();
}

Result<Backend>
program_backend(const std::shared_ptr<ProgramState> &state) noexcept {
  return state == nullptr || state->device == nullptr
             ? Result<Backend>::fail(Reason::ProgramInvalid)
             : Result<Backend>::success(state->device->backend);
}

const graph::Info &
program_graph_info(const std::shared_ptr<ProgramState> &state) noexcept {
  static const graph::Info empty;
  return state == nullptr ? empty : state->graph_info;
}

std::size_t program_input_size(const std::shared_ptr<ProgramState> &state,
                               const std::size_t index) noexcept {
  return state != nullptr && index < state->input_sizes.size()
             ? state->input_sizes[index]
             : 0u;
}

std::size_t program_output_size(const std::shared_ptr<ProgramState> &state,
                                const std::size_t index) noexcept {
  if (state == nullptr) {
    return 0u;
  }
  const std::size_t output = output_index(state->output_aliases, index);
  return output < state->output_sizes.size() ? state->output_sizes[output] : 0u;
}

bool program_input_sizes_match(
    const std::shared_ptr<ProgramState> &state,
    const std::span<const std::size_t> sizes) noexcept {
  if (state == nullptr || !valid_input_shape(*state) ||
      state->input_sizes.size() != sizes.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < sizes.size(); ++index) {
    if (state->input_sizes[index] != sizes[index]) {
      return false;
    }
  }
  return true;
}

} // namespace rund::compute::detail
