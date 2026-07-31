#pragma once

#include "../cpu/state.hpp"
#include "../device/state.hpp"

#include <accel/kernel/run/binding.hpp>
#include <accel/kernel/value.hpp>
#include <kernel/program/compute/graph/schema.hpp>
#include <rund/compute/graph/info.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace rund::compute::detail {

struct JobState;

enum class GraphBindSource : unsigned char {
  Input,
  Output,
  Internal,
};

struct GraphValueRoute final {
  GraphBindSource source{GraphBindSource::Internal};
  std::uint32_t index{};
  std::uint64_t offset_bytes{};
  std::uint64_t bytes{};
  std::uint64_t count{};
  std::uint64_t element_bytes{};
  std::uint64_t alignment{};
};

struct GraphRunBinding final {
  std::uint32_t value_index{};
  kernel::BufferRole role{kernel::BufferRole::Read};
};

struct Chunk final {
  std::size_t count{};
};

struct AccelProgram final {
  rund::AccelKernel kernel;
  std::uint64_t kernel_token_host_bytes{};
};

enum class CacheInput : unsigned char {
  Host,
  Buffers,
};

struct RunCache final {
  std::mutex gate;
  std::shared_ptr<JobState> job;
  CacheInput input{CacheInput::Host};
};

struct ProgramState final {
  std::shared_ptr<DeviceState> device;
  std::string name;
  std::size_t count{};
  std::uint64_t empty_graph_hash{};
  std::vector<Type> input_types;
  std::vector<std::size_t> input_sizes;
  std::vector<FixedFormat> input_formats;
  // Zero marks an Exact input. A nonzero entry is the authored capacity for
  // that scalar count input of a flattened Bounded<T, Count> schema.
  std::vector<std::size_t> bounded_input_capacities;
  std::vector<Type> output_types;
  std::vector<std::size_t> output_sizes;
  std::vector<FixedFormat> output_formats;
  std::vector<std::size_t> output_aliases;
  std::vector<Chunk> chunks;
  // Canonical descending-count/local-ordinal rank, sealed once at Program
  // compilation and consumed by every Pipeline workspace projection.
  std::vector<std::uint32_t> chunk_order;
  std::vector<GraphValueRoute> graph_value_routes;
  std::vector<GraphRunBinding> graph_bindings;
  std::unique_ptr<AccelProgram> accel;
  std::unique_ptr<CpuGraphProgram> cpu_graph;
  graph::Info graph_info;
  RunCache cache;

  [[nodiscard]] bool empty() const noexcept { return count == 0u; }
};

[[nodiscard]] inline bool
valid_input_shape(const ProgramState &program) noexcept {
  return !program.input_types.empty() &&
         program.input_types.size() == program.input_sizes.size() &&
         program.input_types.size() == program.input_formats.size() &&
         program.input_types.size() == program.bounded_input_capacities.size();
}

[[nodiscard]] Status validate_host_bounded_input(const ProgramState &program,
                                                 std::size_t index,
                                                 HostView input) noexcept;

[[nodiscard]] inline BufferState *graph_value_buffer(
    const ProgramState &program, const std::size_t value_index,
    const std::span<const std::shared_ptr<BufferState>> inputs,
    const std::span<const std::shared_ptr<BufferState>> outputs,
    const std::span<const std::shared_ptr<BufferState>> internals) noexcept {
  if (value_index >= program.graph_value_routes.size()) {
    return nullptr;
  }
  const GraphValueRoute route = program.graph_value_routes[value_index];
  const auto resolve = [](const auto owners,
                          const std::size_t index) noexcept -> BufferState * {
    return index < owners.size() ? owners[index].get() : nullptr;
  };
  switch (route.source) {
  case GraphBindSource::Input:
    return resolve(inputs, route.index);
  case GraphBindSource::Output:
    return resolve(outputs, route.index);
  case GraphBindSource::Internal:
    return resolve(internals, route.index);
  }
  return nullptr;
}

[[nodiscard]] inline BufferState *graph_binding_buffer(
    const ProgramState &program, const GraphRunBinding &binding,
    const std::span<const std::shared_ptr<BufferState>> inputs,
    const std::span<const std::shared_ptr<BufferState>> outputs,
    const std::span<const std::shared_ptr<BufferState>> internals) noexcept {
  return graph_value_buffer(program, binding.value_index, inputs, outputs,
                            internals);
}

[[nodiscard]] inline BufferState *graph_value_buffer_id(
    const ProgramState &program, const std::uint32_t value,
    const std::span<const std::shared_ptr<BufferState>> inputs,
    const std::span<const std::shared_ptr<BufferState>> outputs,
    const std::span<const std::shared_ptr<BufferState>> internals) noexcept {
  return value == 0u ? nullptr
                     : graph_value_buffer(program, value - 1u, inputs, outputs,
                                          internals);
}

} // namespace rund::compute::detail
