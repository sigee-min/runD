#pragma once

#include "../projection.hpp"

namespace rund::compute::detail {

struct OrdinaryProjectionInputs final {
  VirtualBufferState *input{};
  PipelineState *pipeline{};
  const residency::StreamPlan *stream{};
  std::uint64_t input_page_bytes{};
  std::uint64_t output_page_bytes{};
  std::uint64_t capacity{};
  std::uint64_t input_payload_bytes{};
  std::uint64_t output_payload_bytes{};
  std::uint64_t input_prefix_bytes{};
  std::uint64_t output_prefix_bytes{};
  std::uint64_t input_arena_bytes{};
  std::uint64_t output_arena_bytes{};
  std::uint64_t host_input_arena_bytes{};
  std::uint64_t host_output_arena_bytes{};
};

[[nodiscard]] bool
validate_virtual_ordinary_projection(VirtualPipelineState &state,
                                     OrdinaryProjectionInputs &inputs) noexcept;

} // namespace rund::compute::detail
