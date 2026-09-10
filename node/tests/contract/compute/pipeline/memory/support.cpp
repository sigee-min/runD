#include "local.hpp"

#include <node/runtime/compute/access.hpp>

#include "src/compute/pipeline/plan/local.hpp"

#include <cstdint>

namespace rund_node_test_pipeline::memory {

[[nodiscard]] std::shared_ptr<rund::compute::detail::ProgramState>
MakeProgram(rund::compute::Device &device,
            const std::span<const std::size_t> chunk_counts) {
  using namespace rund::compute;
  auto program = std::make_shared<detail::ProgramState>();
  program->device = detail::DeviceAccess::state(device);
  program->chunks.reserve(chunk_counts.size());
  program->chunk_order.reserve(chunk_counts.size());
  for (const std::size_t count : chunk_counts) {
    program->chunks.push_back(detail::Chunk{.count = count});
    program->chunk_order.push_back(
        static_cast<std::uint32_t>(program->chunk_order.size()));
  }
  return program;
}

} // namespace rund_node_test_pipeline::memory
