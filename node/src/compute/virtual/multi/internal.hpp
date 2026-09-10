#pragma once

#include "../state.hpp"

#include <memory>
#include <span>

namespace rund::compute::detail::virtual_multi_detail {

struct Schema final {
  std::shared_ptr<ProgramState> semantic{};
  VirtualGeometry geometry{};
  std::uint64_t page_count{};
  std::uint64_t page_bytes{};
  std::uint64_t logical_bytes{};
};

[[nodiscard]] Status
inspect(const std::shared_ptr<ProgramState> &program,
        std::span<const std::shared_ptr<VirtualBufferState>> inputs,
        const std::shared_ptr<VirtualBufferState> &output,
        const VirtualGeometry &geometry, Schema &schema) noexcept;

[[nodiscard]] Result<std::shared_ptr<PipelineState>>
prepare_pipeline(const std::shared_ptr<ProgramState> &semantic) noexcept;

} // namespace rund::compute::detail::virtual_multi_detail
