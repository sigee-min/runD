#pragma once

#include "../claim.hpp"

namespace rund::compute::detail::claim_detail {

struct PrivateTerminalCheck final {
  bool registry{};
  bool identity{};
  bool publication{};
  bool phase{};
  bool valid{};
};

[[nodiscard]] PrivateTerminalCheck
private_terminal_check(const PipelineState &, PipelineTerminal,
                       bool private_authority_proven) noexcept;

} // namespace rund::compute::detail::claim_detail
