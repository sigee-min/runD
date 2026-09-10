#pragma once

#include "../prefetch.hpp"

namespace rund::compute::detail::residency::prefetch_detail {

[[nodiscard]] std::uint64_t now_ns() noexcept;
[[nodiscard]] bool alias_request_matches(const AliasLease &,
                                         const PrefetchRequest &,
                                         std::uint64_t token) noexcept;

} // namespace rund::compute::detail::residency::prefetch_detail
