#pragma once

#include "../resource.hpp"

namespace rund::node::accel::detail::resource_detail {

[[nodiscard]] bool ValidRoute(const std::shared_ptr<PickToken> &) noexcept;

[[nodiscard]] rund::AccelCheck Validate(const std::shared_ptr<PickToken> &,
                                        const rund::Buffer &, const void *,
                                        std::uint64_t bytes,
                                        std::uint64_t offset,
                                        const char *overflow_reason) noexcept;

[[nodiscard]] rund::kernel::ResidentBufferRef
Resident(const rund::Buffer &) noexcept;

} // namespace rund::node::accel::detail::resource_detail
