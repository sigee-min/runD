#include "access.hpp"

#include "../access.hpp"
#include "local.hpp"

namespace rund::node {

namespace detail {

class StopTokenSourceAccess final {
public:
  StopTokenSourceAccess() = delete;

  [[nodiscard]] static bool Identity(task::stop_token token,
                                     std::uint64_t *const scheduler_id,
                                     std::uint64_t *const source_id,
                                     std::uint64_t *const generation,
                                     std::uint64_t *const epoch) noexcept {
    return ::rund::detail::task::StopAccess::Identity(
        token, scheduler_id, source_id, generation, epoch);
  }
};

} // namespace detail

bool scheduler_access::StopTokenIdentity(task::stop_token token,
                                         std::uint64_t *const scheduler_id,
                                         std::uint64_t *const source_id,
                                         std::uint64_t *const generation,
                                         std::uint64_t *const epoch) noexcept {
  return detail::StopTokenSourceAccess::Identity(token, scheduler_id, source_id,
                                                 generation, epoch);
}

} // namespace rund::node
