#include "src/accel/kernel/residency/persistent_sliding.hpp"

namespace {

namespace accel = rund::node::accel::detail;

template <class Request>
concept HasProject = requires(Request value) { value.project; };
template <class Request>
concept HasRelease = requires(Request value) { value.release; };
template <class Request>
concept HasReturned = requires(Request value) { value.returned; };

static_assert(!HasProject<accel::PersistentResidencySlidingRequest>);
static_assert(!HasRelease<accel::PersistentResidencySlidingRequest>);
static_assert(!HasReturned<accel::PersistentResidencySlidingRequest>);

} // namespace
