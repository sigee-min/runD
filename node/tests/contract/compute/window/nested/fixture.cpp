#include "../local.hpp"
#include "local.hpp"

namespace rund::node::test_contract::window {

static_assert(kOuter == 3u);
static_assert(kOuter * kInner < rund::compute::PipelineIterationCapacity);

} // namespace rund::node::test_contract::window
