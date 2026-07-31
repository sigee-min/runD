#include <rund/session.hpp>

namespace rund::compute {
template <class> class Job;
class Pipeline;
}

namespace {

template <class Session, class Job>
concept RunsCompute =
    requires(Session &session, Job &job) { session.compute(job); };

template <class Config>
concept HasDispatchOwner = requires(Config &config) { config.backend; };

using ComputeJob = rund::compute::Job<int(int)>;
using ComputePipeline = rund::compute::Pipeline;
static_assert(!RunsCompute<rund::Session, ComputeJob>,
              "Session alone must not import the Compute execution surface");
static_assert(
    !RunsCompute<rund::Session, ComputePipeline>,
    "Session alone must not import the Pipeline execution surface");
static_assert(!HasDispatchOwner<rund::SessionConfig>,
              "Session configuration must not expose Kernel dispatch policy");

} // namespace

int main() {}
