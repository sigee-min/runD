#include "../local.hpp"

namespace rund::node::runtime_detail {

::rund::TraceRecord MakeTrace(const ::rund::TraceEvent event,
                              const ::rund::TraceCode code,
                              const ::rund::Session::Snapshot &snapshot,
                              const std::uint64_t sequence) {
  return ::rund::TraceRecord{
      .event = event,
      .code = code,
      .snapshot =
          {
              .state = snapshot.state,
              .active_compute_jobs = snapshot.active_compute_jobs,
              .scope_active = snapshot.scope_active,
              .code = snapshot.code,
          },
      .sequence = sequence,
  };
}

} // namespace rund::node::runtime_detail
