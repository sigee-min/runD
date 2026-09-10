#include "../persist.hpp"

#include "../../../../../device/residency/registry/graph_persist_owner.hpp"

#include <array>
#include <span>

namespace rund::compute::detail::graph_reduce::output_persist_detail {

bool cancel_unsubmitted(residency::Authority &authority,
                        residency::execution::GraphPersist &persist) noexcept {
  auto persist_owner = authority.graph_persists();
  std::array<residency::execution::GraphPersistCompletion,
             residency::execution::GraphPersistCapacity>
      completions{};
  const auto pages = persist.pages();
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    completions[index] = residency::execution::GraphPersistCompletion{
        .key = pages[index].key,
        .backing_offset = pages[index].backing_offset,
        .bytes = 0u,
        .frame = pages[index].frame,
    };
  }
  return persist_owner.terminal_graph_persist(
             persist, Status::fail(Reason::CompletionInvalid),
             residency::execution::TerminalKind::Known, false,
             std::span<const residency::execution::GraphPersistCompletion>{
                 completions.data(), pages.size()}) &&
         persist_owner.release_graph_persist(persist);
}

} // namespace rund::compute::detail::graph_reduce::output_persist_detail
