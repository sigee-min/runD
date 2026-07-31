#include "model.hpp"

#include "../../memory/arena.hpp"

#include <new>
#include <span>
#include <stdexcept>
#include <utility>

namespace rund::compute::detail::graph_detail {
namespace {

[[nodiscard]] Description failure(const Reason reason) {
  return Description{.status = Status::fail(reason)};
}

} // namespace

Description describe(const std::shared_ptr<GraphState> &state) {
  if (state == nullptr) {
    return failure(Reason::GraphCapacity);
  }
  if (!state->status) {
    return failure(state->status.reason());
  }
  if (state->inputs.empty() || state->outputs.empty() || state->steps.empty()) {
    return failure(Reason::GraphIncomplete);
  }

  try {
    describe_detail::Draft draft{.zero_work = state->count == 0u};
    Status status =
        describe_detail::build_resources(*state, draft.description.info);
    if (!status) {
      return failure(status.reason());
    }
    status = describe_detail::build_nodes(*state, draft);
    if (!status) {
      return failure(status.reason());
    }

    const std::span<const std::uint32_t> identity_outputs =
        state->identity_outputs.empty()
            ? std::span<const std::uint32_t>{state->outputs}
            : std::span<const std::uint32_t>{state->identity_outputs};
    status = describe_detail::validate_bindings(draft.description.info,
                                                identity_outputs);
    if (!status) {
      return failure(status.reason());
    }

    const GraphValue &root = state->values[state->inputs.front() - 1u];
    if (draft.zero_work) {
      draft.description.info.nodes.clear();
      draft.memory.clear();
    }
    draft.description.info.authored_nodes = state->authored_nodes == 0u
                                                ? state->steps.size()
                                                : state->authored_nodes;
    draft.description.info.lowered_nodes = draft.description.info.nodes.size();
    return describe_detail::finish(std::move(draft), root.type,
                                   root.fixed_format, identity_outputs,
                                   memory::arena_bytes(*state->device));
  } catch (const std::bad_alloc &) {
    return failure(Reason::GraphCapacity);
  } catch (const std::out_of_range &) {
    return failure(Reason::GraphBindingInvalid);
  }
}

} // namespace rund::compute::detail::graph_detail
