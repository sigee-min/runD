#include "local.hpp"

#include <optional>
#include <utility>

namespace rund_node_test_pipeline::checkpoint {

int CheckPortability(Context &context) {
  using namespace rund::compute;

  // Portable storage crosses Device ownership; the live resident handle does
  // not. A different graph with the same byte shape is rejected unchanged.
  auto replacement =
      open(rund::node::test_contract::target_for(context.backend, 2u));
  if (!replacement) {
    return 29;
  }
  auto replacement_advance =
      on(*replacement)
          .map<std::int32_t>("pipeline-reusable-checkpoint", Initial.size(),
                             [](auto value) { return value + 1; })
          .compile();
  auto replacement_first = replacement->buffer<std::int32_t>(Initial.size());
  auto replacement_second = replacement->buffer<std::int32_t>(Initial.size());
  auto portable =
      replacement_advance && replacement_first && replacement_second
          ? pipeline(*replacement)
                .state(*replacement_first, *replacement_second)
                .then(*replacement_advance, read(*replacement_first),
                      write(*replacement_second))
                .restore(context.storage)
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  if (!portable || portable->generation() != 3u ||
      !ReadExact(*portable, *replacement_first, context.observed) ||
      context.observed != Thrice || !portable->run() ||
      portable->generation() != 4u ||
      !ReadExact(*portable, *replacement_second, context.observed) ||
      context.observed != Fourth) {
    return 30;
  }

  // Host-owned storage is backend-independent. When this test target exposes
  // another backend, restore the exact bytes/hash into it and prove execution
  // parity. The live resident handle must reject that distinct Device/backend
  // without mutating the restored Pipeline.
  Backend portable_backend = Backend::Unavailable;
  std::optional<Device> cross_backend_device{};
  // Accelerator contexts are single-owner on some test adapters. Let the
  // accelerator's own backend pass prove portability to CPU; opening it as a
  // temporary CPU subcase would consume the context before its full contract.
  if (context.backend != Backend::Cpu) {
    for (const Backend candidate :
         rund::node::test_contract::selected_compute_backends()) {
      if (candidate == context.backend) {
        continue;
      }
      auto opened = open(rund::node::test_contract::target_for(candidate, 2u));
      if (opened) {
        portable_backend = candidate;
        cross_backend_device.emplace(std::move(opened).value());
        break;
      }
      if (opened.reason() != Reason::AdapterUnavailable) {
        return 38;
      }
    }
  }
  if (portable_backend != Backend::Unavailable) {
    auto cross_backend_advance =
        on(*cross_backend_device)
            .map<std::int32_t>("pipeline-reusable-checkpoint", Initial.size(),
                               [](auto value) { return value + 1; })
            .compile();
    auto cross_backend_first =
        cross_backend_device->buffer<std::int32_t>(Initial.size());
    auto cross_backend_second =
        cross_backend_device->buffer<std::int32_t>(Initial.size());
    auto cross_backend =
        cross_backend_advance && cross_backend_first && cross_backend_second
            ? pipeline(*cross_backend_device)
                  .state(*cross_backend_first, *cross_backend_second)
                  .then(*cross_backend_advance, read(*cross_backend_first),
                        write(*cross_backend_second))
                  .restore(context.storage)
                  .commit()
                  .prepare()
            : Result<Pipeline>::fail(Reason::PipelineInvalid);
    if (!cross_backend || cross_backend->generation() != 3u ||
        !ReadExact(*cross_backend, *cross_backend_first, context.observed) ||
        context.observed != Thrice || !cross_backend->run() ||
        cross_backend->generation() != 4u ||
        !ReadExact(*cross_backend, *cross_backend_second, context.observed) ||
        context.observed != Fourth) {
      return 39;
    }
    const Status resident_backend_mismatch =
        cross_backend->restore(context.latest);
    if (resident_backend_mismatch ||
        resident_backend_mismatch.reason() != Reason::BindingDeviceMismatch ||
        cross_backend->generation() != 4u || cross_backend->poisoned() ||
        !ReadExact(*cross_backend, *cross_backend_second, context.observed) ||
        context.observed != Fourth) {
      return 40;
    }
  }

  auto resident_first = Upload(*replacement, Initial);
  auto resident_second = replacement->buffer<std::int32_t>(Initial.size());
  auto cross_device =
      replacement_advance && resident_first && resident_second
          ? pipeline(*replacement)
                .state(*resident_first, *resident_second)
                .then(*replacement_advance, read(*resident_first),
                      write(*resident_second))
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const std::shared_ptr<detail::PipelineState> cross_device_state =
      cross_device ? detail::PipelineStateAccess::state(*cross_device)
                   : std::shared_ptr<detail::PipelineState>{};
  const std::shared_ptr<detail::PipelinePublicationState>
      cross_device_publication =
          cross_device_state == nullptr ? nullptr
                                        : cross_device_state->publication;
  const Status cross_device_restore =
      cross_device ? cross_device->restore(context.latest)
                   : Status::fail(Reason::PipelineInvalid);
  if (!cross_device || cross_device_restore ||
      cross_device_restore.reason() != Reason::BindingDeviceMismatch ||
      cross_device->poisoned() || cross_device->generation() != 0u ||
      cross_device_state->publication != cross_device_publication ||
      !ReadExact(*cross_device, *resident_first, context.observed) ||
      context.observed != Initial) {
    return 31;
  }

  auto incompatible_program =
      on(context.device)
          .map<std::int32_t>("pipeline-reusable-incompatible", Initial.size(),
                             [](auto value) { return value + 2; })
          .compile();
  auto incompatible_first = Upload(context.device, Initial);
  auto incompatible_second =
      context.device.buffer<std::int32_t>(Initial.size());
  auto incompatible =
      incompatible_program && incompatible_first && incompatible_second
          ? pipeline(context.device)
                .state(*incompatible_first, *incompatible_second)
                .then(*incompatible_program, read(*incompatible_first),
                      write(*incompatible_second))
                .commit()
                .prepare()
          : Result<Pipeline>::fail(Reason::PipelineInvalid);
  const Status incompatible_restore =
      incompatible ? incompatible->restore(context.storage)
                   : Status::fail(Reason::PipelineInvalid);
  if (!incompatible || incompatible_restore ||
      incompatible_restore.reason() != Reason::PipelineInvalid ||
      incompatible->poisoned() || incompatible->generation() != 0u ||
      !ReadExact(*incompatible, *incompatible_first, context.observed) ||
      context.observed != Initial) {
    return 32;
  }
  return 0;
}

} // namespace rund_node_test_pipeline::checkpoint
