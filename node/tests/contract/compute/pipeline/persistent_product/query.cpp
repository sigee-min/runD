#include "local.hpp"

#if !defined(RUND_NODE_TEST_BACKEND_CPU)

#include "src/accel/backend/ops/table.hpp"
#include "src/accel/kernel/prepared/interface/api.hpp"
#include "src/accel/kernel/prepared/model.hpp"

#include <array>
#include <memory>
#include <string_view>

namespace rund_node_test_persistent_product {
namespace {

using namespace rund::node::accel::detail;

struct QueryConfig final {
  const char *reason{"accel_kernel_pipeline_invalid"};
  bool mode_mismatch{};
  bool memory_mismatch{};
  bool non_product{};
  std::uint8_t width{2u};
};

QueryConfig &query_config() noexcept {
  thread_local QueryConfig config{};
  return config;
}

[[nodiscard]] PersistentResidencySlidingCapability
TaxonomyQuery(const std::span<const PreparedResidencyPersistentSlidingRole>,
              const std::uint64_t, const ResidencySlidingMemory memory,
              const PersistentResidencySlidingMode mode) noexcept {
  QueryConfig &config = query_config();
  return {.check = {std::string_view{config.reason} == "ok", config.reason},
          .memory = config.memory_mismatch
                        ? ResidencySlidingMemory::NoncoherentIntegratedCopy
                        : memory,
          .width = config.width,
          .whole_run_preencoded = !config.non_product,
          .host_epoch_callbacks_zero = true,
          .mode = config.mode_mismatch
                      ? PersistentResidencySlidingMode::BackendChunked
                      : mode};
}

[[nodiscard]] PersistentResidencySlidingPreparation
TaxonomyPrepare(const PersistentResidencySlidingRequest &request) noexcept {
  PersistentResidencySlidingPreparation result{};
  result.capability.check = {false, "compute_pipeline_capacity"};
  result.capability.memory = request.memory;
  result.capability.mode = request.mode;
  return result;
}

[[nodiscard]] auto make_roles(BackendOps &ops) {
  const auto make_pipeline = [&ops] {
    auto state = std::make_shared<prepared::PipelineState>();
    state->ops = &ops;
    state->backend = std::make_shared<std::uint8_t>(1u);
    return PreparedKernelPipeline{.owner = std::move(state), .ok = true};
  };
  return std::array<PreparedResidencyPersistentSlidingRole, 2u>{
      PreparedResidencyPersistentSlidingRole{.pipeline = make_pipeline(),
                                             .local_count = 1u,
                                             .first_control_generation = 1u,
                                             .control_generation_stride = 1u,
                                             .first_descriptor_generation = 1u,
                                             .descriptor_generation_stride = 1u,
                                             .slot = 0u},
      PreparedResidencyPersistentSlidingRole{.pipeline = make_pipeline(),
                                             .local_count = 1u,
                                             .first_control_generation = 1u,
                                             .control_generation_stride = 1u,
                                             .first_descriptor_generation = 1u,
                                             .descriptor_generation_stride = 1u,
                                             .slot = 1u}};
}

[[nodiscard]] PersistentResidencySlidingCapability
query(const std::span<const PreparedResidencyPersistentSlidingRole> roles) {
  return QueryPreparedKernelPipelinePersistentSlidingCapability(
      roles, 2u, ResidencySlidingMemory::HostCoherent,
      PersistentResidencySlidingMode::OneSubmit);
}

} // namespace

bool CheckPersistentCapabilityTaxonomy() noexcept {
  BackendOps ops{};
  ops.query_persistent_sliding_capability = TaxonomyQuery;
  ops.prepare_persistent_sliding = TaxonomyPrepare;
  auto roles = make_roles(ops);
  const std::span<const PreparedResidencyPersistentSlidingRole> role_span{
      roles.data(), roles.size()};

  constexpr std::array<const char *, 5u> failure_reasons{
      {"accel_kernel_pipeline_invalid", "accel_native_failure",
       "compute_device_lost", "compute_pipeline_capacity",
       "compute_backend_unsupported"}};
  for (const char *const expected : failure_reasons) {
    QueryConfig &config = query_config();
    config = QueryConfig{.reason = expected};
    const auto result = query(role_span);
    if (result.check.ok || result.check.reason == nullptr ||
        std::string_view{result.check.reason} != expected ||
        result.memory != ResidencySlidingMemory::HostCoherent ||
        result.mode != PersistentResidencySlidingMode::OneSubmit) {
      return false;
    }
  }

  query_config() = QueryConfig{.reason = "ok", .mode_mismatch = true};
  const auto mode_mismatch = query(role_span);
  if (mode_mismatch.check.ok || mode_mismatch.check.reason == nullptr ||
      std::string_view{mode_mismatch.check.reason} !=
          "accel_kernel_pipeline_invalid" ||
      mode_mismatch.mode != PersistentResidencySlidingMode::OneSubmit) {
    return false;
  }

  query_config() = QueryConfig{.reason = "ok", .memory_mismatch = true};
  const auto memory_mismatch = query(role_span);
  if (memory_mismatch.check.ok || memory_mismatch.check.reason == nullptr ||
      std::string_view{memory_mismatch.check.reason} !=
          "accel_kernel_pipeline_invalid" ||
      memory_mismatch.memory != ResidencySlidingMemory::HostCoherent) {
    return false;
  }

  query_config() = QueryConfig{.reason = "compute_backend_unsupported"};
  const auto explicit_unsupported = query(role_span);
  if (explicit_unsupported.check.ok ||
      explicit_unsupported.check.reason == nullptr ||
      std::string_view{explicit_unsupported.check.reason} !=
          "compute_backend_unsupported") {
    return false;
  }

  query_config() = QueryConfig{.reason = "ok", .non_product = true};
  const auto non_product = query(role_span);
  if (non_product.check.ok || non_product.check.reason == nullptr ||
      std::string_view{non_product.check.reason} !=
          "accel_kernel_pipeline_invalid") {
    return false;
  }

  query_config() = QueryConfig{.reason = "ok", .width = 1u};
  const auto wrong_width = query(role_span);
  if (wrong_width.check.ok || wrong_width.check.reason == nullptr ||
      std::string_view{wrong_width.check.reason} !=
          "accel_kernel_pipeline_invalid") {
    return false;
  }

  BackendOps absent_query_ops = ops;
  absent_query_ops.query_persistent_sliding_capability = nullptr;
  auto absent_query_roles = make_roles(absent_query_ops);
  query_config() = QueryConfig{};
  const auto absent_query =
      query(std::span<const PreparedResidencyPersistentSlidingRole>{
          absent_query_roles.data(), absent_query_roles.size()});
  if (absent_query.check.ok || absent_query.check.reason == nullptr ||
      std::string_view{absent_query.check.reason} !=
          "compute_backend_unsupported") {
    return false;
  }

  BackendOps absent_prepare_ops = ops;
  absent_prepare_ops.prepare_persistent_sliding = nullptr;
  auto absent_prepare_roles = make_roles(absent_prepare_ops);
  const auto absent_prepare =
      query(std::span<const PreparedResidencyPersistentSlidingRole>{
          absent_prepare_roles.data(), absent_prepare_roles.size()});
  if (absent_prepare.check.ok || absent_prepare.check.reason == nullptr ||
      std::string_view{absent_prepare.check.reason} !=
          "compute_backend_unsupported") {
    return false;
  }

  BackendOps malformed_ops = absent_query_ops;
  auto malformed_roles = make_roles(malformed_ops);
  malformed_roles[1u].local_count = 0u;
  const auto malformed =
      query(std::span<const PreparedResidencyPersistentSlidingRole>{
          malformed_roles.data(), malformed_roles.size()});
  BackendOps malformed_state_ops = ops;
  auto malformed_state_roles = make_roles(malformed_state_ops);
  malformed_state_roles[0u].pipeline.ok = false;
  const auto malformed_state =
      query(std::span<const PreparedResidencyPersistentSlidingRole>{
          malformed_state_roles.data(), malformed_state_roles.size()});
  query_config() = QueryConfig{};
  return !malformed.check.ok && malformed.check.reason != nullptr &&
         std::string_view{malformed.check.reason} ==
             "accel_kernel_pipeline_invalid" &&
         !malformed_state.check.ok && malformed_state.check.reason != nullptr &&
         std::string_view{malformed_state.check.reason} ==
             "accel_kernel_pipeline_invalid";
}

} // namespace rund_node_test_persistent_product

#else

namespace rund_node_test_persistent_product {

bool CheckPersistentCapabilityTaxonomy() noexcept { return true; }

} // namespace rund_node_test_persistent_product

#endif
