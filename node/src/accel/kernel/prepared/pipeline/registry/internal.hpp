#pragma once

#include "../registry.hpp"

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace rund::node::accel::detail {

inline constexpr std::uint64_t kTemplateRegistryMagic = 0x72756e442e74706cull;

// These records are the private retained registry model.  They are shared by
// the cache, budget, transaction, and observation owners, but no public
// caller can name or mutate them directly.
struct PreparedKernelTemplateEntry final {
  const KernelExecutionStep *authority{};
  std::uint64_t variant_hi{};
  std::uint64_t variant_lo{};
  const BackendOps *ops{};
  std::shared_ptr<void> prepared{};
};

enum class PreparedKernelTemplateChargeKind : std::uint8_t {
  Program,
  RecurrenceTerminal,
  RecurrenceHistory,
};

struct PreparedKernelTemplateCharge final {
  std::shared_ptr<prepared::RunState> probe{};
  const BackendOps *ops{};
  PreparedKernelTemplateChargeKind kind{
      PreparedKernelTemplateChargeKind::Program};
};

struct PreparedKernelTemplateRegistryState final {
  // Cold preparation holds this recursively across reservation, template
  // publication, and backend finalization.  Find/Publish can therefore be
  // called by the same preparation thread while a budget transaction excludes
  // a competing primary/alternate stream.
  std::recursive_mutex mutex{};
  std::vector<PreparedKernelTemplateEntry> entries{};
  std::vector<PreparedKernelTemplateCharge> template_charges{};
  PreparedKernelPipelineReservation consumed{};
  std::uint64_t magic{kTemplateRegistryMagic};
  std::uint64_t context_id{};
  rund::AccelApi api{rund::AccelApi::Auto};
};

[[nodiscard]] PreparedKernelTemplateRegistryState *
registry_state(const PreparedKernelTemplateRegistry &registry) noexcept;

} // namespace rund::node::accel::detail
