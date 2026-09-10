#pragma once

// Included by prepared/template/registry.hpp inside rund::node::accel::detail.

// Compute owns this value and may share one instance across primary and
// transactional-alternate streams. Planning writes only the fixed reservation;
// the opaque owner is allocated later, after the caller accepts that plan.
struct PreparedKernelTemplateRegistry final {
  std::shared_ptr<void> owner{};
  // Frozen public-plan authority. Runtime preparation may consume less but
  // cannot enlarge any retained byte or structural count after this point.
  PreparedKernelPipelineReservation limit{};
  PreparedKernelPipelineReservation reservation{};
};

using PreparedKernelTemplateMatch = bool (*)(const void *prepared,
                                             const void *probe) noexcept;

// Backend template lookup is keyed first by the immutable admitted step
// authority, then by a cheap variant partition. Match is the collision-safe
// semantic equality check and is mandatory.
[[nodiscard]] std::shared_ptr<void>
FindPreparedKernelTemplate(const PreparedKernelTemplateRegistry &registry,
                           const KernelExecutionStep *authority,
                           std::uint64_t variant_hi, std::uint64_t variant_lo,
                           PreparedKernelTemplateMatch match,
                           const void *probe) noexcept;

// Publishes one fully prepared immutable template. If another equal template
// already won publication, `prepared` is replaced with that sole owner.
[[nodiscard]] rund::AccelCheck PublishPreparedKernelTemplate(
    PreparedKernelTemplateRegistry &registry,
    const KernelExecutionStep *authority, std::uint64_t variant_hi,
    std::uint64_t variant_lo, const BackendOps &ops,
    PreparedKernelTemplateMatch match, const void *probe,
    std::shared_ptr<void> &prepared) noexcept;

// Materialization initializes/validates the opaque registry only after the
// allocation-free reservation was accepted.
[[nodiscard]] rund::AccelCheck BindPreparedKernelTemplateRegistry(
    const rund::AccelApi api, std::uint64_t context_id,
    PreparedKernelTemplateRegistry &registry) noexcept;
