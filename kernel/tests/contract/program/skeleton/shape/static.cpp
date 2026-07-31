#include "local.hpp"

namespace program_skeleton_contract {

int SkeletonShapeStatic() {
  static_assert(std::is_same_v<rund::kernel::Index<3u>,
                               std::array<rund::kernel::u64, 3u>>);
  static_assert(!HasSpace<0u>);
  static_assert(!HasI32View<0u>);
  static_assert(!rund::kernel::skeleton_detail::DirectCallback<
                decltype(FunctionPointerCallback)>);
  static_assert(!rund::kernel::skeleton_detail::DirectCallback<void (*)(
                    rund::kernel::Index<1u>)>);
  static_assert(!rund::kernel::skeleton_detail::DirectCallback<
                std::function<void(rund::kernel::Index<1u>)>>);
  static_assert(
      !rund::kernel::skeleton_detail::DirectCallback<std::reference_wrapper<
          std::function<void(rund::kernel::Index<1u>)>>>);
  static_assert(
      !rund::kernel::skeleton_detail::DirectCallback<VirtualCallback>);
  static_assert(
      rund::kernel::skeleton_detail::DirectCallback<RvalueOnlyIndexCallback>);
  static_assert(noexcept(rund::kernel::skeleton_detail::ExecutePlan(
      std::declval<const rund::kernel::SkeletonPlan<2u> &>(),
      std::declval<RvalueOnlyIndexCallback &>())));
  return 0;
}

} // namespace program_skeleton_contract
