#include <rund/compute/pipeline.hpp>

#include <utility>

namespace rund::compute {

PipelineBuilder &
PipelineBuilder::restore(const StateSnapshot &snapshot) & noexcept {
  detail::seed_pipeline(state_, snapshot.state_);
  return *this;
}

PipelineBuilder &&
PipelineBuilder::restore(const StateSnapshot &snapshot) && noexcept {
  static_cast<PipelineBuilder &>(*this).restore(snapshot);
  return std::move(*this);
}

PipelineBuilder &
PipelineBuilder::restore(const LatestDeviceState &latest) & noexcept {
  detail::seed_pipeline(state_, latest.state_);
  return *this;
}

PipelineBuilder &&
PipelineBuilder::restore(const LatestDeviceState &latest) && noexcept {
  static_cast<PipelineBuilder &>(*this).restore(latest);
  return std::move(*this);
}

PipelineBuilder &
PipelineBuilder::restore(const SnapshotStorage &storage) & noexcept {
  detail::seed_pipeline(state_, storage.state_);
  return *this;
}

PipelineBuilder &&
PipelineBuilder::restore(const SnapshotStorage &storage) && noexcept {
  static_cast<PipelineBuilder &>(*this).restore(storage);
  return std::move(*this);
}

PipelineBuilder &PipelineBuilder::commit() & noexcept {
  detail::commit_pipeline(state_);
  return *this;
}

PipelineBuilder &&PipelineBuilder::commit() && noexcept {
  static_cast<PipelineBuilder &>(*this).commit();
  return std::move(*this);
}

Result<Pipeline> PipelineBuilder::prepare() && noexcept {
  auto prepared = detail::prepare_pipeline(std::move(state_));
  if (!prepared) {
    return Result<Pipeline>::fail(prepared.reason(), prepared.location());
  }
  return Result<Pipeline>::success(Pipeline{std::move(prepared).value()});
}

} // namespace rund::compute
