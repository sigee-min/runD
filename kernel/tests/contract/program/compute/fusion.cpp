namespace program_compute_contract {

int RunFusionPlanContract();
int RunFusionBuildContract();

int RunFusionContract() {
  if (RunFusionPlanContract() != 0) {
    return 1;
  }
  return RunFusionBuildContract();
}

}  // namespace program_compute_contract
