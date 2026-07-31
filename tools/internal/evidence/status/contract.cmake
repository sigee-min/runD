foreach(required IN ITEMS ROOT BUILD)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "evidence-status contract requires ${required}")
  endif()
endforeach()

set(fixture "${BUILD}/evidence-status-contract")
set(fixture_root "${fixture}/root")
set(evidence "${fixture_root}/.cache/evidence")
file(REMOVE_RECURSE "${fixture}")
file(MAKE_DIRECTORY "${evidence}")
file(MAKE_DIRECTORY "${fixture_root}/docs/reference/performance")
file(MAKE_DIRECTORY "${fixture_root}/tools/internal/measure")
configure_file("${ROOT}/tools/internal/measure/compare"
               "${fixture_root}/tools/internal/measure/compare" COPYONLY)
configure_file("${ROOT}/tools/internal/measure/schema.pm"
               "${fixture_root}/tools/internal/measure/schema.pm" COPYONLY)

execute_process(COMMAND uname -a OUTPUT_VARIABLE current_host
                OUTPUT_STRIP_TRAILING_WHITESPACE)
set(project "${fixture}/project.pl")
file(WRITE "${project}" [=[
use strict;
use warnings;

use Digest::SHA qw(sha256_hex);

my ($source, $fixture, $flow_path, $metrics_path) = @ARGV;
require "$source/tools/internal/measure/schema.pm";
my $schema = RundMeasureSchema::load($source, 1);
my $host = RundMeasureSchema::host();
my ($source_profile) = grep {
  $schema->{profiles}{$_}{environment}{system}{value} eq $host->{system}
} sort keys %{$schema->{profiles}};
defined $source_profile
    or die "checked-in baseline has no profile for $host->{system}\n";
my $source_rows = $schema->{profiles}{$source_profile};

my $flow_rows = $source_rows->{'measure-flow'};
my @flow_metrics = sort grep { $_ ne 'semantic' } keys %$flow_rows;
open my $flow, '>', $flow_path or die "cannot write $flow_path: $!";
my @flow_semantic = ("metric\tsample\tunit\tvalue");
print {$flow} "$flow_semantic[0]\n";
for my $metric (@flow_metrics) {
  my $name = $metric;
  $name =~ s{/median\z}{}
      or die "Flow baseline metric is not a median: $metric\n";
  my $rule = $flow_rows->{$metric};
  my $row = join("\t", $name, 'median', $rule->{unit}, $rule->{value});
  print {$flow} "$row\n";
  push @flow_semantic, join("\t", $name, 'median', $rule->{unit});
}
close $flow or die "cannot close $flow_path: $!";
my $flow_digest = sha256_hex(join("\n", @flow_semantic) . "\n");

my $baseline = "$fixture/docs/reference/performance/baseline.tsv";
open my $output, '>', $baseline or die "cannot write $baseline: $!";
print {$output} "profile\troute\tmetric\tpolicy\tvalue\tenvelope\tunit\n";
sub write_rule {
  my ($output, $owner, $metric, $rule, $value) = @_;
  print {$output} join("\t", 'fixture', $owner, $metric, $rule->{policy},
                       $value, $rule->{envelope}, $rule->{unit}), "\n";
}
for my $metric (RundMeasureSchema::identity_order()) {
  my $rule = $source_rows->{baseline}{$metric};
  write_rule($output, 'baseline', $metric, $rule, $rule->{value});
}
for my $metric (RundMeasureSchema::environment_order()) {
  next unless exists $host->{$metric};
  my $rule = $source_rows->{environment}{$metric}
      // die "source profile lacks environment fact $metric\n";
  write_rule($output, 'environment', $metric, $rule, $host->{$metric});
}
for my $route (RundMeasureSchema::routes()) {
  my $rows = $source_rows->{$route};
  my $semantic = $rows->{semantic};
  write_rule($output, $route, 'semantic', $semantic,
             $route eq 'measure-flow' ? $flow_digest : $semantic->{value});
  for my $metric (sort grep { $_ ne 'semantic' } keys %$rows) {
    my $rule = $rows->{$metric};
    write_rule($output, $route, $metric, $rule, $rule->{value});
  }
}
close $output or die "cannot close $baseline: $!";

my $projected = RundMeasureSchema::load($fixture, 1);
RundMeasureSchema::select_profile($projected) eq 'fixture'
    or die "projected fixture profile did not select\n";
open my $metrics, '>', $metrics_path
    or die "cannot write $metrics_path: $!";
print {$metrics} RundMeasureSchema::route('measure-flow')->{metrics}, "\n";
close $metrics or die "cannot close $metrics_path: $!";
]=])
set(flow_log "${fixture}/flow.tsv")
set(flow_metrics_file "${fixture}/flow.metrics")
execute_process(
  COMMAND perl "${project}" "${ROOT}" "${fixture_root}" "${flow_log}"
    "${flow_metrics_file}"
  RESULT_VARIABLE project_result
  OUTPUT_VARIABLE project_output
  ERROR_VARIABLE project_error)
if(NOT project_result EQUAL 0)
  message(FATAL_ERROR
    "complete checked-in performance fixture could not be projected\n"
    "${project_output}${project_error}")
endif()
file(READ "${flow_metrics_file}" flow_metrics)
string(STRIP "${flow_metrics}" flow_metrics)
if(NOT flow_metrics MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "invalid projected Flow metric cardinality: ${flow_metrics}")
endif()
file(READ "${flow_log}" flow_log_text)

function(write_packet route run_id status manifest_text corrupt)
  set(packet "${evidence}/${route}/${run_id}")
  file(MAKE_DIRECTORY "${packet}")
  file(WRITE "${packet}/source-manifest.tsv" "${manifest_text}")
  file(WRITE "${packet}/source-identity.tsv" "fixture\t${route}\n")
  file(SHA256 "${packet}/source-manifest.tsv" manifest_sha256)
  file(SHA256 "${packet}/source-identity.tsv" identity_sha256)
  if(corrupt)
    set(manifest_sha256
        "0000000000000000000000000000000000000000000000000000000000000000")
  endif()
  file(WRITE "${packet}/run.tsv"
    "route\t${route}\n"
    "status\t${status}\n"
    "source_manifest_sha256\t${manifest_sha256}\n"
    "source_identity_sha256\t${identity_sha256}\n")
endfunction()

function(write_measure_packet route run_id packet_host log_name log_text
         include_log metrics)
  write_packet("${route}" "${run_id}" passed "current\n" FALSE)
  set(packet "${evidence}/${route}/${run_id}")
  if(include_log)
    file(WRITE "${packet}/${log_name}" "${log_text}")
    file(SHA256 "${packet}/${log_name}" log_sha256)
  else()
    set(log_sha256
        "0000000000000000000000000000000000000000000000000000000000000000")
  endif()
  file(WRITE "${packet}/baseline.log"
    "baseline\t${route}\tprofile=fixture\tmetrics=${metrics}\tstatus=passed\n")
  file(SHA256 "${packet}/baseline.log" result_sha256)
  file(APPEND "${packet}/run.tsv"
    "host\t${packet_host}\n"
    "proof:kind\tperformance\n"
    "proof:route\t${route}\n"
    "proof:status\tpassed\n"
    "proof:profile\tfixture\n"
    "proof:metrics\t${metrics}\n"
    "proof:log\t${log_name}\n"
    "proof:log:sha256\t${log_sha256}\n"
    "proof:result\tbaseline.log\n"
    "proof:result:sha256\t${result_sha256}\n")
endfunction()

file(WRITE "${fixture}/current.tsv" "current\n")
file(SHA256 "${fixture}/current.tsv" current_sha256)
write_packet(pass 20260716T000000Z passed "current\n" FALSE)
write_packet(stale 20260716T000000Z passed "old\n" FALSE)
write_packet(fail 20260716T000000Z failed "current\n" FALSE)
write_packet(corrupt 20260716T000000Z passed "current\n" TRUE)
write_measure_packet(measure-flow 20260716T000000Z "${current_host}"
                     samples.tsv "${flow_log_text}" TRUE "${flow_metrics}")

execute_process(
  COMMAND sh "${ROOT}/tools/internal/evidence/status/run"
    "${fixture_root}" "${current_sha256}" measure-flow
  RESULT_VARIABLE measure_result
  OUTPUT_VARIABLE measure_output
  ERROR_VARIABLE measure_error)
if(NOT measure_result EQUAL 0 OR
   NOT measure_output MATCHES "measure-flow\tpassed\t" OR
   NOT measure_output MATCHES "overall\tpassed")
  message(FATAL_ERROR
    "sealed measurement evidence did not pass\n${measure_output}\n${measure_error}")
endif()

set(flow_packet "${evidence}/measure-flow/20260716T000000Z")
configure_file("${flow_packet}/run.tsv" "${fixture}/flow.run" COPYONLY)
file(READ "${flow_packet}/run.tsv" flow_run)
string(REPLACE "host\t${current_host}\n" "host\twrong-host\n"
               flow_run "${flow_run}")
file(WRITE "${flow_packet}/run.tsv" "${flow_run}")
execute_process(
  COMMAND sh "${ROOT}/tools/internal/evidence/status/run"
    "${fixture_root}" "${current_sha256}" measure-flow
  RESULT_VARIABLE host_result OUTPUT_VARIABLE host_output)
if(host_result EQUAL 0 OR NOT host_output MATCHES "measure-flow\tinvalid\t")
  message(FATAL_ERROR "foreign-host measurement evidence passed\n${host_output}")
endif()
configure_file("${fixture}/flow.run" "${flow_packet}/run.tsv" COPYONLY)
file(APPEND "${flow_packet}/baseline.log" "tampered\n")

write_measure_packet(measure-compute 20260716T000000Z "${current_host}"
                     measure.log "" FALSE 1)
write_measure_packet(measure-telemetry 20260716T000000Z "${current_host}"
                     measure.log "" FALSE 1)
execute_process(COMMAND sh "${ROOT}/tools/internal/measure/attempt"
  "${fixture_root}" measure-telemetry start
  RESULT_VARIABLE telemetry_start_result)
execute_process(COMMAND sh "${ROOT}/tools/internal/measure/attempt"
  "${fixture_root}" measure-telemetry fail 11
  RESULT_VARIABLE telemetry_fail_result)
execute_process(COMMAND sh "${ROOT}/tools/internal/measure/attempt"
  "${fixture_root}" measure-telemetry complete
  RESULT_VARIABLE telemetry_complete_result)
if(NOT telemetry_start_result EQUAL 0 OR
   NOT telemetry_fail_result EQUAL 0 OR
   NOT telemetry_complete_result EQUAL 0)
  message(FATAL_ERROR "telemetry measurement attempt lifecycle failed")
endif()
execute_process(COMMAND sh "${ROOT}/tools/internal/measure/attempt"
  "${fixture_root}" measure-scheduler start RESULT_VARIABLE start_result)
execute_process(COMMAND sh "${ROOT}/tools/internal/measure/attempt"
  "${fixture_root}" measure-graph-services fail 9 RESULT_VARIABLE fail_result)
if(NOT start_result EQUAL 0 OR NOT fail_result EQUAL 0)
  message(FATAL_ERROR "measurement attempt fixture setup failed")
endif()

execute_process(
  COMMAND sh "${ROOT}/tools/internal/evidence/status/run"
    "${fixture_root}" "${current_sha256}"
    pass stale fail missing corrupt measure-flow measure-compute
    measure-scheduler measure-graph-services measure-telemetry
  RESULT_VARIABLE open_result
  OUTPUT_VARIABLE open_output
  ERROR_VARIABLE open_error)
if(open_result EQUAL 0)
  message(FATAL_ERROR "an incomplete evidence matrix passed")
endif()
foreach(expected IN ITEMS
    "pass\tpassed\t"
    "stale\tstale\t"
    "fail\tfailed\t"
    "missing\tmissing\t-\t-"
    "corrupt\tinvalid\t"
    "measure-flow\tinvalid\t"
    "measure-compute\tinvalid\t"
    "measure-scheduler\tin-progress\t-\t"
    "measure-graph-services\tfailed\t-\t"
    "measure-telemetry\tinvalid\t"
    "overall\topen")
  string(FIND "${open_output}" "${expected}" position)
  if(position EQUAL -1)
    message(FATAL_ERROR
      "evidence-status output lacks ${expected}\n${open_output}\n${open_error}")
  endif()
endforeach()

execute_process(COMMAND sh "${ROOT}/tools/internal/measure/attempt"
  "${fixture_root}" measure-scheduler complete RESULT_VARIABLE complete_running)
execute_process(COMMAND sh "${ROOT}/tools/internal/measure/attempt"
  "${fixture_root}" measure-graph-services complete RESULT_VARIABLE complete_failed)
if(NOT complete_running EQUAL 0 OR NOT complete_failed EQUAL 0)
  message(FATAL_ERROR "measurement attempt fixture cleanup failed")
endif()

execute_process(
  COMMAND sh "${ROOT}/tools/internal/evidence/status/run"
    "${fixture_root}" "${current_sha256}" pass
  RESULT_VARIABLE pass_result
  OUTPUT_VARIABLE pass_output
  ERROR_VARIABLE pass_error)
if(NOT pass_result EQUAL 0 OR
   NOT pass_output MATCHES "pass\tpassed\t" OR
   NOT pass_output MATCHES "overall\tpassed")
  message(FATAL_ERROR
    "complete evidence matrix did not pass\n${pass_output}\n${pass_error}")
endif()

file(REMOVE_RECURSE "${fixture}")
