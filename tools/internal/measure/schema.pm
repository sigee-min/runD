package RundMeasureSchema;

use strict;
use warnings;

use Cwd qw(abs_path);
use Digest::SHA;
use File::Basename qw(basename dirname);
use POSIX qw(uname);

my @IDENTITIES = qw(manifest scheduler1 scheduler2 scheduler3
                     compute1 compute2 compute3 flow1 flow2 flow3
                     graph1 graph2 graph3 telemetry1 telemetry2 telemetry3);
my @ENVIRONMENT = qw(system release machine workers model cpu);
my @ROUTES = qw(measure-scheduler measure-compute measure-flow
                 measure-graph-services measure-telemetry);
my %ROUTE = (
  'measure-scheduler' => {
    identity => 'scheduler', log => 'measure.log', artifact => 'runD-scheduler-measure',
    metrics => 95,
  },
  'measure-compute' => {
    identity => 'compute', log => 'measure.log', artifact => 'runD-compute-measure',
    metrics => 145,
  },
  'measure-flow' => {
    identity => 'flow', log => 'samples.tsv', artifact => 'runD-flow-measure',
    metrics => 5,
  },
  'measure-graph-services' => {
    identity => 'graph', log => 'measure.tsv', artifact => 'runD-graph-services-measure',
    metrics => 9,
  },
  'measure-telemetry' => {
    identity => 'telemetry', log => 'measure.log', artifact => 'runD-telemetry-measure',
    metrics => 49,
  },
);
my %ENVIRONMENT_UNIT = (
  system => 'text', release => 'text', machine => 'text', workers => 'count',
  model => 'text', cpu => 'text',
);

sub fail {
  die "$_[0]\n";
}

sub decimal {
  my ($value) = @_;
  return defined($value) &&
         $value =~ /\A(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\z/;
}

sub sha256 {
  my ($path) = @_;
  open my $input, '<', $path or fail("cannot read $path: $!");
  binmode $input;
  my $digest = Digest::SHA->new(256)->addfile($input)->hexdigest();
  close $input or fail("cannot close $path: $!");
  return $digest;
}

sub read_value {
  my (@command) = @_;
  open my $pipe, '-|', @command or return '';
  local $/;
  my $value = <$pipe> // '';
  close $pipe or return '';
  $value =~ s/[\r\n]+\z//;
  return $value;
}

sub host {
  my ($system, undef, $release, undef, $machine) = uname();
  my %facts = (
    system => $system,
    release => $release,
    machine => $machine,
    workers => read_value('/usr/bin/getconf', '_NPROCESSORS_ONLN'),
  );
  if ($system eq 'Darwin') {
    $facts{model} = read_value('/usr/sbin/sysctl', '-n', 'hw.model');
    $facts{cpu} =
        read_value('/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string');
  }
  return \%facts;
}

sub host_text {
  open my $pipe, '-|', 'uname', '-a' or fail('cannot observe current host');
  my $value = <$pipe> // '';
  close $pipe or fail('cannot observe current host');
  $value =~ s/[\r\n]+\z//;
  $value ne '' or fail('current host identity is empty');
  return $value;
}

sub routes {
  return @ROUTES;
}

sub route {
  my ($name) = @_;
  return $ROUTE{$name} // fail("unregistered measurement route: $name");
}

sub identity_order {
  return @IDENTITIES;
}

sub environment_order {
  return @ENVIRONMENT;
}

sub data_rows {
  my ($system) = @_;
  defined($system) && $system ne '' or fail('missing baseline system');
  my $environment = 4 + ($system eq 'Darwin' ? 2 : 0);
  my $metrics = 0;
  $metrics += $ROUTE{$_}{metrics} for @ROUTES;
  return scalar(@IDENTITIES) + $environment + $metrics;
}

sub order_key {
  my ($profile, $owner, $metric) = @_;
  my %owner_rank = (
    baseline => 0, environment => 1, 'measure-scheduler' => 2,
    'measure-compute' => 3, 'measure-flow' => 4,
    'measure-graph-services' => 5, 'measure-telemetry' => 6,
  );
  my $owner_index = $owner_rank{$owner};
  defined($owner_index) or fail("invalid route: $owner");
  my $metric_index;
  if ($owner eq 'baseline') {
    my %rank;
    @rank{@IDENTITIES} = (0 .. $#IDENTITIES);
    exists $rank{$metric} or fail("unknown baseline identity: $metric");
    $metric_index = sprintf('%04u', $rank{$metric});
  } elsif ($owner eq 'environment') {
    my %rank;
    @rank{@ENVIRONMENT} = (0 .. $#ENVIRONMENT);
    exists $rank{$metric} or fail("unknown environment fact: $metric");
    $metric_index = sprintf('%04u', $rank{$metric});
  } else {
    $metric_index = $metric eq 'semantic' ? "0\t" : "1\t$metric";
  }
  return join("\t", $profile, sprintf('%02u', $owner_index), $metric_index);
}

sub load {
  my ($root, $complete) = @_;
  my $baseline = "$root/docs/reference/performance/baseline.tsv";
  -f $baseline or fail("missing $baseline");
  open my $input, '<', $baseline or fail("cannot read $baseline: $!");
  my $header = <$input> // '';
  chomp $header;
  $header eq "profile\troute\tmetric\tpolicy\tvalue\tenvelope\tunit"
      or fail('baseline header changed');
  my @rules;
  my %profiles;
  my %seen;
  my $previous = '';
  while (my $line = <$input>) {
    chomp $line;
    $line ne '' or fail('empty baseline row');
    my @field = split /\t/, $line, -1;
    @field == 7 or fail("malformed baseline row: $line");
    my ($profile, $owner, $metric, $policy, $value, $envelope, $unit) = @field;
    $profile =~ /\A[a-z0-9]+\z/ or fail("invalid profile: $profile");
    exists $ROUTE{$owner} || $owner eq 'baseline' || $owner eq 'environment'
        or fail("invalid route: $owner");
    $metric ne '' or fail('empty metric');
    $policy =~ /\A(?:exact|upper|identity)\z/
        or fail("invalid policy for $owner/$metric: $policy");
    my $identity = "$profile\t$owner\t$metric";
    !$seen{$identity}++ or fail("duplicate baseline row: $identity");
    my $key = order_key($profile, $owner, $metric);
    ($previous eq '' || $key gt $previous)
        or fail("baseline rows are not in canonical order: $identity");
    $previous = $key;
    if ($policy eq 'upper') {
      decimal($value)
          or fail("upper value for $owner/$metric is not a nonnegative finite decimal: $value");
      decimal($envelope)
          or fail("envelope for $owner/$metric is not a nonnegative finite decimal: $envelope");
      $unit ne '' && $unit ne '-'
          or fail("upper metric has no unit: $owner/$metric");
    } elsif ($policy eq 'exact') {
      $envelope eq '-'
          or fail("exact metric has an envelope: $owner/$metric");
      $owner ne 'baseline'
          or fail("baseline identity has exact policy: $metric");
    } else {
      $owner eq 'baseline' &&
          $value =~ /\A[0-9a-f]{64}\z/ && $envelope eq '-' &&
          $unit eq 'sha256'
          or fail('invalid baseline identity row');
    }
    my $rule = {
      profile => $profile, route => $owner, metric => $metric,
      policy => $policy, value => $value, envelope => $envelope, unit => $unit,
    };
    push @rules, $rule;
    $profiles{$profile}{$owner}{$metric} = $rule;
  }
  close $input or fail("cannot close $baseline: $!");
  @rules or fail('baseline table is empty');

  for my $profile (sort keys %profiles) {
    my $rows = $profiles{$profile};
    my $environment = $rows->{environment}
        // fail("profile $profile has no environment");
    for my $metric (qw(system release machine workers)) {
      my $rule = $environment->{$metric}
          // fail("profile $profile is missing environment $metric");
      $rule->{policy} eq 'exact' &&
          $rule->{unit} eq $ENVIRONMENT_UNIT{$metric} &&
          $rule->{value} ne ''
          or fail("invalid environment rule $profile/$metric");
    }
    $environment->{workers}{value} =~ /\A[1-9][0-9]*\z/
        or fail("invalid positive worker count for $profile");
    my @optional = $environment->{system}{value} eq 'Darwin'
        ? qw(model cpu) : ();
    for my $metric (@optional) {
      my $rule = $environment->{$metric}
          // fail("profile $profile is missing environment $metric");
      $rule->{policy} eq 'exact' &&
          $rule->{unit} eq $ENVIRONMENT_UNIT{$metric} &&
          $rule->{value} ne ''
          or fail("invalid environment rule $profile/$metric");
    }
    scalar(keys %$environment) == 4 + scalar(@optional)
        or fail("profile $profile has an unknown environment fact");

    if ($complete) {
      my $baseline_rows = $rows->{baseline}
          // fail("profile $profile has no baseline identity");
      for my $metric (@IDENTITIES) {
        exists $baseline_rows->{$metric}
            or fail("profile $profile is missing baseline identity $metric");
      }
      scalar(keys %$baseline_rows) == scalar(@IDENTITIES)
          or fail("profile $profile has an unknown baseline identity");
      for my $route (@ROUTES) {
        my $route_rows = $rows->{$route}
            // fail("profile $profile has no $route baseline");
        scalar(keys %$route_rows) == $ROUTE{$route}{metrics}
            or fail("profile $profile has invalid $route metric cardinality");
      }
    }
    if (exists $rows->{baseline}) {
      for my $metric (keys %{$rows->{baseline}}) {
        $rows->{baseline}{$metric}{policy} eq 'identity'
            or fail("invalid baseline identity policy: $profile/$metric");
      }
    }
    for my $route (@ROUTES) {
      next unless exists $rows->{$route};
      my $semantic = $rows->{$route}{semantic};
      if ($complete || defined($semantic)) {
        defined($semantic) && $semantic->{policy} eq 'exact' &&
            $semantic->{unit} eq 'sha256' &&
            $semantic->{value} =~ /\A[0-9a-f]{64}\z/
            or fail("invalid semantic identity for $profile/$route");
      }
      for my $metric (keys %{$rows->{$route}}) {
        next if $metric eq 'semantic';
        $rows->{$route}{$metric}{policy} eq 'upper'
            or fail("non-semantic route metric is not upper: $profile/$route/$metric");
      }
    }
    for my $owner (keys %$rows) {
      $owner eq 'baseline' || $owner eq 'environment' || exists $ROUTE{$owner}
          or fail("profile $profile has unknown owner $owner");
    }
  }
  return {rules => \@rules, profiles => \%profiles};
}

sub select_profile {
  my ($schema) = @_;
  my $host = host();
  my @matched;
  PROFILE:
  for my $profile (sort keys %{$schema->{profiles}}) {
    my $environment = $schema->{profiles}{$profile}{environment};
    for my $metric (keys %$environment) {
      next PROFILE if !exists $host->{$metric} ||
                      "$host->{$metric}" ne $environment->{$metric}{value};
    }
    push @matched, $profile;
  }
  @matched == 1
      or fail('current host has no unique checked-in baseline profile');
  return $matched[0];
}

sub read_fields {
  my ($path, $expected_order) = @_;
  open my $input, '<', $path or fail("cannot read $path: $!");
  my %fields;
  my @order;
  while (my $line = <$input>) {
    chomp $line;
    my ($key, $value, @extra) = split /\t/, $line, -1;
    defined($key) && defined($value) && !@extra && $key ne ''
        or fail("malformed row in $path");
    !exists $fields{$key} or fail("duplicate field $key in $path");
    $fields{$key} = $value;
    push @order, $key;
  }
  close $input or fail("cannot close $path: $!");
  if (defined $expected_order) {
    join("\n", @order) eq join("\n", @$expected_order)
        or fail("field order or set changed in $path");
  }
  return \%fields;
}

sub packet {
  my ($root, $schema, $profile, $route_name, $packet_argument) = @_;
  my $spec = route($route_name);
  select_profile($schema) eq $profile
      or fail("packet requested a foreign profile: $profile");
  my $packet = abs_path($packet_argument)
      // fail("missing calibration packet $packet_argument");
  my $parent = "$root/.cache/evidence/$route_name";
  dirname($packet) eq $parent &&
      basename($packet) =~ /\A[0-9]{8}T[0-9]{6}Z\z/
      or fail("packet is outside the $route_name evidence boundary: $packet");
  -d $packet && !-l $packet
      or fail("invalid packet directory: $packet");

  my @run_order = qw(route status revision dirty generator compiler
                      compiler_sha256 host profile artifact artifact_sha256
                      source_manifest_kind source_manifest_sha256
                      source_identity_sha256 workload:status workload:exit
                      proof:kind proof:route proof:status proof:profile
                      proof:metrics proof:log proof:log:sha256 proof:result
                      proof:result:sha256);
  my $run = "$packet/run.tsv";
  my $field = read_fields($run, \@run_order);
  $field->{route} eq $route_name or fail("packet route mismatch: $packet");
  $field->{status} =~ /\A(?:passed|failed)\z/
      or fail("packet status is invalid: $packet");
  $field->{revision} ne '' or fail("packet revision is empty: $packet");
  $field->{dirty} =~ /\A(?:true|false)\z/
      or fail("packet dirty state is invalid: $packet");
  $field->{generator} ne '' && $field->{compiler} ne ''
      or fail("packet toolchain is empty: $packet");
  $field->{compiler_sha256} =~ /\A[0-9a-f]{64}\z/
      or fail("packet compiler identity is invalid: $packet");
  -f $field->{compiler} &&
      sha256($field->{compiler}) eq $field->{compiler_sha256}
      or fail("packet compiler artifact mismatch: $packet");
  $field->{host} eq host_text()
      or fail("packet host mismatch: $packet");
  $field->{profile} eq $profile
      or fail("packet profile mismatch: $packet");
  $field->{artifact} eq $spec->{artifact} &&
      $field->{artifact_sha256} =~ /\A[0-9a-f]{64}\z/
      or fail("packet artifact identity is invalid: $packet");
  $field->{source_manifest_kind} eq 'verification_after'
      or fail("packet source manifest kind changed: $packet");
  $field->{source_manifest_sha256} =~ /\A[0-9a-f]{64}\z/ &&
      $field->{source_identity_sha256} =~ /\A[0-9a-f]{64}\z/
      or fail("packet source identity is invalid: $packet");
  $field->{'workload:status'} eq 'passed' &&
      $field->{'workload:exit'} eq '0'
      or fail("calibration workload failed: $packet");
  $field->{'proof:kind'} eq 'performance' &&
      $field->{'proof:route'} eq $route_name &&
      $field->{'proof:status'} =~ /\A(?:passed|failed)\z/ &&
      $field->{status} eq $field->{'proof:status'}
      or fail("packet comparison status mismatch: $packet");
  $field->{'proof:log'} eq $spec->{log} &&
      $field->{'proof:result'} eq 'baseline.log'
      or fail("packet proof names changed: $packet");
  if ($field->{'proof:status'} eq 'passed') {
    $field->{'proof:profile'} eq $profile &&
        $field->{'proof:metrics'} eq "$spec->{metrics}"
        or fail("packet passed proof metadata changed: $packet");
  } else {
    $field->{'proof:profile'} eq '-' && $field->{'proof:metrics'} eq '0'
        or fail("packet failed proof metadata changed: $packet");
  }

  my %allowed = map { $_ => 1 }
      qw(run.tsv source-manifest.tsv source-identity.tsv baseline.log),
      $spec->{log};
  opendir my $directory, $packet
      or fail("cannot read packet directory $packet: $!");
  my @files = grep { $_ ne '.' && $_ ne '..' } readdir $directory;
  closedir $directory or fail("cannot close packet directory $packet: $!");
  for my $name (@files) {
    $allowed{$name} or fail("unknown packet entry $name: $packet");
    my $path = "$packet/$name";
    -f $path && !-l $path or fail("invalid packet entry $name: $packet");
  }
  for my $name (qw(run.tsv source-manifest.tsv source-identity.tsv baseline.log),
                  $spec->{log}) {
    -f "$packet/$name" && !-l "$packet/$name"
        or fail("missing immutable packet file $name: $packet");
  }

  my $source = "$packet/source-manifest.tsv";
  my $identity = "$packet/source-identity.tsv";
  my $log = "$packet/$spec->{log}";
  my $proof = "$packet/baseline.log";
  -s $source && -s $identity && -s $log && -s $proof
      or fail("immutable packet contains an empty payload: $packet");
  sha256($source) eq $field->{source_manifest_sha256}
      or fail("retained source manifest mismatch: $packet");
  sha256($identity) eq $field->{source_identity_sha256}
      or fail("retained source identity mismatch: $packet");
  sha256($log) eq $field->{'proof:log:sha256'}
      or fail("packet raw-log proof mismatch: $packet");
  sha256($proof) eq $field->{'proof:result:sha256'}
      or fail("packet comparison proof mismatch: $packet");

  if ($route_name eq 'measure-telemetry') {
    open my $telemetry, '<', $log
        or fail("cannot read telemetry packet identity: $packet");
    my $schema_row = <$telemetry> // '';
    my $manifest_row = <$telemetry> // '';
    my $artifact_row = <$telemetry> // '';
    close $telemetry
        or fail("cannot close telemetry packet identity: $packet");
    chomp($schema_row, $manifest_row, $artifact_row);
    $schema_row eq "telemetry\t1" &&
        $manifest_row eq
            "identity\tmanifest\tsha256\t$field->{source_manifest_sha256}" &&
        $artifact_row eq
            "identity\tartifact\tsha256\t$field->{artifact_sha256}"
        or fail("telemetry packet identity mismatch: $packet");
  }

  my @identity_order = qw(source_manifest_sha256 revision dirty);
  my $source_field = read_fields($identity, \@identity_order);
  $source_field->{source_manifest_sha256} eq $field->{source_manifest_sha256} &&
      $source_field->{revision} eq $field->{revision} &&
      $source_field->{dirty} eq $field->{dirty}
      or fail("source identity projection mismatch: $packet");

  if ($field->{'proof:status'} eq 'passed') {
    open my $proof_input, '<', $proof
        or fail("cannot read performance proof: $packet");
    my @lines = <$proof_input>;
    close $proof_input or fail("cannot close performance proof: $packet");
    chomp @lines;
    @lines == 1 &&
        $lines[0] eq "baseline\t$route_name\tprofile=$profile\tmetrics=$spec->{metrics}\tstatus=passed"
        or fail("passed performance proof changed: $packet");
  }

  return {
    path => $packet, log => $log, manifest => $field->{source_manifest_sha256},
    identity => $field->{source_identity_sha256},
    revision => $field->{revision}, dirty => $field->{dirty},
    generator => $field->{generator}, compiler => $field->{compiler},
    compiler_sha256 => $field->{compiler_sha256},
    artifact => $field->{artifact}, artifact_sha256 => $field->{artifact_sha256},
    log_sha256 => $field->{'proof:log:sha256'}, profile => $profile,
  };
}

sub adjacent {
  my ($root, $route_name, @packets) = @_;
  @packets == 3 or fail("$route_name requires three calibration packets");
  my $parent = "$root/.cache/evidence/$route_name";
  opendir my $directory, $parent
      or fail("cannot read $route_name evidence boundary: $!");
  my @canonical = sort grep { /\A[0-9]{8}T[0-9]{6}Z\z/ } readdir $directory;
  closedir $directory
      or fail("cannot close $route_name evidence boundary: $!");
  my %position;
  @position{@canonical} = (0 .. $#canonical);
  my @paths = map { $_->{path} } @packets;
  $paths[0] lt $paths[1] && $paths[1] lt $paths[2]
      or fail("$route_name calibration packets are not chronological");
  my @positions = map { $position{basename($_)} } @paths;
  defined($positions[0]) && defined($positions[1]) && defined($positions[2]) &&
      $positions[1] == $positions[0] + 1 &&
      $positions[2] == $positions[1] + 1
      or fail("$route_name calibration packets are not adjacent");
}

1;
