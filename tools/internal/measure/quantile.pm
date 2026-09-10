package RundMeasureQuantile;

use strict;
use warnings;
use Math::BigFloat;

# Canonical signed integer strings have an exact order without conversion to
# machine floats or allocating an arbitrary-precision object per sample.
sub order {
  my ($left, $right) = @_;
  my $left_negative = substr($left, 0, 1) eq '-';
  my $right_negative = substr($right, 0, 1) eq '-';
  return $right_negative <=> $left_negative
      if $left_negative != $right_negative;
  my $magnitude = length($left) <=> length($right) || $left cmp $right;
  return $left_negative ? -$magnitude : $magnitude;
}

sub summarize {
  my ($samples) = @_;
  @$samples or die "integer quantile requires samples\n";
  for my $value (@$samples) {
    defined($value) && $value =~ /\A(?:0|-?[1-9][0-9]*)\z/
        or die "integer quantile requires canonical signed integers\n";
  }
  my @ordered = sort { order($a, $b) } @$samples;
  my $count = scalar @ordered;
  my $median = Math::BigFloat->new($ordered[int(($count - 1) / 2)]);
  if ($count % 2 == 0) {
    $median->badd($ordered[$count / 2]);
    $median->bdiv(2);
  }
  my $p95 = Math::BigFloat->new($ordered[int((95 * $count + 99) / 100) - 1]);
  return ($median, $p95);
}

1;
