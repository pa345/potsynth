#!/bin/sh
#
# check.sh -- build and test potsynth in every supported configuration
#
#   release   default build (OpenMP, vector extensions)
#   shared    shared library (also runs the Python interface test)
#   generic   portable scalar kernels (POTSYNTH_NO_VECEXT)
#   asan      AddressSanitizer + UndefinedBehaviorSanitizer + leak check, no OpenMP
#   tsan      ThreadSanitizer, no OpenMP in the library (GCC's OpenMP runtime is
#             not instrumented); the concurrency test uses pthreads
#
# Usage: tests/check.sh [BUILDROOT [config...]]
# Environment: CMAKE (default: cmake), CTEST, CMAKE_ARGS (extra configure
# arguments, e.g. -DGSL_ROOT_DIR=...), GENERATOR (default: Ninja if available)

set -e

SRC=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${1:-$SRC/build-check}
[ $# -gt 0 ] && shift
CONFIGS=${*:-"release shared generic asan tsan"}
CMAKE=${CMAKE:-cmake}
CTEST=${CTEST:-$(dirname "$(command -v "$CMAKE")")/ctest}

if [ -z "$GENERATOR" ]; then
  if command -v ninja > /dev/null 2>&1; then GENERATOR=Ninja; else GENERATOR="Unix Makefiles"; fi
fi

mkdir -p "$ROOT"
status=0

for cfg in $CONFIGS; do
  dir=$ROOT/$cfg
  flags=""
  case $cfg in
    release) opts="-DCMAKE_BUILD_TYPE=Release" ;;
    shared)  opts="-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON" ;;
    generic) opts="-DCMAKE_BUILD_TYPE=Release -DPOTSYNTH_NO_VECEXT=ON" ;;
    asan)    opts="-DCMAKE_BUILD_TYPE=Debug -DPOTSYNTH_OPENMP=OFF"
             flags="-fsanitize=address,undefined -fno-omit-frame-pointer -O1" ;;
    tsan)    opts="-DCMAKE_BUILD_TYPE=Debug -DPOTSYNTH_OPENMP=OFF"
             flags="-fsanitize=thread -O1" ;;
    *)       echo "unknown configuration: $cfg"; exit 2 ;;
  esac

  echo "=== $cfg ==="
  rm -rf "$dir"
  # shellcheck disable=SC2086
  if ! CFLAGS="$flags" "$CMAKE" -S "$SRC" -B "$dir" -G "$GENERATOR" $opts $CMAKE_ARGS \
         > "$dir.configure.log" 2>&1; then
    cat "$dir.configure.log"
    status=1
    continue
  fi

  if ! "$CMAKE" --build "$dir" > "$dir.build.log" 2>&1; then
    cat "$dir.build.log"
    status=1
    continue
  fi
  grep -E "warning" "$dir.build.log" || true

  # ThreadSanitizer cannot cope with the address-space randomisation of recent
  # Linux kernels ("unexpected memory mapping"): run without ASLR
  run=""
  if [ "$cfg" = tsan ] && command -v setarch > /dev/null 2>&1; then
    run="setarch $(uname -m) -R"
  fi

  # shellcheck disable=SC2086
  if ! (cd "$dir" && UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
          TSAN_OPTIONS=halt_on_error=1 $run "$CTEST" --output-on-failure); then
    status=1
  fi
done

exit $status
