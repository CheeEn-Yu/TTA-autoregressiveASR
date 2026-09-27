#!/usr/bin/env bash
# Table 2 entry point; uses the shared runner in this repository.
set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/scripts/main_tables.sh"
run_main_table 2 "$@"
