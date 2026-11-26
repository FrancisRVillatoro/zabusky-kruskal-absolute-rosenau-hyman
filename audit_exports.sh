#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p audit_reproduced
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for p in A B C; do unzip -q "archives/CNSNS_Fase9${p}_cierre_20261125.zip" -d "$tmp"; done
python "$tmp/CNSNS_Fase9A_cierre_20261125/audit.py" archives/Fase9A_HDF5_3831810.zip audit_reproduced/A
python "$tmp/CNSNS_Fase9B_cierre_20261125/audit.py" archives/Fase9B_CNSNS_resultados_20261125.zip --out audit_reproduced/B
python "$tmp/CNSNS_Fase9C_cierre_20261125/audit9c.py" archives/Fase9C_CNSNS_resultados_20261125.zip --out audit_reproduced/C
