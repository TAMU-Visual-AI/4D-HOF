#!/usr/bin/env bash
# Download HO3D release assets and extract the initialization bundles.
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: bash download_assets.sh"
    echo "Download checkpoints and HO3D initialization from eggplant71/4D-HOF, verify checksums, and extract into _DATA."
    exit 0
fi
if (( $# != 0 )); then
    echo "Usage: bash download_assets.sh" >&2
    exit 1
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"
for program in hf sha256sum tar; do
    if ! command -v "$program" >/dev/null; then
        echo "Missing command: $program. Run setup_env.sh and activate the environment; sha256sum and tar must also be installed." >&2
        exit 1
    fi
done

printf '\nDownloading HO3D initialization and checkpoints\n'
hf download eggplant71/4D-HOF --repo-type model \
    --include "_DATA/*" "checkpoints/*" --local-dir .

printf '\nVerifying downloaded files\n'
sha256sum -c _DATA/SHA256SUMS

printf '\nExtracting HO3D initialization\n'
mkdir -p _DATA
tar -xzf _DATA/ho3dv3_init.tar.gz -C _DATA
rm -- _DATA/ho3dv3_init.tar.gz

printf '\nReady: checkpoints/4dhof.pth and _DATA/ho3dv3_init/\n'
