#!/usr/bin/env bash
# initialize.sh
# Bootstrap script for new PF-AGCN VM instances.
# - Clones or updates the PF-AGCN repository.
# - Synchronizes cached datasets and artifacts from GCS.

set -euo pipefail

REPO_URL="https://github.com/yzhong5021/pf-agcn-extension.git"
REPO_DIR="${PF_AGCN_REPO_DIR:-pf_agcn}"
GCS_CACHE_URI="${PF_AGCN_GCS_CACHE_URI:-}"
LOCAL_CACHE_DIR="${PF_AGCN_LOCAL_CACHE_DIR:-/mnt/disks/nvme/cache}"

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    log "Error: required command '$1' not found in PATH."
    exit 1
  fi
}

log "Validating prerequisites..."
require_command git
require_command gsutil

if [[ -z "${GCS_CACHE_URI}" ]]; then
  log "Error: PF_AGCN_GCS_CACHE_URI is unset. Provide the source GCS URI (e.g. gs://pf-agcn-cache)."
  exit 1
fi

log "Ensuring repository is present at '${REPO_DIR}'."
if [[ -d "${REPO_DIR}/.git" ]]; then
  log "Repository exists. Fetching latest changes."
  git -C "${REPO_DIR}" fetch --all --tags
  git -C "${REPO_DIR}" pull --ff-only
else
  log "Cloning repository from ${REPO_URL}."
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

log "Preparing local cache directory at '${LOCAL_CACHE_DIR}'."
mkdir -p "${LOCAL_CACHE_DIR}"

log "Syncing cached assets from ${GCS_CACHE_URI} -> ${LOCAL_CACHE_DIR}."
gsutil -m rsync -r "${GCS_CACHE_URI}" "${LOCAL_CACHE_DIR}"

log "Initialization complete."
