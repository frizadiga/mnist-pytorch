#!/usr/bin/env bash
# alias: `n/a`
# desc: fn_sym_model description.
# usage: fn_sym_model.sh [args]
# flags: @WIP:0 @TODO:0 @FIXME:0 @BUG:0 @OPTIMIZE:0 @REFACTOR:0 @DEPRECATED:0

set -e -u -o pipefail
# set -x # uncomment to debug

declare -r __self_path_file=$(readlink -f "$0")
declare -r __self_path_dir=$(dirname "${__self_path_file}")

fn_sym_model() {
  local models_dir="${__self_path_dir}/models"
  echo "[INFO] dist_path: ${models_dir}"

  local latest_model=$(
    ls -1t "${models_dir}"/*.pth 2>/dev/null | grep -v "_latest" | head -1
  )
  echo "[INFO] latest_model: ${latest_model}"

  if [[ -z "${latest_model}" ]]; then
    echo "[ERROR] No pth files found in ${models_dir}"
    exit 1
  fi

  ln -sfv "${latest_model}" "${models_dir}/model_latest.pth" || {
    echo "[ERROR] Failed to create symlink for latest model"
    exit 1
  }
}

fn_sym_model "$@"
