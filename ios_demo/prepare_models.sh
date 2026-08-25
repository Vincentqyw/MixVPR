#!/bin/sh
# Compiles the CoreML packages into the app bundle and (re)generates the Xcode project.
# Requires: Xcode, xcodegen (brew install xcodegen).
#
#   MixVPR  : ../coreml_models/mixvpr_{fp16,int8}.mlpackage       (HF: mixvpr/coreml/)
#   MegaLoc : ../../megaloc/coreml_ios_322/megaloc_ane{,_int8}.mlpackage
#             (megaloc/export_ios.py --size 322 --variants ane; MEGALOC_SIZE=518 picks coreml_ios/)
# Missing MegaLoc packages are skipped — the app only lists models that are bundled.
# NOTE: VPRModel.inputSize in Models.swift must match MEGALOC_SIZE.
set -e
cd "$(dirname "$0")"
MEGALOC_SIZE="${MEGALOC_SIZE:-322}"
[ "$MEGALOC_SIZE" = "518" ] && DEFAULT_DIR=../../megaloc/coreml_ios || DEFAULT_DIR=../../megaloc/coreml_ios_$MEGALOC_SIZE
MEGALOC_DIR="${MEGALOC_DIR:-$DEFAULT_DIR}"
mkdir -p MixVPRDemo/Models

compile() {  # compile <src.mlpackage> <bundle name>
  rm -rf "MixVPRDemo/Models/$2.mlmodelc"
  tmp=$(mktemp -d)
  xcrun coremlcompiler compile "$1" "$tmp" >/dev/null
  mv "$tmp"/*.mlmodelc "MixVPRDemo/Models/$2.mlmodelc"
  rm -rf "$tmp"
  echo "  $2.mlmodelc  <-  $1"
}

for m in mixvpr_fp16 mixvpr_int8; do
  src="../coreml_models/$m.mlpackage"
  if [ ! -d "$src" ]; then
    echo "missing $src — download with:"
    echo "  huggingface-cli download Realcat/image_retrieval_checkpoints mixvpr/coreml/ --local-dir ../hf && cp -r ../hf/mixvpr/coreml/* ../coreml_models/"
    exit 1
  fi
  compile "$src" "$m"
done
[ -d "$MEGALOC_DIR/megaloc_ane.mlpackage" ]      && compile "$MEGALOC_DIR/megaloc_ane.mlpackage" megaloc_fp16      || echo "  (MegaLoc fp16 not found, skipped)"
[ -d "$MEGALOC_DIR/megaloc_ane_int8.mlpackage" ] && compile "$MEGALOC_DIR/megaloc_ane_int8.mlpackage" megaloc_int8 || echo "  (MegaLoc int8 not found, skipped)"

xcodegen generate
