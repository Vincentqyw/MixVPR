#!/bin/sh
# Compiles the CoreML packages from ../coreml_models into the app bundle and
# (re)generates the Xcode project. Requires: Xcode, xcodegen (brew install xcodegen).
set -e
cd "$(dirname "$0")"
mkdir -p MixVPRDemo/Models
for m in mixvpr_fp16 mixvpr_int8; do
  src="../coreml_models/$m.mlpackage"
  if [ ! -d "$src" ]; then
    echo "missing $src — download with:"
    echo "  huggingface-cli download Realcat/image_retrieval_checkpoints mixvpr/coreml/ --local-dir ../hf && cp -r ../hf/mixvpr/coreml ../coreml_models"
    exit 1
  fi
  rm -rf "MixVPRDemo/Models/$m.mlmodelc"
  xcrun coremlcompiler compile "$src" MixVPRDemo/Models/
done
xcodegen generate
