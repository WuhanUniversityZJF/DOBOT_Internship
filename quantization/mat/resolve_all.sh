#! /usr/bin bash
# Copyright (c) 2023 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.
set -e

SCRIPTS_DIR=$(readlink -f "$(dirname "$0")")
cd "$SCRIPTS_DIR" || exit

sh -e ./package/host/resolve.sh
sh -e ./samples/ai_toolchain/model_zoo/runtime/ai_benchmark/resolve_ai_benchmark_ptq.sh
sh -e ./samples/ai_toolchain/model_zoo/runtime/ai_benchmark/resolve_ai_benchmark_qat.sh
sh -e ./samples/ai_toolchain/model_zoo/runtime/horizon_runtime_sample/resolve_runtime_sample.sh

find ./samples/ai_toolchain/horizon_model_convert_sample -type f -name "00_init.sh" -exec sh {} \;
