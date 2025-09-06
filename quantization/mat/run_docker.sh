#!/bin/bash

dataset_path=$1
run_type=$2
version=v1.2.8
container_name=$(whoami)_OE_v1.2.8
host_name=$(echo "1.2.8" |awk -F "." '{ print $1"-"$2"-"$3 }')

if [ -z "$dataset_path" ];then
  echo "Please specify the dataset path"
  exit
fi
dataset_path=$(readlink -f "$dataset_path")

echo "Docker version is ${version}"
echo "Dataset path is $(readlink -f "$dataset_path")"

open_explorer_path=$(readlink -f "$(dirname "$0")")
echo "OpenExplorer package path is $open_explorer_path"

if [ "$run_type" == "cpu" ];then
    echo "Start Docker container in CPU mode."
    docker run -it --rm \
      --hostname "OE-X5-CPU-$host_name" \
      --name $container_name \
      -v "$open_explorer_path":/open_explorer \
      -v "$dataset_path":/data/horizon_x5/data \
      openexplorer/ai_toolchain_ubuntu_20_x5_cpu:"$version"
else
    echo "Start Docker container in GPU mode."
    docker run -it --rm \
      --hostname "OE-X5-GPU-$host_name" \
      --name $container_name \
      --gpus all \
      --shm-size="15g" \
      -v "$open_explorer_path":/open_explorer \
      -v "$dataset_path":/data/horizon_x5/data \
      openexplorer/ai_toolchain_ubuntu_20_x5_gpu:"$version"
fi
