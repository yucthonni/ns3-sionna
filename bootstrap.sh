#!/bin/bash

echo "Run protoc"
cd ./common
protoc --cpp_out=../ns3-sionna/lib/message.proto
protoc --python_out=../sionna_server/message.proto

echo "Create symlink"
cd /ns3-workspace/scratch/
ln -s /ns3-workspace/contrib/ns3sionna/ns3-sionna/ .

cd ..
 
