#!/bin/bash

# Authors: Pratik Bhatu.

# Copyright:
# Copyright (c) 2021 Microsoft Research
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Virtual environment 
sudo apt install -y python3.7-venv
python3.7 -m venv mpc_venv
source mpc_venv/bin/activate
pip install -U pip
pip install tensorflow==1.15.0 keras==2.3.0 scipy==1.1.0 matplotlib scikit-learn==0.24.2 torchvision
pip install onnx onnx-simplifier onnxruntime black
pip install torchvision==0.13.1 idx2numpy==1.2.3
pip install pytest pytest-cov 
pip install protobuf==3.20.1 numpy==1.21.0 # onnxBridge中的
python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com


build_boost () {
  sudo apt-get -y install python3.7-dev autotools-dev libicu-dev libbz2-dev
  echo "Building and installing boost from source"
  wget https://boostorg.jfrog.io/artifactory/main/release/1.67.0/source/boost_1_67_0.tar.gz
  tar -zxvf boost_1_67_0.tar.gz
  cd boost_1_67_0
  ./bootstrap.sh
  ./b2 -j $(nproc)
  sudo ./b2 install
  sudo ldconfig
  cd ..
  rm -rf boost_1_67_0.tar.gz boost_1_67_0
}

BOOST_REQUIRED_VERSION="1.66"
if dpkg -s libboost-dev >/dev/null; then
  BOOST_VERSION=$(dpkg -s libboost-dev | grep 'Version' | grep -oE '[0-9]+.[0-9]+(\.)*[0-9]*')
  LATEST_VERSION=$(printf "$BOOST_VERSION\n$BOOST_REQUIRED_VERSION\n" | sort | tail -n1)
  if [[ "$BOOST_VERSION" == "$LATEST_VERSION" ]]; then
    echo "Boost already installed.."
  else
    sudo apt purge libboost-all-dev -y
    build_boost
  fi
else
  build_boost
fi
