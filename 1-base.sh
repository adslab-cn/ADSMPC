# 基础环境安装

# 获取输入参数
mode=$1

# If 2nd argument is provided, then SCI build will be modified. See SCI readme.
NO_REVEAL_OUTPUT=$2 # 只和SCI库有关

# 添加PPA源并安装依赖项
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo add-apt-repository ppa:avsm/ppa -y
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install -y build-essential cmake make libgmp-dev libglib2.0-dev libssl-dev \
                    libboost-all-dev m4 python3.7 opam unzip bubblewrap \
                    graphviz tmux bc time
# 原MPC-GPU代码中安装的包
sudo apt install -y python3-pip libmpfr-dev libeigen3-dev;

# Install gcc 9
echo "Installing g++-9"
sudo apt install -y gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
sudo update-alternatives --config gcc

build_cmake () {
  echo "Building and installing cmake from source"
  wget https://github.com/Kitware/CMake/releases/download/v3.13.4/cmake-3.13.4.tar.gz
  tar -zxvf cmake-3.13.4.tar.gz
  cd cmake-3.13.4
  sudo ./bootstrap
  sudo make
  sudo make install
  cd ..
  rm -rf cmake-3.13.4 cmake-3.13.4.tar.gz
}

if which cmake >/dev/null; then
  CMAKE_VERSION=$(cmake --version | grep -oE '[0-9]+.[0-9]+(\.)*[0-9]*')
  LATEST_VERSION=$(printf "$CMAKE_VERSION\n3.13\n" | sort | tail -n1)
  if [[ "$CMAKE_VERSION" == "$LATEST_VERSION" ]]; then
    echo "CMake already installed.."
  else
    sudo apt purge cmake
    build_cmake
  fi
else
  build_cmake
fi

# 从GitHub上下载opam的安装脚本, 并安装
wget "https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh"
if [ $? -ne 0 ]; then
  echo "Downloading of opam script failed"; exit
fi

chmod +x install.sh
if [[ "$mode" == "quick" ]]; then
	yes "" | ./install.sh
else
	./install.sh
fi
if [ $? -ne 0 ]; then
  rm install.sh
  echo "Opam installation failed"; exit
fi
rm install.sh

# environment setup
if [[ "$mode" == "quick" ]]; then
	yes "" | opam init --disable-sandboxing
else 
	opam init --disable-sandboxing
fi
if [ $? -ne 0 ]; then
  echo "opam init failed"; exit
fi

# install given version of the compiler
eval `opam env`
if [[ "$mode" == "quick" ]]; then
	yes "" | opam switch create 4.10.0
else
	opam switch create 4.10.0
fi
opam switch list | grep "4.10.0" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "opam switch create 4.10.0 failed"; exit
fi
eval `opam env`
# check if we got what we wanted
which ocaml
ocaml -version
opam install -y Stdint
opam install -y menhir
opam install -y ocamlbuild 
opam install -y ocamlfind
