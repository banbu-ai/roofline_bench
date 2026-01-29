# 1. Miniconda 

运行`install_miniconda.sh`

==！所有脚本请使用 bash 运行 因为 jetson sh 默认指向 dash==

```bash
bash install_miniconda.sh
```

# 2. Repository

```bash
cd ~
mkdir Code
cd Code
git clone https://github.com/ggml-org/llama.cpp.git
git clone https://github.com/banbu-ai/llm_inference_roofline_detect.git
```

如果需要在 jetson 上配置 GCM（记录 git token）

```bash
bash install_GCM.sh
```

# 3. venv

```bash
cd ~/Code/llm_inference_roofline_detect
conda create -n llm python=3.8
conda activate llm
pip install -r requirements.txt
cd ~/Code/llama.cpp
conda create -n llama.cpp python=3.8
conda activate llama.cpp
pip install -r requirements.txt
```

==！代码位置和我写的保持一致 避免修改代码==

==！第一个环境名字随便 第二个一定要叫 llama.cpp==

# 4. llama.cpp

```bash
cd ~/Code/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

jetson 自带 cuda 直接编译就行

但是由于性能不强编译很慢 可以用我编译好的版本

配置 llama.cpp 环境变量

（为 bash 和 zsh都配置了环境变量 个人习惯使用 zsh）

```bash
bash set_llama.cpp_path_jetson.sh
```

# 5. bash_profile

script 里都用的 `source ~/.bash_profile`

Linux 和 macOS 对于`~/.bashrc` `~/bash_profile`加载顺序略有差异

因为 mac 上和 bash 相关配置都写在`~/.bash_profile`下了

最简单的方法是在 jetson 上新建`~/bash_profile`

然后加载`~/.bashrc`

运行如下命令行即可

```bash
echo 'if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi' > ~/.bash_profile
```

# 6. VSCode

如果需要使用 VSCode

~~或者不需要可以使用 vim~~

```bash
wget https://code.visualstudio.com/sha/download?build=stable&os=linux-deb-arm64
```

然后 `dpkg -i xxx.deb`安装一下

# 7. zsh & oh-my-zsh & zsh-autosuggestions & zsh-syntax-highlighting（非必须）

==非必须 只是比较好用==

`oh-my-zsh`：zsh 增强

`zsh-autosuggestions`：zsh 可以根据历史记录自动补全

`zsh-syntax-highlighting`：zsh 高亮提示

安装 zsh & oh-my-zsh & zsh-autosuggestions & zsh-syntax-highlighting

```bash
bash set_custom_zsh.sh
```

安装 `oh-my-zsh`需要 terminal 代理（export 单次会话生效）

```bash
export https_proxy=http://127.0.0.1:7890;export http_proxy=http://127.0.0.1:7890;export all_proxy=socks5://127.0.0.1:7890
```
然后运行

```bash
bash install_oh_my_zsh.sh
```