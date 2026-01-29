#!/bin/bash
sudo apt update
sudo apt install zsh zsh-autosuggestions zsh-syntax-highlighting -y
echo "source /usr/share/zsh-autosuggestions/zsh-autosuggestions.zsh" >> ~/.zshrc
echo "source /usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh" >> ~/.zshrc
echo 'export HF_ENDPOINT="https://hf-mirror.com"' >> ~/.zshrc
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
source ~/.zshrc
echo "~/.zshrc as follows:"
cat ~/.zshrc
echo "pip config as follows:"
pip config list