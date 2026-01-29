#!/bin/bash
sudo apt update
sudo apt install -y ca-certificates git gnupg
export https_proxy=http://127.0.0.1:7890;export http_proxy=http://127.0.0.1:7890;export all_proxy=socks5://127.0.0.1:7890
git clone https://github.com/git-ecosystem/git-credential-manager.git
wget -qO- https://aka.ms/gcm/linux-install-source.sh | sudo bash
git-credential-manager configure
git config --global credential.credentialStore secretservice
sudo rm -rf git-credential-manager
echo "git config as follows:"
# git config --global --list
cat ~/.gitconfig