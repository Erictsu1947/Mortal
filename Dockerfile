# syntax=docker/dockerfile:1.4-labs

# FROM archlinux:base-devel as libriichi_build
FROM rust:1.67 as libriichi_build


WORKDIR /
COPY exe-wrapper /exe-wrapper
COPY Cargo.toml Cargo.lock .
COPY libriichi libriichi

COPY <<'EOF' $HOME/.cargo/config
[source.crates-io]
registry = "https://github.com/rust-lang/crates.io-index"
# 指定镜像
replace-with = 'tuna' # 如：tuna、sjtu、ustc，或者 rustcc

# 清华大学
[source.tuna]
registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"

EOF

RUN cargo build -p libriichi --lib --release

# -----
FROM archlinux:base

RUN <<EOF
pacman -Syu --noconfirm --needed python python-pytorch python-toml python-tqdm tensorboard
pacman -Scc
EOF

WORKDIR /mortal
COPY mortal .
COPY --from=libriichi_build /target/release/libriichi.so .

ENV MORTAL_CFG config.toml
COPY <<'EOF' config.toml
[control]
state_file = '/mnt/mortal.pth'

[resnet]
conv_channels = 192
num_blocks = 40
enable_bn = true
bn_momentum = 0.99
EOF

VOLUME /mnt

ENTRYPOINT ["python", "mortal.py"]
