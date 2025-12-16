# 1. 基础镜像：使用官方 Miniconda3
FROM continuumio/miniconda3:latest

# 设置工作目录
WORKDIR /app

# ------------------------------------------------------------------
# [优化 A] 系统底层依赖与 APT 源加速
# 目的：修复 XGBoost (libgomp1) 和 Graphviz 缺失问题；使用 USTC 源加速下载
# ------------------------------------------------------------------
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list.d/debian.sources && \
    sed -i 's|security.debian.org/debian-security|mirrors.ustc.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y \
    libgomp1 \
    graphviz \
    git \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# [优化 B] Conda 镜像源配置
# 目的：使用清华 TUNA 源加速 Python 包下载
# ------------------------------------------------------------------
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ && \
    conda config --set show_channel_urls yes

# 复制环境配置文件
COPY requirement.yml .

# ------------------------------------------------------------------
# [核心步骤] 创建环境与补充安装
# 1. 基于 requirement.yml 创建环境
# 2. 额外安装 jupyter (原配置仅含 ipykernel，需补充 notebook 服务)
# ------------------------------------------------------------------
RUN conda env create -f requirement.yml && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda activate mymodels && \
    conda install -y jupyter -c conda-forge && \
    conda clean -afy

# ------------------------------------------------------------------
# [配置] 环境变量与路径
# ------------------------------------------------------------------
# 确保默认使用 mymodels 环境的 Python 和 Jupyter
ENV PATH=/opt/conda/envs/mymodels/bin:$PATH
ENV PYTHONPATH=/app

# 复制项目代码到镜像中
COPY . .

# ------------------------------------------------------------------
# [数据持久化] 声明数据卷
# ------------------------------------------------------------------
VOLUME ["/app/results", "/app/data"]

# 暴露 Jupyter 默认端口
EXPOSE 8888

# ------------------------------------------------------------------
# [启动] 默认命令
# 启动 Jupyter Notebook，允许 root 运行，监听所有 IP
# ------------------------------------------------------------------
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

