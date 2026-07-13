# Epilepsee-AI Dockerfile for portable AI training
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /workspace

# Copy environment.yml and install Conda dependencies
COPY environment.yml ./
RUN conda env create -f environment.yml && conda clean -afy

# Activate environment by default
SHELL ["/bin/bash", "-c"]
ENV PATH /opt/conda/envs/epilepsee-ai/bin:$PATH

# Copy project files
COPY . .

# Install pip dependencies (if any)
RUN conda run -n epilepsee-ai pip install --no-cache-dir .

# Set default command to bash (can override for training)
CMD ["bash"]
