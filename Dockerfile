FROM condaforge/miniforge3:24.3.0-0

RUN apt update && apt upgrade -y
RUN apt install -y bash coreutils wget curl

# Install mamba for faster dependency resolution
RUN conda install mamba -y -c conda-forge

ARG conda_env=pride-bluesky
WORKDIR /app

COPY environment.yml ./
#COPY config.ini ./
COPY *.py ./
SHELL ["/bin/bash", "-c"]
RUN mamba env create -n $conda_env -f environment.yml

RUN echo "conda activate $conda_env" >> ~/.bashrc

RUN source ~/.bashrc

# Add conda installation dir to PATH (instead of doing 'conda activate')
ENV PATH /opt/conda/envs/$conda_env/bin:$PATH

CMD [ "/bin/bash", "-c", "source ~/.bashrc && /bin/bash"]
ENTRYPOINT python bluesky_bot.py -c PRODUCTION