FROM nvidia/cuda:11.8.0-base-ubuntu20.04

# Add DW apt-get proxy
RUN touch /etc/apt/apt.conf.d/proxy.conf \
   && echo 'Acquire {HTTP::proxy "http://proxy.dwelle.de:8081/";}' >> /etc/apt/apt.conf.d/proxy.conf
# Add DW proxy for other packages (e.g. wget)
ENV HTTP_PROXY="http://proxy.dwelle.de:8081/"

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# # Create a non-root user and switch to it
# RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
#  && chown -R user:user /app
# RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
# USER user

ARG USERNAME=web
ARG UID=15004
ARG GROUPNAME=domainusers
ARG GID=10063

#Add group and user
RUN groupadd -g "${GID}" "${GROUPNAME}" && useradd -u "${UID}" "${USERNAME}" -g "${GROUPNAME}" -m \
    && chown -R "${USERNAME}":"${GROUPNAME}" /app

# COPY  --chown="${UID}":"${GID}" ./src ./src
RUN mkdir -p /app/cache && \
    chown -R "${USERNAME}":"${GID}" /app/cache

# All users can use /home/user as their home directory
ENV HOME=/home/"${USERNAME}"
RUN chmod 777 /home/"${USERNAME}"

ENV TRANSFORMERS_CACHE=/app/cache
RUN chmod 777 /app/cache

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/home/"${USERNAME}"/miniconda/bin:$PATH

COPY environment.yml /app/environment.yml

COPY requirements.txt /app/requirements.txt



RUN wget -q -O ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda env update -n base -f /app/environment.yml \
 && rm /app/environment.yml \
 && pip install -r requirements.txt \
 && conda clean -ya


CMD ["/bin/bash"]