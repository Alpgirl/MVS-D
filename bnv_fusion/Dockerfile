FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && yes|apt-get upgrade

RUN apt-get install -y wget bzip2
RUN apt-get -y install sudo

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion libgl1-mesa-dev && \
    apt-get clean

RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

CMD [ "/bin/bash" ]

RUN conda install pip

# install pytorch related
# RUN conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 cudatoolkit=11.8 -c pytorch
# RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.2+cu118.html

# # install some extra packages
# # RUN conda install -c conda-forge kornia
# RUN pip install -U kornia==0.5.0
# RUN	pip install pytorch-lightning==1.2.6
# RUN pip install hydra-core==1.1.1
# RUN pip install multidict
# RUN pip install pyquaternion
# RUN pip install pillow
# RUN pip install rich
# RUN pip install opencv-python
# RUN pip install -U scikit-learn
# RUN conda install -c conda-forge trimesh
# RUN pip install opencv-python
# RUN apt-get install ffmpeg libsm6 libxext6  -y
# RUN pip install numpy-quaternion
# RUN pip install plyfile
# RUN pip install --upgrade PyMCubes
# RUN python3 -m pip install --no-cache-dir --upgrade open3d==0.14.1 --ignore-installed PyYAML