# 🟣 PURPLE
Code for the paper **PURPLE: Making a Large Language Model a Better SQL Writer**.

**Overview**: PURPLE is a cutting-edge solution designed to enhance the capabilities of large language models in generating SQL queries efficiently and accurately.

## Dataset Download

- [Spider](https://drive.google.com/uc?export=download&id=1_AckYkinAnhqmRQtGsQgUKAnTHxxX5J0): `./datasets/spider`
- [Spider-DK](https://github.com/ygan/Spider-DK): `./datasets/spider_dk`
- [Spider-SYN](https://github.com/ygan/Spider-Syn): `./datasets/spider_syn`
- [Spider-Realistic](https://zenodo.org/record/5205322#.YTts_o5Kgab): `./datasets/spider_realistic`

Unzip the data and organize into the following format:
```
spider
├── database
├── dev.json
├── train_spider_pruned.json
└── tables.json
```

## Environment Build

We publish our docker image for easier experiments reproduction, you can achieve such a image by:

```shell
docker pull thren20/purple:v3
docker run -itd --rm --name YOUR_CONTAINER_NAME --mount type=bind,source=PATH_TO_YOUR_CODE,target=/workspace/ thren20/purple:v3
```

NOTE: The trained models are also included in the docker image.

Of course, you can build such an environment without docker, the packages are included in the requirements.txt. We offer an environment building script as `env.sh` for you:

```shell
chmod 744 env.sh
bash env.sh
```

## Pipeline

To reproduce the experiments in the paper, we prepare a script for that.

```shell
chmod 744 script/infer_pipeline.sh
bash script/infer_pipeline.sh
```