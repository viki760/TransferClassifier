from distutils.core import setup

setup(
name="Transfer Classifier",
version="1.0",
description="using fg-net finetuning to classify Cifar10",
author="viki",
py_modules=["trainer.train_s","trainer.transfer"]
)
