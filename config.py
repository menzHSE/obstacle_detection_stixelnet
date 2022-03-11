#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


class Config(object):
    def __init__(self):
        self.CURRENT_DIR = _CURRENT_DIR

        self.DATA_PATH = os.path.abspath(os.path.join(_CURRENT_DIR, "data", "StixelNet_Waymo"))

        # Stixel ground truth from the following third party dataset:
        # https://sites.google.com/view/danlevi/datasets
        self.GROUND_TRUTH_PATH = os.path.join(
            self.DATA_PATH, "waymo_train.txt"
        )

        self.SAVED_MODELS_PATH = "saved_models"
        if not os.path.isdir(self.SAVED_MODELS_PATH):
            os.system("mkdir -p {}".format(self.SAVED_MODELS_PATH))

        self.NUM_EPOCHS = 50

    def display(self):
        """
        Display Configuration values.
        """
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
