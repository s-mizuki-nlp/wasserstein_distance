#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os, sys, io
import argparse

wd = os.path.dirname(__file__)
wd = "." if wd == "" else wd
os.chdir(wd)

parser = argparse.ArgumentParser(description="hoge")
parser.add_argument("ARG_FULL", "ARG_SHORT", required=(True | False), [type = TYPE,] [default = None,] [
    choices = LIST,] help = "MESSAGE")
args = parser.parse_args()
