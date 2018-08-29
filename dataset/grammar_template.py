#!/usr/bin/env python
# -*- coding:utf-8 -*-

bos = "S"
terminal = set(["r","s","w"])

simple_pcfg = """
s_0	s_1	s_2	prob
S	P		1
P	Pr		0.1
P	Ps		0.1
P	w		0.1
P	P	Pr	0.2
P	P	Ps	0.2
P	P	w	0.3
Pr	r	w	0.8
Pr	r	Pr	0.1
Pr	r	Ps	0.1
Ps	s	w	0.8
Ps	s	Ps	0.1
Ps	s	Pr	0.1
"""