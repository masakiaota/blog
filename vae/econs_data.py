"""
6.7.1ステートフルLSTMで電気消費の予測の前準備
"""
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import re

import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = "./data"

# 顧客番号(cid)250番の電気消費データを抽出
with open(os.path.join(DATA_DIR, "LD2011_2014.txt"), "r") as fld:
    data = []
    cid = 250
    for line_num, line in enumerate(fld):
        if line.startswith("\"\";"):
            # データを覗いてみたら一行目に対応していた。つまり一行目をスキップしているということ
            continue
        if line_num % 1000 == 0:
            print("{:d} lines read".format(line_num))
        cols = [float(re.sub(",", ".", x)) for x in
                line.strip().split(";")[1:]]  # .strip()は前後の空白削除
        data.append(cols[cid])

NUM_ENTRIES = 1000

# 最初の10日間のデータをプロット
plt.plot(range(NUM_ENTRIES), data[0:NUM_ENTRIES])
plt.ylabel("electricity consumption")
plt.xlabel("time (1pt = 15 mins)")
plt.show()

np.save(os.path.join(DATA_DIR, "LD_250.npy"), np.array(data))
