# -*- coding: utf-8 -*-
import os
import csv
import pandas as pd

# CSV_PATH = "./csv/action.csv"
# mins = (xmin, ymin)
# maxs = (xmax, ymax)

"""
DataFrame内の第一引数のリストとcolumnを調整
"""

def write_csv(CSV_PATH, VIDEO_DATA, frame_x, frame_y, frame_num, mins, maxs, category):
	# if csv の中身が空 -> ヘッダーを追加
	CSV_MOVIE_PATH = "/mnt/storage/clients/rakuten/all_games_high_resolution" + "/" + VIDEO_DATA.split("/")[-1]
	if not os.path.exists(CSV_PATH): # はじめ
		df = pd.DataFrame([[CSV_MOVIE_PATH, frame_x, frame_y, frame_num,
		mins[0], mins[1], maxs[0], maxs[1], category]],
		columns=["movie_path", "frame_x", "frame_y", "frame_num", "xmin", "ymin", "xmax", "ymax", "category"])#, index = [1])
		print("File: {0} made...\n{1}".format(CSV_PATH, df))
		df.to_csv(CSV_PATH, index=False)
	else: # 追記
		df = pd.DataFrame([[CSV_MOVIE_PATH, frame_x, frame_y, frame_num, mins[0], mins[1], maxs[0], maxs[1], category]],
						  columns=["movie_path", "frame_x", "frame_y", "frame_num", "xmin", "ymin", "xmax", "ymax", "category"])
		df.to_csv(CSV_PATH, index=False, encoding="utf-8", mode="a", header=False) # mode="a"で追記 lineterminatorを書いとかないと，次のセルからになる？
		print(df)

if __name__ == "__main__":
	# VIDEO_DATA = './movie_full/20170510E-M-Fi.mpg'
	# write_csv(VIDEO_DATA, 480, 720, 1, [15, 20], [23, 36], 0)
	print(pd.read_csv("./csv/action.csv", index_col=0))
