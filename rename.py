# -*- coding: utf-8 -*-

import os

l = []
def rename_files(path):
	for (root, dirs, files) in os.walk(path):
		for f in files:
			changed = int(f.split(".")[0]) + 53
			os.rename(os.path.join(path, f), os.path.join(path, str(changed)+".mp4"))
	for (root, dirs, files) in os.walk(path):
		for f in files:
			print(f)

if __name__=="__main__":
	rename_files("./0630")
