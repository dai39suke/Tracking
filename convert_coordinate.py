# -*- coding: utf-8 -*-
import os
import re
from lxml import etree
from change_path import walk_files_with

source_dir = "ds/xml/"
bnd_dir = "object/bndbox/"

"""
X:35 Y:90 +する
但し，Ymaxが0のものは，元々どこに座標があったかが不明なため，オブジェクトタグごと削除する
"""

X_LOSS = 35
Y_LOSS = 90
CUT_WIDTH = 8*2 # 追加分の小さい動画
CUT_HEIGHT = 16*2 # 追加分の小さい動画

# <path>タグ(source_tag)の中身を書き換える
def coordinate_rewrite(t, source_dir, source_tag):
	xml_files = target_walk_files_with(t, "xml", source_dir) # 拡張子, ディレクトリ
	# print([xml_file for xml_file in xml_files])
	for xml_file in xml_files:
		print(xml_file)
		tree = etree.parse(xml_file)
		# directory, fname = os.path.split(tree.find(source_tag).text)
		# print(tree.find(source_tag).text)
		objs  = tree.findall("object")
		xmins = tree.findall(source_tag+"xmin")
		ymins = tree.findall(source_tag+"ymin")
		xmaxs = tree.findall(source_tag+"xmax")
		ymaxs = tree.findall(source_tag+"ymax")
		for i in range(len(xmins)): # findallで全ての要素をリストで返す
			if delete_check(int(xmaxs[i].text), int(ymaxs[i].text)) == 0:
				xmn, ymn, xmx, ymx = update_obj(xmins[i], ymins[i], xmaxs[i], ymaxs[i])
				overwrite_obj(tree, i, source_tag+"xmin", xmn)
				overwrite_obj(tree, i, source_tag+"ymin", ymn)
				overwrite_obj(tree, i, source_tag+"xmax", xmx)
				overwrite_obj(tree, i, source_tag+"ymax", ymx)
		for i in range(len(xmins)):
			if delete_check(int(xmaxs[i].text), int(ymaxs[i].text)):
				delete_obj(objs[i])
		with open(xml_file, 'wb') as f:
			f.write(etree.tostring(tree, xml_declaration=True, encoding='utf-8'))
		# delete_empty_obj(xml_file)

# 書き換えるべき要素かどうかチェック
def delete_check(xmax, ymax):
	print("@delete_check: ", xmax, ymax)
	return 1 if (ymax == 0) or (xmax == 0) else 0

# オブジェクトの要素を消去する
def delete_obj(source_tag):
	# source_tag.clear()
	source_tag.getparent().remove(source_tag)
	# print("@delete_obj: ", etree.tostring(source_tag))

# オブジェクトの要素を更新値を求める
# yminが0の場合は，バウンディングボックスのサイズが32になるように調節する
def update_obj(xmin, ymin, xmax, ymax):
	print()
	print("Xmin: {0}\t->{1}".format(xmin.text, int(xmin.text)+X_LOSS))
	print("Ymin: {0}\t->{1}".format(ymin.text, int(ymin.text)+Y_LOSS))
	print("Xmax: {0}\t->{1}".format(xmax.text, int(xmax.text)+X_LOSS))
	print("Ymax: {0}\t->{1}".format(ymax.text, int(ymax.text)+Y_LOSS))
	print()
	xmn = str(int(xmin.text)+X_LOSS) if int(xmin.text) != 0 else str(int(xmax.text)+X_LOSS-CUT_WIDTH)
	ymn = str(int(ymin.text)+Y_LOSS) if int(ymin.text) != 0 else str(int(ymax.text)+Y_LOSS-CUT_HEIGHT)
	xmx = str(int(xmax.text)+X_LOSS)
	ymx = str(int(ymax.text)+Y_LOSS)
	return xmn, ymn, xmx, ymx

# オブジェクトの要素を更新する
def overwrite_obj(tree, n, tag, upv):
	tree.findall(tag)[n].text = upv

# targetと名のつくextentionという拡張子のfileを全て取得
def target_walk_files_with(target, extension, directory='.'):
	for root, dirnames, filenames in os.walk(directory):
		for filename in filenames:
			if target in filename and filename.lower().endswith('.' + extension):
				yield os.path.join(root, filename)

if __name__ == "__main__":
	# for fn in target_walk_files_with("2017", "xml", source_dir):
	# 	print(fn)
	coordinate_rewrite("2017", source_dir, bnd_dir)
