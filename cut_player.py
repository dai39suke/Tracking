# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
# from arg_parse import arg_parse
from XMLconvert import addframe as addframe_to_xml
from save_to_csv import write_csv

"""
# トラッキング
## 事前準備
	1. SHIFT_TIME: 開始する秒数を指定
	2. VIDEO_DATA: 読み込む動画のパス名を指定
	3. ROOT_DIR: XMLの<path></path>タグのパス変更
	4. LABEL_NUM: playerのデータセットを作る場合は1(そのまま)，offense等だと変更
	5. LABEL_NAME: データセットを格納するフォルダ名を指定(自動で作られる)
## 実行
	1. スクリプトを実行
	2. sで一時停止 rで再開 ESCで終了
	3. sで一時停止している状態で、トラッキング対象の選手をクリック
	4. 複数選手を同時に捕捉できる．再びクリックすれば，バウンディングボックスを削除
"""

# 開始をずらす時間(s) 600s->10mからスタート
SHIFT_TIME = 500
# プレイの始まる秒 フレームがリセットされるため 0なら無視(小さい動画)

# ビデオデータ
VIDEO_DATA = './movie_full/20171008E-H-Fi.mp4'

# 画像保存用のディレクトリ
IM_DIR_NAME = "images"
# ラベル保存用のディレクトリ
LB_DIR_NAME = "labels"
# データセットの種類: "player" or "offense" or "defense"
LABEL_NAME = "player"
LABEL_NUM = 1 # 1: player 2: offense 3: defense 4: glove
# XMLの<path>タグに指定するディレクトリ
ROOT_DIR = "/mnt/storage/users/kataoka/dataset/"
#
# # 1s = 10Frame?
# # offenseかdefenseか
# CSV_PATH = "./csv/{}_action.csv".format(LABEL_DICT[LABEL_NUM])

# Esc キー: 終了
ESC_KEY = 0x1b
# s キー: ストップ
S_KEY = 0x73
# r キー: 再開
R_KEY = 0x72
# 特徴点の最大数
MAX_FEATURE_NUM = 510
# 反復アルゴリズムの終了条件
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
# インターバル （1000 / フレームレート）
INTERVAL = 30

# 矩形のサイズ
CUT_WIDTH = 10 # 追加分の小さい動画
CUT_HEIGHT = 20 # 追加分の小さい動画

## グローブ？
X_LEFT = 0
X_RIGHT = 1280
Y_UPPER = 0
Y_LOWER = 720

# フレームの一部を切り取るときの変換分の座標
CONVERT_WIDTH = X_LEFT # フィールド(小)
CONVERT_HEIGHT = Y_UPPER # フィールド(小)

class Motion(object):
	# コンストラクタ
	def __init__(self, flag=True, csv_flag=False):
		self.im_dir_name = IM_DIR_NAME
		self.lb_dir_name = LB_DIR_NAME

		# 表示ウィンドウ
		cv2.namedWindow("motion")
		# マウスイベントのコールバック登録
		cv2.setMouseCallback("motion", self.onMouse)
		# 映像
		self.video = cv2.VideoCapture(VIDEO_DATA)
		# インターバル
		self.interval = INTERVAL
		# 現在のフレーム（カラー）
		self.frame = None
		# 現在のフレーム（グレー）
		self.gray_next = None
		# 前回のフレーム（グレー）
		self.gray_prev = None
		# 特徴点
		self.features = None
		# 特徴点のステータス
		self.status = None
		# 個別か全体か
		self.flag = flag
		# CSVに書き出すか否か
		self.csv_flag = csv_flag
		# 画像の幅、高さを取得
		WIDTH = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
		HEIGHT = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		self.WIDTH = X_RIGHT - X_LEFT #int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
		self.HEIGHT = Y_LOWER - Y_UPPER #int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
		print("Original  WIDTH: {0} HEIGHT: {1}".format(WIDTH, HEIGHT))
		print("Converted WIDTH: {0} HEIGHT: {1}".format(self.WIDTH, self.HEIGHT))
		# 何プレイ目か
		self.scene_flag = SHIFT_TIME

	# メインループ
	def run(self):

		#スタート地点をsetする
		self.video.set(0, SHIFT_TIME*1000)
		# 最初のフレームの処理 ## Return: True/False, Flame
		end_flag, self.frame = self.video.read()
		# カラースケールの場合 ## グレースケールに変換
		self.gray_prev = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		#元々グレースケールの場合
		#self.gray_prev = self.frame

		while end_flag:
			# グレースケールに変換
			#カラースケールの場合
			frame_count = int(self.video.get(cv2.CAP_PROP_POS_FRAMES))
			self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
			#元々グレースケールの場合
			#self.gray_next = self.frame
			#print("Frame: ", self.video.get(cv2.CAP_PROP_POS_FRAMES))

			# 特徴点が登録されている場合にOpticalFlowを計算する
			if self.features is not None:
				# オプティカルフローの計算
				features_prev = self.features
				self.features, self.status, err = cv2.calcOpticalFlowPyrLK( \
													self.gray_prev, \
													self.gray_next, \
													features_prev, \
													None, \
													winSize = (10, 10), \
													maxLevel = 3, \
													criteria = CRITERIA, \
													flags = 0)

				# 有効な特徴点のみ残す
				self.refreshFeatures()
				# フレームに有効な特徴点を描画 ## 長方形を描画
				if self.features is not None:
					if self.scene_flag != 0:
						fname = "{0}_{1}_{2}".format(VIDEO_DATA.split("/")[-1].split(".")[0], str(self.scene_flag), str(frame_count).zfill(7))
					else:
						fname = "{0}_{1}".format(VIDEO_DATA.split("/")[-1].split(".")[0], str(frame_count).zfill(7))
					folder_name = "data_"+LABEL_NAME if self.flag == True else "data_person"
					if LABEL_NUM == 1: # CSVに書き込むか否か
						addframe_to_xml(ROOT_DIR, LABEL_NAME, folder_name, fname, self.features, self.WIDTH, self.HEIGHT, CUT_WIDTH, CUT_HEIGHT, CONVERT_WIDTH, CONVERT_HEIGHT)
					if self.flag==True:
						self.saveFrameAndPosition(frame_count, self.features, self.scene_flag)
					for i,feature in enumerate(self.features):
						# cv2.circle(self.frame, (feature[0][0], feature[0][1]), 4, (15, 241-50*i, 200+i*10), -1, 8, 0)
						if self.csv_flag:
							write_csv(CSV_PATH, VIDEO_DATA, self.video.get(cv2.CAP_PROP_FRAME_HEIGHT), self.video.get(cv2.CAP_PROP_FRAME_WIDTH), self.video.get(cv2.CAP_PROP_POS_FRAMES),
							  [int(feature[0][0])-CUT_WIDTH, int(feature[0][1])-CUT_HEIGHT], [int(feature[0][0])+CUT_WIDTH, int(feature[0][1])+CUT_HEIGHT], PLAY_CATEGORY)
						if self.flag == False:
							self.saveRectangle(frame_count, i, feature)
							self.savePosition(frame_count, i, feature)
						bnd_box = (int(feature[0][0])-CUT_WIDTH, int(feature[0][1])-CUT_HEIGHT, int(feature[0][0])+CUT_WIDTH, int(feature[0][1])+CUT_HEIGHT)
						cv2.rectangle(self.frame, (int(feature[0][0])-CUT_WIDTH, int(feature[0][1])-CUT_HEIGHT), (int(feature[0][0])+CUT_WIDTH, int(feature[0][1])+CUT_HEIGHT), (15, 241-50*i, 200+i*10), 1, 8, 0)
						# self.saveRectangle(frame_count, i, feature)
						# self.savePosition(frame_count, i, feature)
						print("{0}:({1},{2})".format(i, int(feature[0][0]), int(feature[0][1])))
			# 表示
			cv2.imshow("motion", self.frame)

			# 次のループ処理の準備
			self.gray_prev = self.gray_next
			end_flag, self.frame = self.video.read()
			if end_flag:
				self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

			# インターバル
			key = cv2.waitKey(self.interval)
			# "Esc"キー押下で終了
			if key == ESC_KEY:
				break
			# "s"キー押下で一時停止
			elif key == S_KEY:
				self.interval = 0
			elif key == R_KEY:
				self.interval = INTERVAL

		# 終了処理
		cv2.destroyAllWindows()
		self.video.release()


	# マウスクリックで特徴点を指定する
	#	 クリックされた近傍に既存の特徴点がある場合は既存の特徴点を削除する
	#	 クリックされた近傍に既存の特徴点がない場合は新規に特徴点を追加する
	def onMouse(self, event, x, y, flags, param):
		# 左クリック以外
		if event != cv2.EVENT_LBUTTONDOWN:
			return

		# 最初の特徴点追加
		if self.features is None:
			self.addFeature(x, y)
			return

		# 探索半径（pixel）
		radius = 5
		# 既存の特徴点が近傍にあるか探索
		index = self.getFeatureIndex(x, y, radius)

		# クリックされた近傍に既存の特徴点があるので既存の特徴点を削除する
		if index >= 0:
			self.features = np.delete(self.features, index, 0)
			self.status = np.delete(self.status, index, 0)

		# クリックされた近傍に既存の特徴点がないので新規に特徴点を追加する
		else:
			self.addFeature(x, y)

		return


	# 指定した半径内にある既存の特徴点のインデックスを１つ取得する
	#	 指定した半径内に特徴点がない場合 index = -1 を応答
	def getFeatureIndex(self, x, y, radius):
		index = -1

		# 特徴点が１つも登録されていない
		if self.features is None:
			return index

		max_r2 = radius ** 2
		index = 0
		for point in self.features:
			dx = x - point[0][0]
			dy = y - point[0][1]
			r2 = dx ** 2 + dy ** 2
			if r2 <= max_r2:
				# この特徴点は指定された半径内
				return index
			else:
				# この特徴点は指定された半径外
				index += 1

		# 全ての特徴点が指定された半径の外側にある
		return -1


	# 特徴点を新規に追加する
	def addFeature(self, x, y):

		# 特徴点が未登録
		if self.features is None:
			# ndarrayの作成し特徴点の座標を登録
			self.features = np.array([[[x, y]]], np.float32)
			self.status = np.array([1])
			# 特徴点を高精度化
			cv2.cornerSubPix(self.gray_next, self.features, (10, 10), (-1, -1), CRITERIA)

		# 特徴点の最大登録個数をオーバー
		elif len(self.features) >= MAX_FEATURE_NUM:
			print("max feature num over: " + str(MAX_FEATURE_NUM))

		# 特徴点を追加登録
		else:
			# 既存のndarrayの最後に特徴点の座標を追加
			self.features = np.append(self.features, [[[x, y]]], axis = 0).astype(np.float32)
			self.status = np.append(self.status, 1)
			# 特徴点を高精度化
			cv2.cornerSubPix(self.gray_next, self.features, (10, 10), (-1, -1), CRITERIA)

	# 有効な特徴点のみ残す
	def refreshFeatures(self):
		# 特徴点が未登録
		if self.features is None:
			return

		# 全statusをチェックする
		i = 0
		while i < len(self.features):

			# 特徴点として認識できず
			if self.status[i] == 0:
				# 既存のndarrayから削除
				self.features = np.delete(self.features, i, 0)
				self.status = np.delete(self.status, i, 0)
				i -= 1

			i += 1

	# 人物が含まれる矩形を切り取る
	def saveRectangle(self, frame_count, i, feature):
		cutted_frame = self.frame[int(feature[0][1])-CUT_HEIGHT*2:int(feature[0][1])+CUT_HEIGHT*2, int(feature[0][0])-CUT_WIDTH*2:int(feature[0][0])+CUT_WIDTH*2]
		if not os.path.exists("./data_person"):
			os.mkdir("./data_person")
			if not os.path.exists("./data_frame/{}".format("xml")):
				os.mkdir("./data_frame/{}".format("xml"))
			if not os.path.exists("./data_person/" + self.im_dir_name):
				os.mkdir("./data_person/{}".format(self.im_dir_name))
		img_path = "./data_person/{0}/{1}_{2}.png".format(self.im_dir_name, str(i).zfill(7), str(frame_count))
		cv2.imwrite(img_path, cutted_frame)

	# バウンディングボックスの位置を保存する
	def savePosition(self, frame_count, i, feature):
		if not os.path.exists("./data_person"):
			os.mkdir("./data_person")
		if not os.path.exists("./data_person/{}".format("xml")):
			os.mkdir("./data_person/{}".format("xml"))
		if not os.path.exists("./data_person/" + self.lb_dir_name):
			os.mkdir("./data_person/{}".format(self.lb_dir_name))
		txt_path = "./data_person/{0}/{1}_{2}.txt".format(self.lb_dir_name, str(i).zfill(7), str(frame_count))
		with open(txt_path, "w") as wf:
			wf.write("{}\n".format(LABEL_NUM))
			xmin = int(feature[0][0])-CUT_WIDTH-CONVERT_WIDTH if int(feature[0][0])-CUT_WIDTH-CONVERT_WIDTH > 0 else 0
			ymin = int(feature[0][1])-CUT_HEIGHT-CONVERT_HEIGHT if int(feature[0][1])-CUT_HEIGHT-CONVERT_HEIGHT > 0 else 0
			xmax = int(feature[0][0])+CUT_WIDTH-CONVERT_WIDTH if int(feature[0][0])+CUT_WIDTH-CONVERT_WIDTH > 0 else 0
			ymax = int(feature[0][1])+CUT_HEIGHT-CONVERT_HEIGHT if int(feature[0][1])+CUT_HEIGHT-CONVERT_HEIGHT > 0 else 0
			wf.write("{0} {1} {2} {3}\n".format(xmin, ymin, xmax, ymax))

	# フレームごと切り取る
	def saveFrameAndPosition(self, i, features, scene_flag):
		v_title = VIDEO_DATA.split("/")[-1].split(".")[0]
		if not os.path.exists("./data_"+LABEL_NAME):
			os.mkdir("./data_"+LABEL_NAME)
		if not os.path.exists("./data_{0}/{1}".format(LABEL_NAME, "xml")):
			os.mkdir("./data_{0}/{1}".format(LABEL_NAME, "xml"))
		if not os.path.exists("./data_{0}/{1}".format(LABEL_NAME, self.im_dir_name)):
			os.mkdir("./data_{0}/{1}".format(LABEL_NAME, self.im_dir_name))
		if not os.path.exists("./data_{0}/{1}".format(LABEL_NAME, self.lb_dir_name)):
			os.mkdir("./data_{0}/{1}".format(LABEL_NAME, self.lb_dir_name))
		if scene_flag != 0:
			v_title += "_" + str(scene_flag)
		txt_path = "./data_{0}/{1}/{2}_{3}".format(LABEL_NAME, self.lb_dir_name, v_title, str(i).zfill(7))
		img_path = "./data_{0}/{1}/{2}_{3}".format(LABEL_NAME, self.im_dir_name, v_title, str(i).zfill(7))
		with open(txt_path+".txt", "w") as wf:
			wf.write("{}\n".format(LABEL_NUM))
			for feature in features:
				xmin = int(feature[0][0])-CUT_WIDTH-CONVERT_WIDTH if int(feature[0][0])-CUT_WIDTH-CONVERT_WIDTH > 0 else 0
				ymin = int(feature[0][1])-CUT_HEIGHT-CONVERT_HEIGHT if int(feature[0][1])-CUT_HEIGHT-CONVERT_HEIGHT > 0 else 0
				xmax = int(feature[0][0])+CUT_WIDTH-CONVERT_WIDTH if int(feature[0][0])+CUT_WIDTH-CONVERT_WIDTH > 0 else 0
				ymax = int(feature[0][1])+CUT_HEIGHT-CONVERT_HEIGHT if int(feature[0][1])+CUT_HEIGHT-CONVERT_HEIGHT > 0 else 0
				wf.write("{0} {1} {2} {3}\n".format(xmin, ymin, xmax, ymax))
		cv2.imwrite(img_path+".png", self.frame[Y_UPPER:Y_LOWER, X_LEFT:X_RIGHT])
		# cv2.imwrite(img_path+"_l.txt", self.frame[, int(self.width*1/8):int(self.width/2)])
		# cv2.imwrite(img_path+"_r.txt", self.frame[, int(self.width/2):int(self.width*7/8)])

if __name__ == '__main__':
	# Motion(arg_parse()).run()
	Motion().run()
