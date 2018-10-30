# -*- coding: utf-8 -*-
import cv2
import numpy as np

"""
Usage:
	1. sで一時停止 rで再開 ESCで終了
	2. sで一時停止している状態で、トラッキング対象の選手をクリック
Idea:
	1.  特徴点が何ピクセル以内に重なったとき、同じ速度で動くようにする
		← 急に止まることが無いため
Issue:
	1. 選手が重なった際に片方の選手に引っ張られる現象をどう解決するか
	2. ベースと重なったときに、特徴点が置き去りになる現象をどう解決するか
"""


# Esc キー
ESC_KEY = 0x1b
# s キー
S_KEY = 0x73
# r キー
R_KEY = 0x72
# 特徴点の最大数
MAX_FEATURE_NUM = 500
# 反復アルゴリズムの終了条件
CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
# インターバル （1000 / フレームレート）
INTERVAL = 30
# ビデオデータ
# VIDEO_DATA = './02.mp4'
VIDEO_DATA = './front.mp4'
# 特徴点から矩形までの横幅
CUT_WIDTH = 60 # 60:グローブ
# 特徴点から矩形までの縦幅
CUT_HEIGHT = 60 # 60:グローブ

class Motion:
	# コンストラクタ
	def __init__(self):
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
		# 出力先のディレクトリ
		self.out_dir = ''

	# メインループ
	def run(self):

		# 最初のフレームの処理 ## Return: True/False, Flame
		end_flag, self.frame = self.video.read()
		# カラースケールの場合 ## グレースケールに変換
		self.gray_prev = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
		#元々グレースケールの場合
		#self.gray_prev = self.frame

		while end_flag:
			# グレースケールに変換
			#カラースケールの場合
			self.gray_next = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
			#元々グレースケールの場合
			#self.gray_next = self.frame

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
					for i,feature in enumerate(self.features):
						cv2.circle(self.frame, (feature[0][0], feature[0][1]), 4, (15, 241-50*i, 200+i*10), -1, 8, 0)
						cv2.rectangle(self.frame, (int(feature[0][0])-CUT_WIDTH, int(feature[0][1])-CUT_HEIGHT), (int(feature[0][0])+CUT_WIDTH, int(feature[0][1])+CUT_HEIGHT), (15, 241-50*i, 200+i*10), 1, 8, 0)
						print("{0}:({1},{2})".format(i,int(feature[0][0]),int(feature[0][1])))

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


if __name__ == '__main__':
	Motion().run()
