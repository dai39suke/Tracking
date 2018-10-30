# -*- coding: utf-8 -*-

# from xml.dom.minidom import parseString
# import xml.etree.ElementTree as ET
from lxml import etree
import lxml
import os

img_path = "images"
lb_path = "labels"

# ここの変更し忘れ@New Dataset
# フレームの一部を切り取るときの変換分の座標
# CONVERT_WIDTH = 80
# CONVERT_HEIGHT = 250
# CONVERT_WIDTH = 0 # 変換なし
# CONVERT_HEIGHT = 0 # 変換なし

def addobject(label_name, xml, features, add_width, add_height, CONVERT_WIDTH, CONVERT_HEIGHT):
	for feature in features:
		# XMLテキストを読み込み
		xmin = int(feature[0][0]) - add_width  - CONVERT_WIDTH  if int(feature[0][0]) - add_width  - CONVERT_WIDTH  > 0 else 0
		ymin = int(feature[0][1]) - add_height - CONVERT_HEIGHT if int(feature[0][1]) - add_height - CONVERT_HEIGHT > 0 else 0
		xmax = int(feature[0][0]) + add_width  - CONVERT_WIDTH  if int(feature[0][0]) + add_width  - CONVERT_WIDTH  > 0 else 0
		ymax = int(feature[0][1]) + add_height - CONVERT_HEIGHT if int(feature[0][1]) + add_height - CONVERT_HEIGHT > 0 else 0
		xml = xml + """\t<object>
		<name>{0}</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{1}</xmin>
			<ymin>{2}</ymin>
			<xmax>{3}</xmax>
			<ymax>{4}</ymax>
		</bndbox>
	</object>""".format(label_name, xmin, ymin, xmax, ymax)
		# obj = etree.XML(xml_template)
		# obj.xpath("bndbox/xmin")[0].text = str(int(feature[0][0])-add_width)
		# obj.xpath("bndbox/ymin")[0].text = str(int(feature[0][1])-add_height)
		# obj.xpath("bndbox/xmax")[0].text = str(int(feature[0][0])+add_width)
		# obj.xpath("bndbox/ymax")[0].text = str(int(feature[0][1])+add_height)
		# with open("xml/{}.xml".format(filename), "a") as wf:
		#	 wf.write(etree.tostring(obj, method="xml", pretty_print=True).decode("utf-8"))
	return xml


def addframe(root_dir, label_name, folder, filename, features, width, height, add_width, add_height, CONVERT_WIDTH, CONVERT_HEIGHT):
	img_folder_path = root_dir + img_path
	img_folder_abs_path = root_dir + img_path + "/" + label_name
	xml = """<annotation verified="no">
\t<folder>{0}</folder>
\t<filename>{1}</filename>
\t<path>{2}</path>
\t<source>
\t\t<database>Unknown</database>
\t</source>
\t<size>
\t\t<width>{3}</width>
\t\t<height>{4}</height>
\t\t<depth>3</depth>
\t</size>
\t<segmented>0</segmented>
""".format(img_path, filename, os.path.join(img_folder_abs_path, filename+".png"), width, height)
	xml = addobject(label_name, xml, features, add_width, add_height, CONVERT_WIDTH, CONVERT_HEIGHT)
	xml = xml + "\n</annotation>"
	if not os.path.exists("./data_"+label_name):
		os.mkdir("./data_"+label_name)
	if not os.path.exists("./data_{0}/{1}".format(label_name, "xml")):
		os.mkdir("./data_{0}/{1}".format(label_name, "xml"))
	if not os.path.exists("./data_{0}/{1}".format(label_name, img_path)):
		os.mkdir("./data_{0}/{1}".format(label_name, img_path))
	if not os.path.exists("./data_{0}/{1}".format(label_name, lb_path)):
		os.mkdir("./data_{0}/{1}".format(label_name, lb_path))
	with open("./{0}/{1}_{2}.xml".format(folder+"/xml", label_name, filename), "w") as wf:
		# wf.write(etree.tostring(root, method="xml", pretty_print=True).decode("utf-8"))
		wf.write(xml)
	# print(etree.tostring(root, method="html", pretty_print=True).decode("utf-8"))

# <annotation verified="no">
#   <folder>images</folder>
#   <filename>field_00001</filename>
#   <path>/images/field_00001.jpg</path>
#   <source>
#	 <database>Unknown</database>
#   </source>
#   <size>
#	 <width>640</width>
#	 <height>480</height>
#	 <depth>3</depth>
#   </size>
#   <segmented>0</segmented>
# </annotation>

if __name__ == "__main__":
	addframe("images", "field_00001", 640, 480)
	# addobject("field001", (20, 40, 50, 60), 640, 480)
