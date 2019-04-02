#################################################
#    >File Name: q2.py
#    >Author: Tingyu Peng
#    >mail: PengTingyu.d@gmail.com
#    >Created Time: 2019年04月02日 星期二 21时47分15秒
#################################################
#!/usr/bin/python
# -*- coding:utf-8 -*-

from PIL import ImageColor
from PIL import Image
# print(type(ImageColor.getcolor('red','RGBA')))
img1 = Image.open("data/lena.png")
img2 = Image.open("data/lena_modified.png")

w, h = img1.size

for i in range(0,w):
    for j in range(0,h):
        if img1.getpixel((i,j)) == img2.getpixel((i,j)):
            img2.putpixel((i,j),255)

img2.save("ans_two.png")
