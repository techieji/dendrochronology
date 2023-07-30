from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
import scipy.signal
import itertools as it

# https://stackoverflow.com/a/75114333/11555240
def sliding_median(arr, window):
    return np.median(np.lib.stride_tricks.sliding_window_view(arr, (window,)), axis=1)

with Image.open('IMG_2797.jpg') as im:
    h = im.height
    w = im.width
    bwim = ImageEnhance.Color(im).enhance(0)
    px = np.array(bwim)[:, :, 0]

l = np.array([np.median(px[y, round(w*y/h):]) for y in range(h)])
s1 = scipy.signal.savgol_filter(l, 50, 3)
s2 = scipy.signal.savgol_filter(l, 100, 3)
v = sliding_median(np.clip(np.round(s1 - s2), -1, 1), 20)
count = [k for k, g in it.groupby(v)].count(1)
print(count)

plt.figure()
plt.plot(l)
plt.plot(s1)
plt.plot(s2)
plt.plot(s1 - s2)
plt.plot(v)
plt.show()
