
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a color image and visualize each channel separately
img = cv2.imread('/home/rodol/Imagens/PDI/baboon.png', cv2.IMREAD_COLOR)
bgr = cv2.split(img)

plt.subplot('221'), plt.title('B'), plt.imshow(bgr[0], 'gray')
plt.subplot('222'), plt.title('G'), plt.imshow(bgr[1],'gray')
plt.subplot('223'), plt.title('R'), plt.imshow(bgr[2],'gray')
plt.subplot('224'), plt.title('Original'), plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a color image and visualize each channel separately
img = cv2.imread('/home/rodol/Imagens/PDI/baboon.png', cv2.IMREAD_COLOR)
#bgr2rgb = cv2.split(img)
# img = bgr2rgb(img)
r = [1,0,0] # red
g = [0,1,0] # green
b = [0,0,1] # blue
c = [0,1,1] # cyan
m = [1,0,1] # magenta
y = [1,1,0] # yellow

plt.subplot('241'); plt.title('RGB'); plt.imshow(img)
plt.subplot('242'); plt.title('R'); plt.imshow(r * img)
plt.subplot('243'); plt.title('G'); plt.imshow(g * img)
plt.subplot('244'); plt.title('B'); plt.imshow(b * img)
plt.subplot('245'); plt.title('C'); plt.imshow(c * img)
plt.subplot('246'); plt.title('M'); plt.imshow(m * img)
plt.subplot('247'); plt.title('Y'); plt.imshow(y * img)

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

#  Load a color imagem an visualize each channel separately
img = cv2.imread('/home/rodol/Imagens/PDI/baboon.png', cv2.IMREAD_COLOR)
bgr = cv2.split(img)
colormap = cv2.COLORMAP_JET

plt.subplot('221'), plt.title('B'), plt.imshow(cv2.applyColorMap[0], colormap)
plt.subplot('222'), plt.title('G'), plt.imshow(cv2.applyColorMap[1], colormap)
plt.subplot('223'), plt.title('R'), plt.imshow(cv2.applyColorMap[2], colormap)
plt.subplot('224'), plt.title('Original'), plt.imshow(cv2.cvtColorMap,(img, cv2.COLOR_BGR2RGB))
plt.show()

#%%

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a color image and visualize each channel separetely
img = cv2.imread('/home/rodol/Imagens/baboon.png', cv2.IMREAD_COLOR)
bgr = cv2.split(img)
# colormap = cv2.COLORMAT_JET
colormap = 1

plt.subplot('221'), plt.title('B'), plt.imshow(cv2.applyColorMap(bgr[0]), colormap)
plt.subplot('222'), plt.title('G'), plt.imshow(cv2.applyColorMap(bgr[1]), colormap)
plt.subplot('223'), plt.title('R'), plt.imshow(cv2.applyColorMap(bgr[2]), colormap)
plt.subplot('224'), plt.title('Original'), plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a color image and visualize each channel separetely
img = cv2.imread('/home/rodol/Imagens/baboon.png', cv2.IMREAD_COLOR)
img2 = 255 - img
bgr = cv2.split(img2)
colormap = cv2.COLORMAP_JET

plt.subplot('221'), plt.title('B'), plt.imshow(cv2.applyColorMap(bgr[0]), colormap)
plt.subplot('222'), plt.title('G'), plt.imshow(cv2.applyColorMap(bgr[1]), colormap)
plt.subplot('223'), plt.title('R'), plt.imshow(cv2.applyColorMap(bgr[2]), colormap)
plt.subplot('224'), plt.title('Original'), plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

#%% Creating BGR color disks
import cv2
import numpy as np
import matplotlib.pyplot as plt

rows = 1e3
radius = rows/4
bx = rows/2
by = rows/2 - radius/2
gx = rows/2 - radius/2
gy = rows/2 + radius/2
rx = rows/2 + radius/2
ry = rows/2 + radius/2
#
bgr = [createWhiteDisk(int(rows), int(rows), int(bx), int(by), int(radius)),
       createWhiteDisk(int(rows), int(rows), int(gx), int(gy), int(radius)),
        createWhiteDisk(int(rows), int(rows), int(rx), int(ry), int(radius))]

cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
while 0xFF & cv2.waitKey(1) != ord('q'):
    img = cv2.merge(bgr)
    img = scaleImag2_uchar(img)
    cv2.imshow('img', img)
cv2.destroyAllWindows()

