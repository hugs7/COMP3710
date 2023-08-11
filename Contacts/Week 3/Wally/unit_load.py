import imageio.v2 as ii
import skimage as ski

import matplotlib.pyplot as plt

face = ii.imread('wally_template_castle_edit.png')

face = ski.color.rgb2gray(face)
tx, ty = face.shape
print("Face Shape", face.shape)

scene = ii.imread('Wally_castle.jpg')
scene = ski.color.rgb2gray(scene)

sx, sy = scene.shape
print("Scene Shape", scene.shape)


# Plot scene and face
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 10))
plt.gray()
plt.tight_layout()

ax.imshow(scene, interpolation="nearest")
ax.axis('off')
ax.set_title('Image')

plt.show()
