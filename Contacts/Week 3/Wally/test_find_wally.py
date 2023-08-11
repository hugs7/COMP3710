'''
Find Wally in a scene
'''

import imageio.v2 as ii
import skimage as ski

face = ii.imread('wally_template_castle_edit.png')

face = ski.color.rgb2gray(face)
tx, ty = face.shape
print("Face Shape", face.shape)

scene = ii.imread('Wally_castle.jpg')
scene = ski.color.rgb2gray(scene)

sx, sy = scene.shape
print("Scene Shape", scene.shape)
