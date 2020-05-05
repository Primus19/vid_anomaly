import skvideo.io
from c3d import C3D
from sports1M_utils import preprocess_input, decode_predictions

model = C3D(weights='sports1M')
#VIDEO_PATH='cat.mp4'
VIDEO_PATH='Ballpark.mp4'

vid_path=VIDEO_PATH
#vid_path = 'homerun.mp4'
'''
return vid ndarray of dimension (T, M, N, C),
T is the number of frames
M is the height
N is width
C is depth. (channels or colors)
'''
vid = skvideo.io.vread(vid_path)

# Select 16 frames from video
vid = vid[40:56]
x = preprocess_input(vid)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
#Predicted: [('baseball', 0.91488838)]
