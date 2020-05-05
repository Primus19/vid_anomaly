import numpy as np
import skvideo.io
from c3d_features import get_feature_extractor
from c3d.sports1M_utils import preprocess_input, decode_predictions
from MLP_nn import ClassifierMLP

def get_frames_features(video_frames):
    c3d_model=get_feature_extractor() #feature extractor 4096 output
    x_input = preprocess_input(vid)
    features_vector = c3d_model.predict(x_input)
    return features_vector

def get_video_features(video_path):
        '''
        return vid ndarray of dimension (f, h, w, c),
        f is the number of frames
        h is the height
        w is width
        c is depth. (channels or colors)
        '''
        vid = skvideo.io.vread(video_path)
        f,_,_c=video_frames.shape
        if (f!=16 and c!=3):
            print('The C3D algorithm required 16 frames and 3 channels, you pass f=%s c=%s'%(f,c))
            raise ValueError
        n=f//16
        r=f%16
        features_list=[]
        for i in range(n):
            video_frames= vid[i*16:i*16+15] #fist 16 frames
            features=get_frames_features(video_frames)
            features_list.append(features)
        if r!=0:
            features_list.append(feature_list[-1])
        return feature_list

if __name__=='__main__':
    input_features=get_video_features(video_path)
    config_dic={'input_dim':4096,
                'learning_rate':0.001,
                'epochs':100,
                'batch_size':30,
                'dropout':0.6,
                }
    #create model
    mlp_nn=ClassifierMLP(config) #create an object of the class MLP
    mlp_nn.create_model() #create the model itself
    #train model
    '''
    x_train: in a way is an array of inputs, the inputs should be 4096 vectors
    that came from 16 video frames pre-process.
    y_train: it is a set of outputs that classify eatch bath of inputs as normal or unormal

    '''
    mlp_nn.train(x_train=input_features_array,y_train)
    mlp_model=mlp_nn.model #getting model after it is trained
    nn_estimator.evaluate_nn_model(x_test,y_test)
    test_score=mlp_nn.score[1]*100
    print('Test score:',test_score)
