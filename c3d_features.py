from c3d.c3d import C3D
#from c3d.sports1M_utils import preprocess_input, decode_predictions
#import skvideo.io

def get_feature_extractor():
    model = C3D(weights='sports1M')
    #print('Complete C3D model\n')
    #print(model.summary())
    #Remove two layers
    model._layers.pop(-2)
    model._layers.pop(-1)
    #print('Feature extraction model without last two layers')
    print('Feature extractor model')
    print(model.summary())
    return(model)
