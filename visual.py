"""
Minimal prediction example
"""
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, LayerCAM, score_cam
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
from torchvision.models import resnet50
import torch
from PIL import Image
import os
from model.architecture import COVIDNext50
from data.transforms import val_transforms
from model import architecture
import util
import logging

import config
import numpy as np
import cv2
rev_mapping = {idx: name for name, idx in config.mapping.items()}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if config.gpu and not torch.cuda.is_available():
    raise ValueError("GPU not supported or enabled on this system.")
use_gpu = config.gpu


#vit
def reshape_transform_vit(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                    height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# #swin
def reshape_transform_swin(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def main(net):
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    if net=='resnet50':
        weight = config.weights[1]
    if net == 'resnet_csra':
        weight = config.weights[2]
    if net == 'resnet_csra_fpn':
        weight = config.weights[3]
    if net == 'vit':
        weight = config.weights[3]
    if net == 'vit_csra':
        weight = config.weights[4]
    if net == 'swin':
        weight = config.weights[5]
    if net == 'swin_csra':
        weight = config.weights[6]

    if weight:
        state = torch.load(weight)
        # state = None
        log.info("Loaded model weights from: {}".format(weight))
    else:
        state = None

    state_dict = state["state_dict"] if state else None
    model = architecture.COVIDNext50(net,n_classes=config.n_classes,)
    if state_dict:
        model = util.load_model_weights(model=model, state_dict=state_dict)

    if use_gpu:
        model.cuda()
        # model = torch.nn.DataParallel(model)

    transforms = val_transforms(width=config.width, height=config.height)


    gradacam = False
    gradcammplusplus=False
    layercam = True
    scorecam = False

    pth = config.sample

    i=0
    while(i<3):
        if i==0:
            imgs =os.listdir(pth+'normal/')
            imgs_pth = pth+'normal/'
        elif i==1:
            imgs =os.listdir(pth+'pneumonia/')
            
            imgs_pth = pth+'pneumonia/'
        elif i==2:
            imgs =os.listdir(pth+'COVID-19/')
            imgs_pth = pth+'COVID-19/'

        for img_name in imgs:
            img_pth = imgs_pth+img_name
            print(img_pth)
            img = Image.open(img_pth).convert("RGB")
            # img = np.float32(img) / 255
            img_tensor = transforms(img).unsqueeze(0)
            img = img.resize((224,224))
            img = np.float32(img) / 255
            # rgb_img = cv2.imread(img_pth, 1)[:, :, ::-1]
            # rgb_img = cv2.resize(rgb_img, (224, 224))
            # rgb_img = np.float32(rgb_img) / 255
            # img_tensor = preprocess_image(img,mean=[0.5,0.5,0.5],std=[0.5, 0.5, 0.5])

            img_tensor = util.to_device(img_tensor, gpu=config.gpu)
            with torch.no_grad():
                logits = model(img_tensor)
                cat_id = int(torch.argmax(logits))
            print("Prediction for {} is: {},{}".format(img_pth, rev_mapping[cat_id],cat_id))

            if net =='resnet50':
                target_layers = [model.resnet.layer4[-1]] #resnet
            elif net == 'resnet_csra' or net=='resnet_csra_fpn':
                target_layers = [model.resnet_csra.layer4[-1]] #resnet_csra
            elif net == 'vit' or net == 'vit_csra':
                target_layers = [model.vit.blocks[-1].norm1] #vit
            elif net == 'swin' or net == 'swin_csra':
                target_layers = [model.model.layers[-1].blocks[-1].norm2] #swin


            # Construct the CAM object once, and then re-use it on many images:
            if gradacam==True:
                cam = GradCAM(model=model, target_layers=target_layers, use_cuda=config.gpu)  
                # cam = GradCAM(model=model, target_layers=target_layers, use_cuda=config.gpu,reshape_transform=reshape_transform)  
                cam_type = 'gradcam'
            elif gradcammplusplus==True:
                cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=config.gpu)
                cam_type = 'gradcamplusplus'
            elif scorecam==True:
                cam = ScoreCAM(model=model, target_layers=target_layers, use_cuda=config.gpu)
                cam_type = 'scorecam'
            elif layercam==True:
                if net == 'swin' or net == 'swin_csra':
                    cam = LayerCAM(model=model, target_layers=target_layers, use_cuda=config.gpu,reshape_transform=reshape_transform_swin)  
                elif net == 'vit' or net == 'vit_csra':
                    cam = LayerCAM(model=model, target_layers=target_layers, use_cuda=config.gpu,reshape_transform=reshape_transform_vit)  
                # cam = LayerCAM(model=model, target_layers=target_layers, use_cuda=config.gpu)
                else:
                    cam = LayerCAM(model=model, target_layers=target_layers, use_cuda=config.gpu)  
                cam_type = 'layercam'

            # You can also use it within a with statement, to make sure it is freed,
            # In case you need to re-create it inside an outer loop:
            # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
            #   ...

            # If target_category is None, the highest scoring category
            # will be used for every image in the batch.
            # target_category can also be an integer, or a list of different integers
            # for every image in the batch.
            target_category = None
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            # grayscale_cam = cam(input_tensor=img_tensor, target_category=target_category,aug_smooth=True,eigen_smooth=True)
            grayscale_cam = cam(input_tensor=img_tensor, targets=target_category)


            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
            # plt.show(cam_image.all())
            class_path = './results/'+net+'/'+cam_type
            if 'normal' in imgs_pth:
                img_path = class_path +'/Normal/'
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                img_save =img_path+rev_mapping[cat_id]+'_'+img_name
            if 'COVID-19' in imgs_pth:
                img_path = class_path +'/COVID-19/'
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                img_save =img_path+rev_mapping[cat_id]+'_'+img_name
            if 'pneumonia' in imgs_pth:
                img_path = class_path +'/Pneumonia/'
                if not os.path.exists(img_path):
                    os.makedirs(img_path)
                img_save =img_path+rev_mapping[cat_id]+'_'+img_name
            cv2.imwrite(img_save, cam_image)
        i=i+1
if __name__ == '__main__':
    seed = config.random_seed
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if config.gpu and not torch.cuda.is_available():
        raise ValueError("GPU not supported or enabled on this system.")
    use_gpu = config.gpu
    
    main(net = 'resnet50')
    main(net = 'resnet_csra')
    main(net = 'resnet_csra_fpn')
    # main(net = 'vit')
    # main(net = 'vit_csra')
    # main(net = 'swin')
    # main(net = 'swin_csra')

