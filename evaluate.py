"""
Minimal prediction example
"""

import torch
from PIL import Image
import csv
import os.path as osp

from model.architecture import COVIDNext50
# from data.transforms import val_transforms
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from plot_metric.functions import MultiClassClassification

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from numpy import interp
from itertools import cycle
from prettytable import PrettyTable
from tqdm import tqdm
import json

import config
import os
import util
import numpy as np

import logging

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from data.dataset import COVIDxFolder
from data import transforms
from torch.utils.data import DataLoader
from model import architecture

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import time
from fvcore.nn import FlopCountAnalysis, parameter_count
from tqdm import tqdm

def flops_params_fps(model, input_shape=(1, 3, 224, 224)):
    """count flops:G params:M fps:img/s
        input shape tensor[1, c, h, w]
    """
    total_time = []
    with torch.no_grad():
        model = model.cuda().eval()
        input = torch.randn(size=input_shape, dtype=torch.float32).cuda()
        flops = FlopCountAnalysis(model, input)
        params = parameter_count(model)

        # for i in tqdm(range(100)):
        #     torch.cuda.synchronize()
        #     start = time.time()
        #     output = model(input)
        #     torch.cuda.synchronize()
        #     end = time.time()
        #     total_time.append(end - start)
        # mean_time = np.mean(np.array(total_time))
        # print(model.__class__.__name__)
        # print('img/s:{:.2f}'.format(1 / mean_time))
        print('flops:{:.2f}G params:{:.2f}M'.format(flops.total() / 1e9, params[''] / 1e6))

        return round(params[''] / 1e6,2), round(flops.total() / 1e9,2)

def plot_roc_auc(mc,net):

    plt.figure()
    mc.plot_roc()

    roc_path = './results/'+net+'/roc/'
    if not os.path.exists(roc_path):
        os.makedirs(roc_path)
    roc_file = roc_path+net+'_roc.png'
    plt.savefig(roc_file, format='png', dpi=1200)
    plt.show()
    plt.close()

# def get_pr(trues, preds,net):

#     #roc
#     # roc curve
#     # fpr = dict()
#     # tpr = dict()

#     # for i in range(n_classes):
#     #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i],
#     #                                   y_score[:, i]))
#     #     plt.plot(fpr[i], tpr[i], lw=2, label='class {}'.format(i))

#     # plt.xlabel("false positive rate")
#     # plt.ylabel("true positive rate")
#     # plt.legend(loc="best")
#     # plt.title("ROC curve")
#     # plt.show()
#     # precision recall curve
#     labels = [0,1,2]
#     n_classes = len(labels)
#     precision = dict()
#     recall = dict()
#     for i in range(n_classes):
#         precision[i], recall[i], _ = precision_recall_curve(trues[:, i],
#                                                             preds[:, i])
#         plt.plot(recall[i], precision[i], lw=2, label='Class {}'.format(i))
        
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.legend(loc="best")
#     plt.title("Precision vs. Recall curve")
#     plt.show()
#     pr_path = './results/'+net+'/pr/'
#     if not os.path.exists(pr_path):
#         os.makedirs(pr_path)
#     pr_file = pr_path+net+'_roc.png'
#     plt.savefig(pr_file, format='png', dpi=1200)
#     plt.close()

def plot_confusion_matrix(net,cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Reds, label_rotation=45):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Input
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm_norm = cm
    cm_norm = cm_norm.astype('float') / cm_norm.sum(axis=1)[:, np.newaxis]
    # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=label_rotation)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.colorbar()

    fmt = '.2f' if normalize else 'd'
    normalize=True
    fmt2 = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    thresh2 = cm_norm.max() / 2 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt)+'('+ format(cm_norm[i, j], fmt2)+')',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    metrix_path = './results/'+net+'/confusion_matrices/'
    if not os.path.exists(metrix_path):
        os.makedirs(metrix_path)
    if normalize:
        metrix_file = metrix_path+net+'_confusion_matrices.png'
        plt.savefig(metrix_file, format='png', dpi=1200)
    else:
        metrix_file = metrix_path+net+'_confusion_matrices.png'
        plt.savefig(metrix_file, format='png', dpi=1200)
    plt.close()

def get_specifity(confusionmatrix,net):
    specificitys = []
    classes=['Normal','Pneumonia','COVID-19']
    # precision, recall, specificity
    table = PrettyTable()
    table.field_names = ["", "Precision", "Recall", "Specificity"]
    sum_spec =0
    for i in range(3):
        TP = confusionmatrix[i, i]
        FP = np.sum(confusionmatrix[:, i]) - TP
        FN = np.sum(confusionmatrix[i, :]) - TP
        TN = confusionmatrix[0,0]+confusionmatrix[1,1]+confusionmatrix[2,2] - TP 
        
        Precision = round(TP / (TP + FP), 4) if TP + FP != 0 else 0.
        Recall = round(TP / (TP + FN), 4) if TP + FN != 0 else 0.
        Specificity = round(TN / (TN + FP), 4) if TN + FP != 0 else 0.
        sum_spec+=Specificity
        table.add_row([classes[i], Precision, Recall, Specificity])
        specificitys.append(Specificity)
    # print(table)
    spec = sum_spec/3
    # print("model specificity：{:.4f}".format())
    return specificitys, spec

def validate(net,data_loader, model, best_score, global_step, cfg):

    global bestacc
    model.eval()
    gts, predictions = [], []

    log.info("Validation started...")
    correct_meter = AverageMeter()

    with torch.no_grad():
        for step, (data, targets) in enumerate(data_loader):

                data = data.to(device)
                targets = targets.to(device)

                outputs = model(data)

                _, preds = torch.max(outputs, dim=1)

                correct_ = preds.eq(targets).sum().item()
                num = data.size(0)

                correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(data_loader.dataset)
    log.info('Accuracy {:.4f}'.format(accuracy))

    log.info("Validation end")

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count    


def evaluate(net,data_loader, model, best_score, global_step, cfg):
    global bestacc
    model.eval()
    gts, predictions = [], []
    predic, gts, predictions = [], [], []
    log.info("Validation started...")

    param, flops = flops_params_fps(model)
    print(param)
    print(flops)

    for data in data_loader:
        imgs, labels = data
        imgs = util.to_device(imgs, gpu=cfg.gpu)

        with torch.no_grad():
            logits = model(imgs)
            probs = model.module.probability(logits)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

        labels = labels.cpu().detach().numpy()
        logits = logits.cpu().detach().numpy()

        predic.extend(logits)
        predictions.extend(preds)
        gts.extend(labels)

    predic = np.array(predic)
    predictions = np.array(predictions, dtype=np.int32)
    gts = np.array(gts, dtype=np.int32)
    acc, f1, prec, rec = util.clf_metrics(predictions=predictions,
                                          targets=gts,
                                          average="macro")
    report = classification_report(gts, predictions,digits=4)
    
    ConfusionMatrix_test = confusion_matrix(gts,predictions)

    #confusion_matrix
    # plot_confusion_matrix(net,ConfusionMatrix_test, classes=['Normal','Pneumonia','COVID-19'], normalize=True, title='Normalized confusion matrix')
    plot_confusion_matrix(net,ConfusionMatrix_test, classes=['Normal','Pneumonia','COVID-19'], normalize=False, title='Confusion matrix')
    
    #specifity
    specificitys, spec = get_specifity(ConfusionMatrix_test,net)

    table = PrettyTable()
    table.field_names = ["Class", "Precision", "Recall", "Specificity", "F1-Scpre", "Support"]
    report_list =report.replace('\n', '').replace('\r', '').split(" ")
    results = []
    for re in report_list:
        if re =="accuracy":
            break
        if re!='':
            if re =='0':
                re ='Normal'
            elif re =='1':
                re = "Pneumonia"
            elif re =='2':
                re = "COVID-19"
            results.append(re)


    table.add_row([results[4], results[5], results[6], specificitys[0],results[7],results[8]])
    table.add_row([results[9], results[10], results[11], specificitys[1],results[12],results[13]])
    table.add_row([results[14], results[15], results[16], specificitys[2],results[17],results[18]])
    print(table)
    print(net + ": accuracy {:.4f} |precision {:.4f} |recall {:.4f} |specify {:.4f} |f1 {:.4f} |param {} |flops {} ".format(acc, prec, rec, spec, f1, param, flops))


    # ConfusionMatrix_test = pd.DataFrame(ConfusionMatrix_test, index=['Normal','Pneumonia','COVID-19'], columns=['Normal','Pneumonia','COVID-19'])
    # print(ConfusionMatrix_test)   # 混淆矩阵


    # log.info("VALIDATION | Accuracy {:.4f} | F1 {:.4f} | Precision {:.4f} | "
    #          "Recall {:.4f}".format(acc, f1, prec, rec))
    log.info("Validation end")

    # #绘制ROC曲线
    test_trues = label_binarize(gts, classes=[i for i in range(3)])
    test_preds = label_binarize(predictions, classes=[i for i in range(3)])

        # Visualisation with plot_metric :
    mc = MultiClassClassification(gts, predic, labels=[0, 1, 2])

    plot_roc_auc(mc,net)
    #PR曲线
    # get_pr(test_trues, test_preds,net)

    #结果写人文件
    fields = [
       "Class", "Precision", "Recall", "Specificity", "F1-Score", "Support", "Param", "Flops"
    ]

    rows = []
    rows.append([results[4], results[5], results[6], specificitys[0],results[7],results[8]])
    rows.append([results[9], results[10], results[11], specificitys[1],results[12],results[13]])
    rows.append([results[14], results[15], results[16], specificitys[2],results[17],results[18]])
    rows.append([net,round(acc,4), round(prec,4), round(rec,4), round(spec,4), round(f1,4), round(param,2), round(flops,2)])

    log_file = osp.join('/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/outputs/'+ 'results' +'.csv')
    if osp.isfile(log_file):
        with open(log_file, 'a') as logger:
            csv_writer = csv.writer(logger)
            csv_writer.writerows(rows)

    else:
        with open(log_file, 'w') as logger:
            csv_writer = csv.writer(logger)
            csv_writer.writerow(fields)
            csv_writer.writerows(rows)

    return best_score
    # return predictions


def main(net):
    log.info("Loading val dataset")
    # train_dataset = COVIDxFolder(config.train_imgs, config.train_labels,
    #                                 transforms.train_transforms(config.width,
    #                                                             config.height))
    # train_loader = DataLoader(train_dataset,
    #                         batch_size=config.batch_size,
    #                         shuffle=True,
    #                         drop_last=True,
    #                         num_workers=config.n_threads,
    #                         pin_memory=use_gpu)  

    val_dataset = COVIDxFolder(config.train_imgs, config.test_labels,
                                transforms.val_transforms(config.width,
                                                            config.height))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.val_batch_size,
                            shuffle=False,
                            num_workers=config.n_threads,
                            pin_memory=use_gpu)
    log.info("Number of validation examples {}".format(len(val_dataset)))

    if net =='resnet50-nocutmix':
        weight = config.weights[0]
    if net =='resnet50':
        # weight = config.weights[1]
        weight = "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet50/resnet50_Acc_96.89_epoch_31.pth"
    if net == 'resnet_csra':
        # /root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_csra/resnet_csra_Acc_96.79_epoch_45.pth
        weight = "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_csra/resnet_csra_Acc_96.79_epoch_45.pth"
    if net == 'vit':
        # weight = config.weights[3]
        weight = "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts2/vit/vit_Acc_96.86_F1_95.47_pre_95.55_rec_95.42_step_24800.pth"
    if net == 'vit_csra':
        weight = "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts2/vit_csra/vit_csra_Acc_96.71_F1_95.23_pre_95.45_rec_95.09_step_28000.pth"
    if net == 'swin':
        # weight = config.weights[5]
        weight = "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts_nocutmix/swin/swin_Acc_96.45_epoch_7.pth"
    if net == 'swin_csra':
        # weight = config.weights[6]
        weight = "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts_augcutmix/swin_csra/swin_csra_Acc_97.15_epoch_35.pth"
    if net == 'cswin':
        weight = config.weights[7]
    if net == 'convnext':
        weight = config.weights[8]
    if net == 'convmixer':
        weight = config.weights[9]
    if net == 'poolformer':
        weight = config.weights[10]
    if net == 'volo':
        weight = config.weights[11]
    if net == 'inception_v3':
        weight = config.weights[12]
    if net == 'resnest50d':
        weight = config.weights[13]
    if net == 'covidnet':
        weight = config.weights[14]
    if net == 'resnet_fpn':
        weight = config.weights[15]
    if net == "resnet_MCRA":
        weight = config.weights[16]
    if net == "vgg19":
        weight = config.weights[17]
    if net == "mobilenetv3":
        weight = config.weights[18]
    if net == "efficientnet":
        weight = config.weights[19]
    if net == "densenet":
        weight = config.weights[20]
    if net == "deit_s":
        weight = config.weights[21]
    if net == "regnety_8gf":
        weight = config.weights[22]
    if net == "tnt":
        weight = config.weights[23]
    if net == "xception":
        weight = config.weights[24]
    if net == "cspresnet":
        weight = config.weights[25]
    if net == "resnet_MCRA":
        # weight = config.weights[26]
        weight = "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_96.94_epoch_55.pth"

    weights = [
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_96.95_epoch_73.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_96.97_epoch_74.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.00_epoch_68.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.02_epoch_75.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.02_epoch_79.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.04_epoch_58.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.04_epoch_65.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.05_epoch_72.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.10_epoch_63.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.12_epoch_77.pth",
        "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.12_epoch_78_best.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.13_epoch_62.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.13_epoch_67.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.15_epoch_71.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.18_epoch_70.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.18_epoch_76.pth"
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.30_epoch_106.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.30_epoch_101.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.28_epoch_99.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.28_epoch_84.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.26_epoch_139.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.26_epoch_136.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.26_epoch_105.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.26_epoch_103.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.26_epoch_91.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.25_epoch_146.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.25_epoch_113.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.25_epoch_95.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.25_epoch_85.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.25_epoch_81.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.25_epoch_76.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.23_epoch_121.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.23_epoch_108.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.23_epoch_96.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.23_epoch_92.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.23_epoch_88.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.23_epoch_80.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.22_epoch_114.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.22_epoch_112.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.22_epoch_111.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.22_epoch_109.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.22_epoch_90.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.22_epoch_83.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.20_epoch_130.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.20_epoch_107.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.20_epoch_102.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.20_epoch_100.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.20_epoch_98.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.20_epoch_94.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.18_epoch_141.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.18_epoch_140.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.18_epoch_124.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.18_epoch_97.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.18_epoch_76.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.18_epoch_72.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.18_epoch_70.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.18_epoch_56.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.17_epoch_149.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.17_epoch_144.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.17_epoch_138.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.17_epoch_127.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.17_epoch_120.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.17_epoch_110.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.17_epoch_93.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.17_epoch_87.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.17_epoch_86.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.17_epoch_70.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.15_epoch_143.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.15_epoch_142.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.15_epoch_129.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.15_epoch_123.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.15_epoch_117.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.15_epoch_115.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.15_epoch_82.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.15_epoch_77.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.15_epoch_75.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.15_epoch_71.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.15_epoch_54.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.13_epoch_145.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.13_epoch_126.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.13_epoch_116.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.13_epoch_67.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.13_epoch_62.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.12_epoch_137.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.12_epoch_135.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.12_epoch_118.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.12_epoch_77.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.12_epoch_67.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.10_epoch_147.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.10_epoch_134.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.10_epoch_133.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.10_epoch_132.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.10_epoch_131.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.10_epoch_63.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.08_epoch_125.pth",
        # "/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts/resnet_MCRA/resnet_MCRA_Acc_97.08_epoch_89.pth"
    ] 

    for weight in weights:
        if weight:
            state = torch.load(weight)
            # state = None
            log.info("Loaded model weights from: {}".format(weight))
        else:
            state = None

        state_dict = state["state_dict"] if state else None
        model = architecture.COVIDNext50(net=net,n_classes=config.n_classes)
        if state_dict:
            model = util.load_model_weights(model=model, state_dict=state_dict)

        if use_gpu:
            model.cuda()
            model = torch.nn.DataParallel(model)

        log.info("Number of validation examples {}".format(len(val_dataset)))

        # validate(net,val_loader,model,best_score=0,global_step=0,cfg=config)
    # exit()

        evaluate(net,val_loader,model,best_score=0,global_step=0,cfg=config)
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
    
    # main(net = 'resnet50-nocutmix')
    # main(net = 'resnet50')
    # main(net = 'resnet_csra')
    # main(net = 'vit')
    # main(net = 'vit_csra')
    # main(net = 'swin')
    # main(net = 'swin_csra')
    # main(net = 'cswin')
    # main(net = 'convnext')
    # main(net = 'convmixer')
    # main(net = 'inception_v3')
    # main(net = 'covidnet')
    # main(net = 'vgg19')
    # main(net = 'deit_s')
    # main(net = 'efficientnet')
    # main(net = 'densenet')
    # main(net = 'tnt')
    # main(net = 'xception')

    # main(net = 'regnety_8gf')
    # main(net = 'poolformer')
    # main(net = 'volo')

    main(net = 'resnet_MCRA')


    # main(net = 'cspresnet')
    # main(net = 'resnest50d')
    # main(net = 'resnet_fpn')
    # main(net = 'vit_MCRA')
    # main(net = 'mobilenetv3')


