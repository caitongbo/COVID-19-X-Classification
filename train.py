from genericpath import exists
import logging
import os
import csv
import os.path as osp

import numpy as np
from sklearn.metrics import classification_report
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.dataset import COVIDxFolder
from data import transforms
from torch.utils.data import DataLoader, dataset
from model import architecture

import util
import config
from ranger import Ranger

from cutmix.cutmix import CutMix
from cutmix.utils import CutMixCrossEntropyLoss
from pytorch_loss import FocalLossV1, FocalLossV2, FocalLossV3

import os
from torch import nn, cuda
from transformers import ViTFeatureExtractor, ViTModel, AdamW

from tensorboardX import SummaryWriter
from cutmix2 import CutMixCollator
from cutmix2 import CutMixCriterion

bestacc=0
device = torch.device("cuda" if cuda.is_available() else "cpu")

writer = SummaryWriter()


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def precision(output, target, s=None):
    """Compute the precision"""
    if s:
        output = output*s
    if isinstance(output, tuple):
        output = output[0].data
    accuracy = (output.argmax(dim=1) == target).float().mean().item()
    return accuracy

def save_model(model, config,net):
    global bestacc
    if isinstance(model, torch.nn.DataParallel):
        # Save without the DataParallel module
        model_dict = model.module.state_dict()
    else:
        model_dict = model.state_dict()

    state = {
        "state_dict": model_dict,
        'global_step': config['global_step'],
        "epoch": config['epoch'],
        "accuracy": config['accuracy']
    }
    # f1_macro = config['clf_report']['macro avg']['f1-score'] * 100
    # # acc_macro = config['clf_report']['macro avg']['accuracy'] * 100
    # pre_macro = config['clf_report']['macro avg']['precision'] * 100
    # rec_macro = config['clf_report']['macro avg']['recall'] * 100
    bestacc = bestacc*100
    
    name = "{}_Acc_{:.2f}_epoch_{}.pth".format(net,
                                             bestacc,
                                             config['epoch'])

    model_path = os.path.join(config['save_dir'], net)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, name)

    torch.save(state, model_path)
    log.info("Saved model to {}".format(model_path))


def validate(data_loader, model, best_score, global_step, cfg, net, train_acc, train_loss, epoch):

    global bestacc
    model.eval()
    gts, predictions = [], []


    fields = [
        'epoch', 'global_step','train_loss', 'train_acc', 'valid_loss', 'valid_acc'
    ]
    rows = []

    log.info("Validation started...")
    # loss_fn = CutMixCrossEntropyLoss(True)
    test_criterion = nn.CrossEntropyLoss(reduction='mean')

    loss_meter = AverageMeter()
    correct_meter = AverageMeter()

    with torch.no_grad():
        for step, (data, targets) in enumerate(data_loader):

                data = data.to(device)
                targets = targets.to(device)

                outputs = model(data)
                loss = test_criterion(outputs, targets)

                _, preds = torch.max(outputs, dim=1)

                loss_ = loss.item()
                correct_ = preds.eq(targets).sum().item()
                num = data.size(0)

                loss_meter.update(loss_, num)
                correct_meter.update(correct_, 1)

    accuracy = correct_meter.sum / len(data_loader.dataset)
    log.info('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(
        epoch, loss_meter.avg, accuracy))

    #     imgs, labels = data
    #     imgs = util.to_device(imgs, gpu=cfg.gpu)

    #     with torch.no_grad():
    #         logits = model(imgs)
    #         probs = model.module.probability(logits)
    #         preds = torch.argmax(probs, dim=1).cpu().numpy()

    #     valid_loss = loss_fn(logits, labels)

    #     labels = labels.cpu().detach().numpy()

    #     predictions.extend(preds)
    #     gts.extend(labels)


    # predictions = np.array(predictions, dtype=np.int32)
    # gts = np.array(gts, dtype=np.int32)
    # acc, f1, prec, rec = util.clf_metrics(predictions=predictions,
    #                                       targets=gts,
    #                                       average="macro")
    # report = classification_report(gts, predictions, output_dict=True)


    # log.info("VALIDATION | Accuracy {:.4f} | F1 {:.4f} | Precision {:.4f} | "
    #          "Recall {:.4f}".format(acc, f1, prec, rec))


    # if acc > best_score:
    save_config = {
                'name': net,
                'save_dir': config.ckpts_dir,
                'epoch': epoch,
                'global_step': global_step,
                'accuracy': accuracy
            }
    bestacc = accuracy
    save_model(model=model, config=save_config,net=net)
    best_score = accuracy

    log.info("Validation end")

    rows.append([epoch, global_step, train_loss, train_acc, loss_meter.avg, accuracy])

    log_file = osp.join('/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/outputs/CSVs/'+ net +'.csv')
    if osp.isfile(log_file):
        with open(log_file, 'a') as logger:
            csv_writer = csv.writer(logger)
            csv_writer.writerows(rows)

    else:
        with open(log_file, 'w') as logger:
            csv_writer = csv.writer(logger)
            csv_writer.writerow(fields)
            csv_writer.writerows(rows)

    model.train()
    return best_score

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

def main(net):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config.gpu and not torch.cuda.is_available():
        raise ValueError("GPU not supported or enabled on this system.")
    use_gpu = config.gpu

    log.info("Loading train dataset")
    train_dataset = COVIDxFolder(config.train_imgs, config.train_labels,
                                 transforms.train_transforms(config.width,
                                                             config.height))
    #cutmix methods
    # if config.cutmix:
    #     print("CutMix")
    #     train_dataset = CutMix(train_dataset, num_class=3, beta=1.0, prob=0.5, num_mix=2)    # this is paper's original setting for cifar.
    #     print("No CutMix")

    if config.cutmix:
        collator = CutMixCollator(1.0)
        print("Cutmix use")
    else:
        collator = torch.utils.data.dataloader.default_collate

    train_loader =torch.utils.data.DataLoader(train_dataset,
                                                batch_size=config.batch_size,
                                                shuffle=True,
                                                drop_last=True,
                                                collate_fn=collator,
                                                num_workers=config.n_threads,
                                                pin_memory=use_gpu)

    log.info("Number of training examples {}".format(len(train_dataset)))
    log.info("Loading val dataset")
    val_dataset = COVIDxFolder(config.val_imgs, config.val_labels,
                               transforms.val_transforms(config.width,
                                                         config.height))
    val_loader = DataLoader(val_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=config.n_threads,
                            pin_memory=use_gpu)
    log.info("Number of validation examples {}".format(len(val_dataset)))

    # resume checkpoint
    resume = False
    if resume:
        # state = torch.load(weight)
        state = torch.load('/root/workspace/data/ctb/COVID-19-X/COVID-Net-Pytorch/models/ckpts_nocutmix/resnet_csra/resnet_csra_Acc_96.05_epoch_35.pth')
        # log.info("Loaded model weights from: {}".format(weight))
    else:
        state = None

    state_dict = state["state_dict"] if state else None
    model = architecture.COVIDNext50(net,n_classes=config.n_classes)

    if state_dict:
        model = util.load_model_weights(model=model, state_dict=state_dict)

    if use_gpu:
        model.cuda()
        model = torch.nn.DataParallel(model)
    optim_layers = filter(lambda p: p.requires_grad, model.parameters())

    # optimizer and lr scheduler
    # optimizer = Adam(optim_layers,
    #                  lr=config.lr,
    #                  weight_decay=config.weight_decay)
    optimizer = Ranger(optim_layers,
                    lr=config.lr,
                    weight_decay=config.weight_decay)

    scheduler = ReduceLROnPlateau(optimizer=optimizer,
                                  factor=config.lr_reduce_factor,
                                  patience=config.lr_reduce_patience,
                                  mode='max',
                                  min_lr=1e-7)

    # Load the last global_step from the checkpoint if existing
    global_step = 0 if state is None else state['global_step'] + 1

    class_weights = util.to_device(torch.FloatTensor(config.loss_weights),
                                   gpu=use_gpu)
                       
    # loss_fn = CutMixCrossEntropyLoss(True)
    if config.cutmix:
        train_criterion = CutMixCriterion(reduction='mean')
    else:
        train_criterion = nn.CrossEntropyLoss(reduction='mean')

    # loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05,disable_torch_grad_focal_loss=True)

    # Reset the best metric score
    best_score = -1
    valid_loss_min = np.Inf


    for epoch in range(config.epochs):
        log.info("Started epoch {}/{}".format(epoch + 1,
                                              config.epochs))
        gts, predictions = [], []
        
        loss_meter = AverageMeter()
        accuracy_meter = AverageMeter()

        for step, (data, targets) in enumerate(train_loader):
            global_step += 1

            data = data.to(device)
            if isinstance(targets, (tuple, list)):
                targets1, targets2, lam = targets
                targets = (targets1.to(device), targets2.to(device), lam)
            else:
                targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(data)

            loss = train_criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            _, preds = torch.max(outputs, dim=1)

            loss_ = loss.item()

            num = data.size(0)
            if isinstance(targets, (tuple, list)):
                targets1, targets2, lam = targets
                correct1 = preds.eq(targets1).sum().item()
                correct2 = preds.eq(targets2).sum().item()
                accuracy = (lam * correct1 + (1 - lam) * correct2) / num
            else:
                correct_ = preds.eq(targets).sum().item()
                accuracy = correct_ / num

            loss_meter.update(loss_, num)
            accuracy_meter.update(accuracy, num)


            lr = util.get_learning_rate(optimizer)
            log.info("TRAINING batch: Loss {:.4f} | LR {:.2e} | Global_step {}".format(loss_, lr, global_step)) #使用CutMix


        # predictions = np.array(predictions, dtype=np.int32)
        # gts = np.array(gts, dtype=np.int32)
        # acc, f1, precision, recall = util.clf_metrics(predictions, gts)  ##使用CutMix时，训练过程metrics失效，需关闭

        # log.info("Epoch {} | TRAINING batch: Loss {:.4f} | Accuracy {:.4f} |  "
        #             "Precison {:.4f} | Recall {:.4f} | F1 {:.4f} | LR {:.2e}".format(epoch,
        #                                                 loss.item(),
        #                                                 f1, acc,precision,recall,
        #                                                 lr))
        # log.info("Step {} | TRAINING batch: Loss {:.4f} | LR {:.2e}".format(global_step,loss.item(), lr)) #使用CutMix
        # writer.add_scalar('train/acc',acc, epoch)
        # writer.add_scalar('train/loss', loss.item(), epoch)   
        log.info('Epoch {} Step {}/{} '
                    'Loss {:.4f} ({:.4f}) '
                    'Accuracy {:.4f} ({:.4f})'.format(
                        epoch,
                        step,
                        len(train_loader),
                        loss_meter.val,
                        loss_meter.avg,
                        accuracy_meter.val,
                        accuracy_meter.avg,
                    ))
        train_acc = accuracy_meter.avg
        train_loss = loss_meter.avg

            # if global_step % config.eval_steps == 0 and global_step > 0:
        best_score = validate(val_loader,
                                model,
                                best_score=best_score,
                                global_step=global_step,
                                cfg=config,
                                net=net,
                                train_acc = train_acc,
                                train_loss = train_loss,
                                epoch=epoch)
        scheduler.step(best_score)


if __name__ == '__main__':
    seed = config.random_seed
    if seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    # main(net = 'resnet50')
    # main(net = 'resnet_csra')
    # main(net = 'resnet_csra_fpn')
    # main(net = 'vit')
    # main(net = 'vit_csra')
    # main(net = 'vit_MCRA')
    # main(net = 'swin')
    # main(net = 'swin_csra')
    # main(net = 'swin_MCRA')
    # main(net = 'resnet101')
    # main(net = 'coatnet')
    # main(net = 'cvt')
    # main(net = 'cswin')
    # main(net = 'convnext')
    # main(net = 'convmixer')
    # main(net = 'poolformer')
    # main(net = 'volo')
    # main(net = 'inception_v3')
    # main(net = 'resnest50d')
    # main(net = 'covidnet')
    # main(net = 'resnet_fpn')
    main(net = 'resnet_MCRA')
    # main(net = 'vgg19')
    # main(net = 'deit_s')
    # main(net = 'efficientnet')
    # main(net = 'regnety_8gf')
    # main(net = 'mobilenetv3')
    # main(net = 'densenet')
    # main(net = 'tnt')
    # main(net = 'xception')
    # main(net = 'cspresnet')

    

    

    

    


    


    




    

    

    




