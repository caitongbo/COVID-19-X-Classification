import logging

from zmq.backend import device
from sklearn.metrics import f1_score, precision_score, recall_score, \
     accuracy_score
import torch
import numpy as np
import torchvision.transforms as transforms

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_model_weights(model, state_dict, verbose=True):
    """
    Loads the model weights from the state dictionary. Function will only load
    the weights which have matching key names and dimensions in the state
    dictionary.

    :param state_dict: Pytorch model state dictionary
    :param verbose: bool, If True, the function will print the
        weight keys of parametares that can and cannot be loaded from the
        checkpoint state dictionary.
    :return: The model with loaded weights
    """
    new_state_dict = model.state_dict()
    non_loadable, loadable = set(), set()

    for k, v in state_dict.items():
        if k not in new_state_dict:
            non_loadable.add(k)
            continue

        if v.shape != new_state_dict[k].shape:
            non_loadable.add(k)
            continue

        new_state_dict[k] = v
        loadable.add(k)

    if verbose:
        log.info("### Checkpoint weights that WILL be loaded: ###")
        # {log.info(k) for k in loadable}

        log.info("### Checkpoint weights that CANNOT be loaded: ###")
        # {log.info(k) for k in non_loadable}

    model.load_state_dict(new_state_dict)
    return model


def to_device(tensor, gpu=False):
    """
    Places a Pytorch Tensor object on a GPU or CPU device.

    :param tensor: Pytorch Tensor object
    :param gpu: bool, Flag which specifies GPU placement
    :return: Tensor object
    """
    return tensor.cuda() if gpu else tensor.cpu()


def clf_metrics(predictions, targets, average='macro'):
    f1 = f1_score(targets, predictions, average=average)
    precision = precision_score(targets, predictions, average=average)
    recall = recall_score(targets, predictions, average=average)
    acc = accuracy_score(targets, predictions)

    return acc, f1, precision, recall


def get_learning_rate(optimizer):
    """
    Retrieves the current learning rate. If the optimizer doesn't have
    trainable variables, it will raise an error.
    :param optimizer: Optimizer object
    :return: float, Current learning rate
    """
    if len(optimizer.param_groups) > 0:
        return optimizer.param_groups[0]['lr']
    else:
        raise ValueError('No trainable parameters.')

##add 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]),
    'eval': transforms.Compose([
        transforms.Resize((224, 224)),

        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
}


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def get_all_preds(model, loader):
    model.eval()
    with torch.no_grad():
        all_preds = torch.tensor([], device=device)
        for batch in loader:
            images = batch[0].to(device)
            preds = model(images)
            all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


def get_confmat(targets, preds):
    stacked = torch.stack(
        (torch.as_tensor(targets, device=device),
         preds.argmax(dim=1)), dim=1
    ).tolist()
    confmat = torch.zeros(3, 3, dtype=torch.int16)
    print(3)
    for t, p in stacked:
        confmat[t, p] += 1

    return confmat


def get_results(confmat, classes):
    results = {}
    d = confmat.diagonal()
    for i, l in enumerate(classes):
        tp = d[i].item()
        tn = d.sum().item() - tp
        fp = confmat[i].sum().item() - tp
        fn = confmat[:, i].sum().item() - tp

        accuracy = (tp+tn)/(tp+tn+fp+fn)
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        f1score = (2*precision*recall)/(precision+recall)

        results[l] = [accuracy, recall, precision, f1score]

    return results

dirs = {
    'train': '/home/ubuntu/caitongbo/COVID-Efficientnet-Pytorch/dataset/covidx9/3/train',
    'val': '/home/ubuntu/caitongbo/COVID-Efficientnet-Pytorch/dataset/covidx9/3/test',
    'test': '/home/ubuntu/caitongbo/COVID-Efficientnet-Pytorch/dataset/covidx9/3/test'
}
def deprocess_image(image):
    image = image.cpu().numpy()
    image = np.squeeze(np.transpose(image[0], (1, 2, 0)))
    image = image * np.array((0.229, 0.224, 0.225)) + \
        np.array((0.485, 0.456, 0.406))  # un-normalize
    image = image.clip(0, 1)
    return image

