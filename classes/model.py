# encoding: utf-8

"""
The main CheXNet model implementation.
"""


import os
import json
import torch
import sklearn
import torchvision.transforms as transforms


from classes.dataset import ChestXrayDataSet
from classes.densenet import DenseNet121


N_CLASSES = 14

CKPT_PATH = '/workspace/pytorch-chexnet/classes/model.pth.tar'

CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
               'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

DATA_DIR = '/home/smirnvla/PycharmProjects/pytorch-chexnet/chestx-ray-data/images'

TEST_IMAGE_LIST = '/home/smirnvla/PycharmProjects/pytorch-chexnet/chestx-ray-data/labels/short_test_list.txt'

BATCH_SIZE = 64

normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.TenCrop(224),
    transforms.Lambda
    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    transforms.Lambda
    (lambda crops: torch.stack([normalize(crop) for crop in crops]))
])


def process(image_list=None):

    if image_list is None:
        image_list = TEST_IMAGE_LIST

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list=image_list,
                                    transform=transform)

    torch.backends.cudnn.benchmark = True

    # initialize and load the model
    model = DenseNet121(N_CLASSES).cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])
    # model = torch.backends.cudnn.convert(model, torch.nn)

    if os.path.isfile(CKPT_PATH):
        print("=> loading checkpoint")
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint")
    else:
        print("=> no checkpoint found")

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=8, pin_memory=True)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    # switch to evaluate mode
    model.eval()

    for i, (inp, target) in enumerate(test_loader):
        target = target.cuda()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda(), volatile=True)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)

    # return pred
    pred = pred.double().cpu().numpy()[0]

    ret = []
    i = 0
    for class_name in CLASS_NAMES:
        ret.append({
            class_name: pred[i]
        })
        # ret[class_name] = pred[i]
        i += 1

    return json.dumps(ret, separators=(',', ':'), sort_keys=True, indent=4)
    # AUROCs = compute_AUCs(gt, pred)
    # AUROC_avg = np.array(AUROCs).mean()
    # print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    # for i in range(N_CLASSES):
    #     print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(sklearn.metrics.roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


