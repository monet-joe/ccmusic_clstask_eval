import csv
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from functools import partial
from datetime import datetime
from model import Net
from datasets import load_dataset
from modelscope.msdatasets import MsDataset
from torch.utils.data import DataLoader
from torchvision.transforms import *
from focalLoss import FocalLoss
from utils import time_stamp, create_dir, toCUDA, results_dir
from plot import save_acc, save_loss, save_confusion_matrix
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")


def transform(example_batch, spect='cqt', labelv='label', input_size=300):
    compose = Compose([
        Resize(input_size),
        CenterCrop(input_size),
        RandomAffine(5),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    inputs = [compose(x.convert('RGB')) for x in example_batch[spect]]
    example_batch[spect] = inputs
    keys = list(example_batch.keys())
    for key in keys:
        if not (key == spect or key == labelv):
            del example_batch[key]

    return example_batch


def prepare_data(labelv):
    print('Preparing data...')
    try:
        ds = load_dataset("ccmusic-database/chest_falsetto")
        classes = ds['test'].features[labelv].names
        use_hf = True
    except ConnectionError:
        ds = MsDataset.load('ccmusic/chest_falsetto', subset_name='default')
        classes = ds['test']._hf_ds.features[labelv].names
        use_hf = False

    if args.fl:
        num_samples_in_each_category = {k: 0 for k in classes}
        for item in ds['train']:
            num_samples_in_each_category[classes[item[labelv]]] += 1

        print('Data prepared.')
        return ds, classes, list(num_samples_in_each_category.values()), use_hf

    else:
        print('Data prepared.')
        return ds, classes, [], use_hf


def load_data(ds, input_size, spect, use_hf, labelv, batch_size=4, shuffle=True, num_workers=2):
    print('Loadeding data...')
    if use_hf:
        trainset = ds['train'].with_transform(
            partial(transform, spect=spect, input_size=input_size, labelv=labelv))
        validset = ds['validation'].with_transform(
            partial(transform, spect=spect, input_size=input_size, labelv=labelv))
        testset = ds['test'].with_transform(
            partial(transform, spect=spect, input_size=input_size, labelv=labelv))
    else:
        trainset = ds['train']._hf_ds.with_transform(
            partial(transform, spect=spect, input_size=input_size, labelv=labelv))
        validset = ds['validation']._hf_ds.with_transform(
            partial(transform, spect=spect, input_size=input_size, labelv=labelv))
        testset = ds['test']._hf_ds.with_transform(
            partial(transform, spect=spect, input_size=input_size, labelv=labelv))

    traLoader = DataLoader(trainset, batch_size=batch_size,
                           shuffle=shuffle, num_workers=num_workers)
    valLoader = DataLoader(validset, batch_size=batch_size,
                           shuffle=shuffle, num_workers=num_workers)
    tesLoader = DataLoader(testset, batch_size=batch_size,
                           shuffle=shuffle, num_workers=num_workers)

    print('Data loaded.')
    return traLoader, valLoader, tesLoader


def eval_model_train(model, spect, labelv, trainLoader, tra_acc_list):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in trainLoader:
            inputs, labels = toCUDA(data[spect]), toCUDA(data[labelv])
            outputs = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    acc = 100.0 * accuracy_score(y_true, y_pred)
    print('Training acc   : ' + str(round(acc, 2)) + '%')
    tra_acc_list.append(acc)


def eval_model_valid(model, spect, labelv, validationLoader, val_acc_list):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in validationLoader:
            inputs, labels = toCUDA(data[spect]), toCUDA(data[labelv])
            outputs = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    acc = 100.0 * accuracy_score(y_true, y_pred)
    print('Validation acc : ' + str(round(acc, 2)) + '%')
    val_acc_list.append(acc)


def eval_model_test(model, spect, labelv, testLoader, classes):
    y_true, y_pred = [], []
    with torch.no_grad():
        for data in testLoader:
            inputs, labels = toCUDA(data[spect]), toCUDA(data[labelv])
            outputs = model.forward(inputs)
            predicted = torch.max(outputs.data, 1)[1]
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    report = classification_report(
        y_true, y_pred, target_names=classes, digits=3)
    cm = confusion_matrix(y_true, y_pred, normalize='all')

    return report, cm


def save_log(start_time, finish_time, cls_report, cm, log_dir, classes):
    log_backbone = f'Backbone      : {args.model}'
    log_spect = f'Spect type    : {args.spect}'
    log_start_time = f'Start time    : {time_stamp(start_time)}'
    log_finish_time = f'Finish time   : {time_stamp(finish_time)}'
    log_time_cost = f'Time cost     : {str((finish_time - start_time).seconds)}s'
    log_fullfinetune = f'Full finetune : {str(args.fullfinetune)}'
    log_focal_loss = f'Focal loss    : {str(args.fl)}'

    with open(log_dir + '/result.log', 'w', encoding='utf-8') as f:
        f.write(cls_report + '\n')
        f.write(log_backbone + '\n')
        f.write(log_spect + '\n')
        f.write(log_start_time + '\n')
        f.write(log_finish_time + '\n')
        f.write(log_time_cost + '\n')
        f.write(log_fullfinetune + '\n')
        f.write(log_focal_loss + '\n')
    f.close()

    # save confusion_matrix
    np.savetxt(log_dir + '/mat.csv', cm, delimiter=',')
    save_confusion_matrix(cm, classes, log_dir)

    print(cls_report)
    print('Confusion matrix :')
    print(str(cm.round(3)) + '\n')
    print(log_backbone)
    print(log_spect)
    print(log_start_time)
    print(log_finish_time)
    print(log_time_cost)
    print(log_fullfinetune)
    print(log_focal_loss)


def save_history(model, tra_acc_list, val_acc_list, loss_list, lr_list, cls_report, cm, start_time, finish_time, classes):
    create_dir(results_dir)
    log_dir = f'{results_dir}/{args.model}_{args.spect}_{time_stamp()}'
    create_dir(log_dir)

    acc_len = len(tra_acc_list)
    with open(log_dir + "/acc.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tra_acc_list", "val_acc_list", "lr_list"])
        for i in range(acc_len):
            writer.writerow([tra_acc_list[i], val_acc_list[i], lr_list[i]])

    loss_len = len(loss_list)
    with open(log_dir + "/loss.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["loss_list"])
        for i in range(loss_len):
            writer.writerow([loss_list[i]])

    torch.save(model.state_dict(), log_dir + '/save.pt')
    print('Model saved.')

    save_acc(tra_acc_list, val_acc_list, log_dir)
    save_loss(loss_list, log_dir)
    save_log(start_time, finish_time, cls_report, cm, log_dir, classes)


def train(backbone_ver, spect, labelv, epoch_num=40, iteration=10, lr=0.001):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tra_acc_list, val_acc_list, loss_list, lr_list = [], [], [], []

    # load data
    ds, classes, num_samples, use_hf = prepare_data(labelv)
    cls_num = len(classes)

    # init model
    model = Net(cls_num, m_ver=backbone_ver, full_finetune=args.fullfinetune)
    input_size = model._get_insize()
    traLoader, valLoader, tesLoader = load_data(
        ds, input_size, spect, use_hf, labelv)

    # optimizer and loss
    criterion = FocalLoss(num_samples) if args.fl else nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True,
        threshold=lr, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08
    )

    # gpu
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        criterion = criterion.cuda()
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    # train process
    start_time = datetime.now()
    print('Start training [' + args.model + '] at ' + time_stamp(start_time))
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        epoch_str = f' Epoch {epoch + 1}/{epoch_num} '
        lr_str = optimizer.param_groups[0]["lr"]
        lr_list.append(lr_str)
        print(f'{epoch_str:-^40s}')
        print(f'Learning rate: {lr_str}')
        running_loss = 0.0
        for i, data in enumerate(traLoader, 0):
            # get the inputs
            inputs, labels = toCUDA(data[spect]), toCUDA(data[labelv])
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print every 2000 mini-batches
            if i % iteration == iteration - 1:
                print('[%d, %5d] loss: %.4f' %
                      (epoch + 1, i + 1, running_loss / iteration))
                loss_list.append(running_loss / iteration)
            running_loss = 0.0

        eval_model_train(model, spect, labelv, traLoader, tra_acc_list)
        eval_model_valid(model, spect, labelv, valLoader, val_acc_list)
        scheduler.step(loss.item())

    finish_time = datetime.now()
    cls_report, cm = eval_model_test(model, spect, labelv, tesLoader, classes)
    save_history(model, tra_acc_list, val_acc_list, loss_list,
                 lr_list, cls_report, cm, start_time, finish_time, classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--model', type=str, default='squeezenet1_1')
    parser.add_argument('--spect', type=str, default='cqt')
    parser.add_argument('--labelv', type=str, default='label')
    parser.add_argument('--fl', type=bool, default=True)
    parser.add_argument('--fullfinetune', type=bool, default=True)
    args = parser.parse_args()

    train(
        backbone_ver=args.model,
        spect=args.spect,
        labelv=args.labelv,
        epoch_num=40
    )
