import os
import sys
import argparse
import json
import time
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn import DataParallel
from tensorboardX import SummaryWriter

from config import parser
#from tool.utils import make_dataloder_chesxpert
from tool.utils import make_dataloader_nih, model_name, select_model
from tool.utils import get_optimizer, LoadModel
from tool.utils import SaveModel, weighted_BCELoss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()


def training(model, args):
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, model_name(args)))
    writer.add_text('log', str(args), 0)
    params = model.parameters()
    optimizer = get_optimizer(params, args)

    if args.dataset == "NIH":
        dataloaders, dataset_sizes, class_names = make_dataloader_nih(args)
    else:
        assert "Wrong dataset"

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    best_auc_ave = 0.0
    epoch_init = 1

    best_model_weights = model.state_dict()
    best_auc = []
    iter_num = 0.0

    checkpoint_file = os.path.join(args.model_save_dir, str(model_name(args) + '_' + "checkpoint.pth"))
    bestmodel_file = os.path.join(args.model_save_dir, str(model_name(args) + '_' + "best_model.pth"))

    if os.path.exists(checkpoint_file):
        if args.resume:
            model, optimizer, epoch_init, best_auc_ave = LoadModel(checkpoint_file, model, optimizer, epoch_init,
                                                                   best_auc_ave)
            print("Checkpoint found! Resuming")
        else:
            pass

    print(model)
    start_time = time.time()

    for epoch in range(epoch_init, args.epochs + 1):
        print("Epoch {}/{}".format(epoch, args.epochs))
        print("-" * 10)
        iter_num = 0
        for phase in ["train", "val"]:
            if (phase == "train"):
                model.train(True)
            elif (phase == "val"):
                model.train(False)

            running_loss = 0.0
            output_list = []
            label_list = []
            loss_list = []

            for idx, data in enumerate(tqdm(dataloaders[phase])):
                images, labels, names = data
                images = images.to(device)
                labels = labels.to(device)

                if phase == "train":
                    torch.set_grad_enabled(True)
                else:
                    torch.set_grad_enabled(False)

                P = 0
                N = 0
                for label in labels:
                    for v in label:
                        if int(v) == 1:
                            P += 1
                        else:
                            N += 1
                if P != 0 and N != 0:
                    BP = (P + N) / P
                    BN = (P + N) / N
                    weights = torch.tensor([BP, BN], dtype=torch.float).to(device)
                else:
                    weights = None
                
                optimizer.zero_grad()
                outputs = model(images)

                # classification loss
                if args.weighted_loss:
                    loss = weighted_BCELoss(outputs, labels, weights = weights) 
                else:
                    loss_func = torch.nn.BCELoss(reduction = "none")
                    loss = loss_func(outputs[:, 0].unsqueeze(dim = -1), 
                                     torch.tensor(labels[:, 0].unsqueeze(dim = -1), dtype = torch.float))
                    
                    loss = torch.where((labels >= 0)[:, 0].unsqueeze(axis = -1), loss,
                                        torch.zeros_like(loss)).mean()

                    for index in range(1, args.num_classes):
                        loss_temp = loss_func(outputs[:, index].unsqueeze(dim = -1), 
                                              torch.tensor(labels[:, index].unsqueeze(dim = -1), dtype = torch.float))
                    
                        loss_temp = torch.where((labels >= 0)[:, index].unsqueeze(axis=-1), loss_temp,
                                                 torch.zeros_like(loss_temp)).mean()
                        loss += loss_temp

                
                if phase == "train":
                    loss.backward()
                    optimizer.step()
                    iter_num += 1

                running_loss += loss.item()
                loss_list.append(loss.item())
                outputs = outputs.detach().to("cpu").numpy()
                labels = labels.detach().to("cpu").numpy()

                for i in range(outputs.shape[0]):
                    output_list.append(np.where(labels[i] >= 0, outputs[i], 0).tolist())
                    label_list.append(np.where(labels[i] >= 0, labels[i], 0).tolist())

                if idx % 100 == 0 and idx != 0:
                    if phase == "train":
                        writer.add_scalar("loss/train_batch", loss.item() / outputs.shape[0], iter_num)
                        try:
                            auc = roc_auc_score(np.array(label_list[-100 * args.batch_size:]),
                                                np.array(output_list[-100 * args.batch_size:])
                            )
                            writer.add_scalar("auc/train_batch", auc, iter_num)
                            print("\nAUC/Train", auc)
                            print("Batch Loss", sum(loss_list) / len(loss_list))
                        except:
                            pass
                
            epoch_loss = running_loss / dataset_sizes[phase]

            # Computing AUC
            epoch_auc_ave = roc_auc_score(np.array(label_list), np.array(output_list))
            epoch_auc = roc_auc_score(np.array(label_list), np.array(output_list), average=None)

            if phase == 'train':
                writer.add_scalar('loss/train', epoch_loss, epoch)
                writer.add_scalar('auc/train', epoch_auc_ave, epoch)
            if phase == 'val':
                    writer.add_scalar('loss/validation', epoch_loss, epoch)
                    writer.add_scalar('auc/validation', epoch_auc_ave, epoch)

            log_str = ""
            log_str += "Loss: {:.4f} AUC: {:.4f} \n\n".format(epoch_loss, epoch_auc_ave)
            for i, c in enumerate(class_names):
                log_str += "{}: {:.4f}  \n".format(c, epoch_auc[i])

            log_str += "\n"
            if phase == "train":
                print("\n\nTraining Phase")
            else:
                print("\n\nValidation Phase")

            print(log_str)
            writer.add_text("log", log_str, iter_num)
            print("Best validation average AUC: ", best_auc_ave)
            print("Average AUC of current epoch: ", epoch_auc_ave)

            # save model with best validation AUC
            if phase == 'train' and epoch % 1 == 0:
                SaveModel(epoch, model, optimizer, best_auc_ave, checkpoint_file)
            if phase == 'val' and epoch_auc_ave > best_auc_ave:
                best_auc = epoch_auc
                print("Rewriting model with AUROC :", round(best_auc_ave, 4), " by model with AUROC : ",
                      round(epoch_auc_ave, 4))
                best_auc_ave = epoch_auc_ave
                print('Model saved to %s' % bestmodel_file)
                print("Saving the best checkpoint")
                SaveModel(epoch, model, optimizer, best_auc_ave, bestmodel_file)

        time_elapsed = time.time() - start_time
        print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
        print("Best val AUC: {:.4f}".format(best_auc_ave))
        print()

        for i, c in enumerate(class_names):
                print('{}: {:.4f} '.format(c, best_auc[i]))

    model.load_state_dict(best_model_weights)

    return model





