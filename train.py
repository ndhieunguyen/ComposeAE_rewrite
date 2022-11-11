from img_text_composition_model import ComposeAE
import torchvision
import sys
import gc
import datasets
import torch
from tqdm import tqdm
import time
import test_retrieval
import numpy as np
import socket
import datetime
import os
from torchsummary import summary

def load_dataset(opt):
    """Loads the input datasets."""
    print('Reading dataset ', opt['dataset'])
    if opt['dataset'] == 'fashion200k':
        trainset = datasets.Fashion200k(
            path=opt['dataset_path'],
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.Fashion200k(
            path=opt['dataset_path'],
            split='test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))

    elif opt['dataset'] == 'mitstates':
        trainset = datasets.MITStates(
            path=opt['dataset_path'],
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.MITStates(
            path=opt['dataset_path'],
            split='test',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))

    elif opt['dataset'] == 'fashionIQ':
        trainset = datasets.FashionIQ(
            path=opt['dataset_path'],
            cat_type=opt['category_to_train'],
            split='train',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
        testset = datasets.FashionIQ(
            path=opt['dataset_path'],
            cat_type=opt['category_to_train'],
            split='val',
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
            ]))
    else:
        print('Invalid dataset', opt['dataset'])
        sys.exit()

    print('trainset size:', len(trainset))
    print('testset size:', len(testset))
    return trainset, testset

def create_model(opt):
    text_embed_dim = 512 if not opt['use_bert'] else 768
    trainset, testset = load_dataset(opt)
    texts = [t for t in trainset.get_all_texts()]
    model = ComposeAE(text_query=texts, 
                    image_embed_dim=512,
                    text_embed_dim=text_embed_dim,
                    use_bert=True)
    model = model.cuda()

    params = [{
        'params': [p for p in model.img_model.fc.parameters()],
        'lr': opt['learning_rate']
    }, {
        'params': [p for p in model.img_model.parameters()],
        'lr': 0.1 * opt['learning_rate']
    }, {'params': [p for p in model.parameters()]}]

    for _, p1 in enumerate(params):  # remove duplicated params
        for _, p2 in enumerate(params):
            if p1 is not p2:
                for p11 in p1['params']:
                    for j, p22 in enumerate(p2['params']):
                        if p11 is p22:
                            p2['params'][j] = torch.tensor(0.0, requires_grad=True)

    optimizer = torch.optim.Adam(params,
                                lr=opt['learning_rate'],
                                momentum=0.9,
                                weight_decay=1e-6)

    return model, optimizer

def train_loop(opt, loss_weights, logger, trainset, testset, model, optimizer):
    """Function for train loop"""
    print('Begin training')
    print(len(trainset.test_queries), len(testset.test_queries))
    torch.backends.cudnn.benchmark = True
    losses_tracking = {}
    it = 0
    epoch = -1
    tic = time.time()
    l2_loss = torch.nn.MSELoss().cuda()

    while it < opt['num_iters']:
        epoch += 1

        # show/log stats
        print('It', it, 'epoch', epoch, 'Elapsed time', round(time.time() - tic,
                                                              4), opt['comment'])
        tic = time.time()
        for loss_name in losses_tracking:
            avg_loss = np.mean(losses_tracking[loss_name][-len(trainloader):])
            print('    Loss', loss_name, round(avg_loss, 4))
            logger.add_scalar(loss_name, avg_loss, it)
        logger.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], it)

        if epoch % 1 == 0:
            gc.collect()

        # test
        if epoch % 3 == 1:
            tests = []
            for name, dataset in [('train', trainset), ('test', testset)]:
                if opt['dataset'] == 'fashionIQ':
                    t = test_retrieval.fiq_test(opt, model, dataset)
                else:
                    t = test_retrieval.test(opt, model, dataset)
                tests += [(name + ' ' + metric_name, metric_value)
                          for metric_name, metric_value in t]
            for metric_name, metric_value in tests:
                logger.add_scalar(metric_name, metric_value, it)
                print('    ', metric_name, round(metric_value, 4))

        # save checkpoint
        torch.save({
            'it': it,
            'opt': opt,
            'model_state_dict': model.state_dict(),
        },
            logger.file_writer.get_logdir() + '/latest_checkpoint.pth')

        # run training for 1 epoch
        model.train()
        trainloader = trainset.get_loader(
            batch_size=opt['batch_size'],
            shuffle=True,
            drop_last=True,
            num_workers=opt['loader_num_workers'])

        def training_1_iter(data):
            assert type(data) is list
            img1 = np.stack([d['source_img_data'] for d in data])
            img1 = torch.from_numpy(img1).float()
            img1 = torch.autograd.Variable(img1).cuda()

            img2 = np.stack([d['target_img_data'] for d in data])
            img2 = torch.from_numpy(img2).float()
            img2 = torch.autograd.Variable(img2).cuda()

            if opt['use_complete_text_query']:
                if opt['dataset'] == 'mitstates':
                    supp_text = [str(d['noun']) for d in data]
                    mods = [str(d['mod']['str']) for d in data]
                    # text_query here means complete_text_query
                    text_query = [adj + " " + noun for adj, noun in zip(mods, supp_text)]
                else:
                    text_query = [str(d['target_caption']) for d in data]
            else:
                text_query = [str(d['mod']['str']) for d in data]
            # compute loss
            if opt['loss'] not in ['soft_triplet', 'batch_based_classification']:
                print('Invalid loss function', opt['loss'])
                sys.exit()

            losses = []
            if_soft_triplet = True if opt['loss'] == 'soft_triplet' else False
            loss_value, dct_with_representations = model.compute_loss(img1,
                                                                      text_query,
                                                                      img2,
                                                                      soft_triplet_loss=if_soft_triplet)

            loss_name = opt['loss']
            losses += [(loss_name, loss_weights[0], loss_value.cuda())]

            if opt['model'] == 'composeAE':
                dec_img_loss = l2_loss(dct_with_representations["repr_to_compare_with_source"],
                                   dct_with_representations["img_features"])
                dec_text_loss = l2_loss(dct_with_representations["repr_to_compare_with_mods"],
                                        dct_with_representations["text_features"])

                losses += [("L2_loss", loss_weights[1], dec_img_loss.cuda())]
                losses += [("L2_loss_text", loss_weights[2], dec_text_loss.cuda())]
                losses += [("rot_sym_loss", loss_weights[3], dct_with_representations["rot_sym_loss"].cuda())]
            elif opt['model'] == 'RealSpaceConcatAE':
                dec_img_loss = l2_loss(dct_with_representations["repr_to_compare_with_source"],
                                   dct_with_representations["img_features"])
                dec_text_loss = l2_loss(dct_with_representations["repr_to_compare_with_mods"],
                                        dct_with_representations["text_features"])

                losses += [("L2_loss", loss_weights[1], dec_img_loss.cuda())]
                losses += [("L2_loss_text", loss_weights[2], dec_text_loss.cuda())]

            total_loss = sum([
                loss_weight * loss_value
                for loss_name, loss_weight, loss_value in losses
            ])
            assert not torch.isnan(total_loss)
            losses += [('total training loss', None, total_loss.item())]

            # track losses
            for loss_name, loss_weight, loss_value in losses:
                if loss_name not in losses_tracking:
                    losses_tracking[loss_name] = []
                losses_tracking[loss_name].append(float(loss_value))

            torch.autograd.set_detect_anomaly(True)

            # gradient descendt
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        for data in tqdm(trainloader, desc='Training for epoch ' + str(epoch)):
            it += 1
            training_1_iter(data)

            # decay learning rate
            if it >= opt['learning_rate_decay_frequency'] and it % opt['learning_rate_decay_frequency'] == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= 0.1

    print('Finished training')


if __name__ == '__main__':
    opt = {
            'comment': None,
            'dataset': 'fashion200k',
            'dataset_path': None,
            'model': 'composeAE',
            'image_embed_dim': 512,
            'use_bert': False,
            'use_complete_text_query': False,
            'learning_rate': 1e-2,
            'learning_rate_decay_frequency': 9999999,
            'batch_size': 32,
            'weight_decay': 1e-6,
            'category_to_train': 'all',
            'num_iters': 160000,
            'loss': 'soft_triplet',
            'loader_num_workers': 4,
            'log_dir': '/',
            'test_only': False,
            'model_checkpoint': ''
        }

    for k in opt.keys():
        print('    ', k, ':', str(opt.__dict__[k]))

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    loss_weights = [1.0, 0.1, 0.1, 0.01]
    logdir = os.path.join(opt.log_dir, current_time + '_' + socket.gethostname() + opt.comment)
    model, optimizer = create_model(opt)
    
    summary(model, (3, 224, 224))