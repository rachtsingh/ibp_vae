from __future__ import print_function
import sys
import os
import numpy as np
import time
import argparse

import torch
import torch.utils.data
import torch.nn.init
import torch.optim as optim

from torchvision import transforms

from datasets.fixed_binarization import FixedBinarization
from datasets.omniglot import Omniglot

sys.path.append('src')
from utils import save_checkpoint, gen_id
from models.IBP_DGM import IBP_DGM
from models.MFConcrete import MFConcrete
from models.S_IBP_BBVI import S_IBP_BBVI
from models.S_IBP_Concrete import S_IBP_Concrete
from training import mf_bbvi, mf_concrete, s_ibp_bbvi, s_ibp_concrete

parser = argparse.ArgumentParser(description='VAEs for the Indian Buffet Process (IBP)')

parser.add_argument('--dataset', type=str, default='mnist',
                    help='dataset to train on')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log-epoch', type=int, default=1, metavar='N',
                    help='wait every epochs')
parser.add_argument('--train-from', type=str, default=None, metavar='M',
                    help='model to train from, if any')
parser.add_argument('--load-data', type=str, default=None,
                    help='load dataset')
parser.add_argument('--savefile', type=str, required=True, 
                    help='savefile name')
parser.add_argument('--truncation', type=int, default=100,
                    help='number of sticks')
parser.add_argument('--alpha0', type=float, default=10.,
                    help='prior alpha for stick breaking Betas')
parser.add_argument('--repeat-v', type=int, default=1,
                    help='number of v samples to take (to reduce variance on KL)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--uuid', type=str, default=gen_id(), help='(somewhat) unique identifier for the model/job')
parser.add_argument('--hidden', type=int, default=500, help='hidden states')

parser.add_argument('--iwae', type=bool, default=False, help='use IWAE instead of elbo on test')
parser.add_argument('--n-samples', type=int, default=32, help='number of samples for calculating IWAE loss')
# BBVI specific
parser.add_argument('--no-cv', action='store_true', default=False,
                    help='control variates')
parser.add_argument('--n-cv-samples', type=int, default=3, help='number of samples for calculating control variates')
# concrete specific
parser.add_argument('--temp', type=float, default=1.,
                    help='temperature for concrete')
parser.add_argument('--temp_prior', type=float, default=0.5,
                    help='temperature for concrete prior')

SMALL = 1e-16

# determinisim
np.random.seed(0)
torch.manual_seed(0)

global args
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cv = not args.no_cv
print('args:', args)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.cuda:
    newTensor = torch.cuda.DoubleTensor
else:
    newTensor = torch.DoubleTensor

def main(args, model_type):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    if args.dataset == 'mnist':
        train_data = FixedBinarization('../data/', mode='train', download=True, transform=transforms.ToTensor())
        test_data = FixedBinarization('../data/', mode='test', transform=transforms.ToTensor())
        valid_data = FixedBinarization('../data/', mode='valid', transform=transforms.ToTensor())
    elif args.dataset == 'omniglot':
        train_data = Omniglot('data/', train=True, transform=transforms.ToTensor())
        test_data = Omniglot('data/', train=False, transform=transforms.ToTensor())
        valid_data = test_data
    else:
        raise NotImplementedError

    train_loader = torch.utils.data.DataLoader(train_data,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data,
        batch_size=args.batch_size, shuffle=False, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_data,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    def log_likelihood(pred, data):
        """ In this model both are [0., 1.] """
        return data * (pred + SMALL).log() + (1 - data) * (1 - pred + SMALL).log()

    model_kwargs = {
        'dataset': args.dataset,
        'max_truncation_level': args.truncation,
        'alpha0': args.alpha0,
    }
    eval_kwargs = {
        'log_likelihood': log_likelihood,
        'args': args,
    }

    # All switching logic is here.
    if model_type == 'mf_bbvi':
        model_cls = IBP_DGM
        model_kwargs['cv'] = args.cv
        trainer = mf_bbvi.train
        if args.iwae:
            validator = mf_bbvi.calculate_iwae_loss
            eval_kwargs['num_samples'] = args.n_samples
        else:
            validator = mf_bbvi.test
    elif model_type == 'mf_concrete':
        model_cls = MFConcrete
        model_kwargs['temp'] = args.temp
        trainer = mf_concrete.train
        if args.iwae:
            validator = mf_concrete.calculate_iwae_loss
            eval_kwargs['num_samples'] = args.n_samples
        else:
            validator = mf_concrete.test
    elif model_type == 's_ibp_bbvi':
        model_cls = S_IBP_BBVI
        model_kwargs['cv'] = args.cv
        model_kwargs['hidden'] = args.hidden
        trainer = s_ibp_bbvi.train
        if args.iwae:
            validator = s_ibp_bbvi.eval_iwae_loss
            eval_kwargs['num_samples'] = args.n_samples
        else:
            validator = s_ibp_bbvi.test
    elif model_type == 's_ibp_concrete':
        model_cls = S_IBP_Concrete
        model_kwargs['temp'] = args.temp
        model_kwargs['hidden'] = args.hidden
        trainer = s_ibp_concrete.train
        if args.iwae:
            validator = s_ibp_concrete.eval_iwae_loss
            eval_kwargs['num_samples'] = args.n_samples
        else:
            validator = s_ibp_concrete.test

    model = model_cls(**model_kwargs)
    if args.cuda:
        model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    eval_kwargs['model'] = model

    train_scores = np.zeros(args.epochs)
    validation_scores = np.zeros(args.epochs)
    test_scores = np.zeros(args.epochs)
    epoch_times = np.zeros(args.epochs)

    best_valid = 1000
    if args.train_from is not None:
        if os.path.isfile(args.train_from):
            print("=> loading checkpoint '{}'".format(args.train_from))
            checkpoint = torch.load(args.train_from)
            args.start_epoch = checkpoint['epoch']
            best_valid = checkpoint['best_valid']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.train_from, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.train_from))
            sys.exit(1)

    try:
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            train_scores[epoch - 1] = trainer(train_loader, model, log_likelihood, optimizer, epoch, args)
            valid_loss = validator(test_loader=valid_loader, mode='validation', epoch=epoch, **eval_kwargs)
            validation_scores[epoch - 1] = valid_loss
            test_scores[epoch - 1] = validator(test_loader=test_loader, mode='test', epoch=epoch, **eval_kwargs)
            epoch_times[epoch - 1] = time.time() - start

            is_best = valid_loss < best_valid
            best_valid = min(best_valid, valid_loss)  # do something with best_test / save it
            save_checkpoint({
                'args': args,
                'epoch': epoch + 1,
                'best_valid': best_valid,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, is_best,
                filename=os.path.join('models', args.savefile))
            sys.stdout.flush()

        # calculate the training IWAE again
        validator(test_loader=train_loader, mode='train', epoch=epoch, **eval_kwargs)
    except KeyboardInterrupt:
        pass

    # save the trajectories if possible
    if not os.path.exists("runs"):
        os.mkdir("runs")
    np.save(os.path.join("runs", args.savefile + "_train.npy"), train_scores)
    np.save(os.path.join("runs", args.savefile + "_valid.npy"), validation_scores)
    np.save(os.path.join("runs", args.savefile + "_test.npy"), test_scores)
    np.save(os.path.join("runs", args.savefile + "_times.npy"), epoch_times)

