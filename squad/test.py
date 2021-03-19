"""Test a model and generate submission CSV.

Usage:
    > python test.py --split SPLIT --load_path PATH --name NAME
    where
    > SPLIT is either "dev" or "test"
    > PATH is a path to a checkpoint (e.g., save/train/model-01/best.pth.tar)
    > NAME is a name to identify the test run

Author:
    Chris Chute (chute@stanford.edu)
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import utilz
import numpy as np

from args import get_test_args
from collections import OrderedDict
from json import dumps
#from models import BiDAF
from qanet.QANet import QANet
from qanet.modules.ema import EMA
from os.path import join
from tensorboardX import SummaryWriter
from tqdm import tqdm
from ujson import load as json_load
from utilz import collate_fn, SQuAD
from util.file_utils import pickle_load_large_file

def main(args):
    # Set up logging
    args.save_dir = utilz.get_save_dir(args.save_dir, args.name, training=False)
    log = utilz.get_logger(args.save_dir, args.name)
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    device, gpu_ids = utilz.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = utilz.torch_from_json(args.word_emb_file)

    # Get model
    log.info('Building model...')
    #model = BiDAF(word_vectors=word_vectors,
    #              hidden_size=args.hidden_size)
    
    ## QANet

    # load word vectors
    wv_tensor = torch.FloatTensor(np.array(pickle_load_large_file('./data/processed/SQuAD/word_emb.pkl'), dtype=np.float32))
    cv_tensor = torch.FloatTensor(np.array(pickle_load_large_file('./data/processed/SQuAD/char_emb.pkl'), dtype=np.float32))
    wv_word2ix = pickle_load_large_file('./data/processed/SQuAD/word_dict.pkl')

    # construct model
    model = QANet(
        wv_tensor,
        cv_tensor,
        400,
        50,
        128,
        num_head=8,
        train_cemb=False,
        pad=wv_word2ix["<PAD>"])

    ## QANet End

    # model = nn.DataParallel(model, gpu_ids)
    log.info(f'Loading checkpoint from {args.load_path}...')
    model = utilz.load_model(model, args.load_path, gpu_ids, return_step=False)
    #model = model.to(device)
    
    #ema = EMA(0.9999)
    #ema.assign(model)
    
    model = model.to(device)
    model.eval() 

    # Get data loader
    log.info('Building dataset...')
    record_file = vars(args)[f'{args.split}_record_file']
    dataset = SQuAD(record_file, args.use_squad_v2)
    data_loader = data.DataLoader(dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  collate_fn=collate_fn)

    # Evaluate
    log.info(f'Evaluating on {args.split} split...')
    nll_meter = utilz.AverageMeter()
    pred_dict = {}  # Predictions for TensorBoard
    sub_dict = {}   # Predictions for submission
    eval_file = vars(args)[f'{args.split}_eval_file']
    with open(eval_file, 'r') as fh:
        gold_dict = json_load(fh)
    with torch.no_grad(), \
            tqdm(total=len(dataset)) as progress_bar:
        for cw_idxs, cc_idxs, qw_idxs, qc_idxs, y1, y2, ids in data_loader:
            # Setup for forward
            cw_idxs = cw_idxs.to(device)
            qw_idxs = qw_idxs.to(device)
            cc_idxs = cc_idxs.to(device)
            qc_idxs = qc_idxs.to(device)
            batch_size = cw_idxs.size(0)

            # Forward
            p1, p2 = model(cw_idxs, cc_idxs, qw_idxs, qc_idxs)
            p1 = F.softmax(p1, dim=1)
            p2 = F.softmax(p2, dim=1)
            y1, y2 = y1.to(device), y2.to(device)
            loss1 = torch.nn.CrossEntropyLoss()(p1, y1)
            loss2 = torch.nn.CrossEntropyLoss()(p2, y2)
            loss = torch.mean(loss1 + loss2) 
            nll_meter.update(loss.item(), batch_size)

            #starts, ends = utilz.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)
            #outer = torch.matmul(p1.unsqueeze(2), p2.unsqueeze(1))
            #for j in range(outer.size()[0]):
            #    outer[j] = torch.triu(outer[j])
                    # outer[j] = torch.tril(outer[j], self.args.ans_limit)
            #a1, _ = torch.max(outer, dim=2)
            #a2, _ = torch.max(outer, dim=1)
            #ymin = torch.argmax(a1, dim=1)
            #iymax = torch.argmax(a2, dim=1)
            #idx2pred, uuid2pred = utilz.convert_tokens(gold_dict, ids.tolist(), ymin.tolist(), ymax.tolist(),args.use_squad_v2)
            
            #idx2pred = {}
            #uuid2pred = {}
            #for qid, p1, p2 in zip(ids.tolist(), starts.tolist(), ends.tolist()):
            #    context = gold_dict[str(qid)]["context"]
            #    spans = gold_dict[str(qid)]["spans"]
            #    uuid = gold_dict[str(qid)]["uuid"]
            #    if args.use_squad_v2 and (p1 == 0 or p2 == 0):
            #        idx2pred[str(qid)] = ''
            #        uuid2pred[uuid] = ''
            #    else:
            #        p1, p2 = p1-1, p2-1
            #    start_idx = spans[p1][0]
            #    end_idx = spans[p2][1]
            #    idx2pred[str(qid)] = context[start_idx: end_idx]
            #    uuid2pred[uuid] = context[start_idx: end_idx]

            # Get F1 and EM scores
            starts, ends = utilz.discretize(p1, p2, args.max_ans_len, args.use_squad_v2)

            # Log info
            progress_bar.update(batch_size)
            if args.split != 'test':
                # No labels for the test set, so NLL would be invalid
                progress_bar.set_postfix(NLL=nll_meter.avg)

            idx2pred, uuid2pred = utilz.convert_tokens(gold_dict,
                                                      ids.tolist(),
                                                      starts.tolist(),
                                                      ends.tolist(),
                                                      args.use_squad_v2)
            pred_dict.update(idx2pred)
            sub_dict.update(uuid2pred)

    # Log results (except for test set, since it does not come with labels)
    if args.split != 'test':
        results = utilz.eval_dicts(gold_dict, pred_dict, args.use_squad_v2)
        results_list = [('NLL', nll_meter.avg),
                        ('F1', results['F1']),
                        ('EM', results['EM'])]
        if args.use_squad_v2:
            results_list.append(('AvNA', results['AvNA']))
        results = OrderedDict(results_list)

        # Log to console
        results_str = ', '.join(f'{k}: {v:05.2f}' for k, v in results.items())
        log.info(f'{args.split.title()} {results_str}')

        # Log to TensorBoard
        tbx = SummaryWriter(args.save_dir)
        utilz.visualize(tbx,
                       pred_dict=pred_dict,
                       eval_path=eval_file,
                       step=0,
                       split=args.split,
                       num_visuals=args.num_visuals)

    # Write submission file
    sub_path = join(args.save_dir, args.split + '_' + args.sub_file)
    log.info(f'Writing submission file to {sub_path}...')
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
        csv_writer = csv.writer(csv_fh, delimiter=',')
        csv_writer.writerow(['Id', 'Predicted'])
        for uuid in sorted(sub_dict):
            csv_writer.writerow([uuid, sub_dict[uuid]])


if __name__ == '__main__':
    main(get_test_args())
