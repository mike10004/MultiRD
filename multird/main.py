#!/usr/bin/env python3

import os
import sys
import argparse, torch, gc, random, json
from pathlib import Path
from multird.data import MyDataset, load_data, label_multihot
from multird.data import Manipulator
from multird.model import Encoder
import multird.utils
from multird.utils import Checkpointer
from multird.utils import History
from tqdm import tqdm
from multird.evaluate import evaluate, evaluate_test
import numpy as np

def main(data_path: str, mode: str, device, frequency: int, batch_size: int = 128, epoch_num: int = 25, quiet: bool = False, output_dir: Path = None):
    output_dir = output_dir or Path("output")
    history = History.create()
    checkpointer = Checkpointer(output_dir, history, mode=mode)
    word2index, index2word, word2vec, index2each, label_size_each, data_idx_each = load_data(data_path, frequency)
    (label_size, label_lexname_size, label_rootaffix_size, label_sememe_size) = label_size_each
    (data_train_idx, data_dev_idx, data_test_500_seen_idx, data_test_500_unseen_idx, data_defi_c_idx, data_desc_c_idx) = data_idx_each
    (index2sememe, index2lexname, index2rootaffix) = index2each
    index2word = np.array(index2word)
    test_dataset = MyDataset(data_test_500_seen_idx + data_test_500_unseen_idx + data_desc_c_idx)
    valid_dataset = MyDataset(data_dev_idx)
    train_dataset = MyDataset(data_train_idx + data_defi_c_idx)

    m = Manipulator(device)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=m.my_collate_fn)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=m.my_collate_fn)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=m.my_collate_fn)
    
    print('DataLoader prepared. Batch_size [%d]'%batch_size)
    print('Train dataset: ', len(train_dataset))
    print('Valid dataset: ', len(valid_dataset))
    print('Test dataset: ', len(test_dataset))
    data_all_idx = data_train_idx + data_dev_idx + data_test_500_seen_idx + data_test_500_unseen_idx + data_defi_c_idx
    
    sememe_num = len(index2sememe)
    wd2sem = m.word2feature(data_all_idx, label_size, sememe_num, 'sememes') # label_size, not len(word2index). we only use target_words' feature
    wd_sems = label_multihot(wd2sem, sememe_num)
    wd_sems = torch.from_numpy(np.array(wd_sems)).to(device) #torch.from_numpy(np.array(wd_sems[:label_size])).to(device)
    lexname_num = len(index2lexname)
    wd2lex = m.word2feature(data_all_idx, label_size, lexname_num, 'lexnames')
    wd_lex = label_multihot(wd2lex, lexname_num)
    wd_lex = torch.from_numpy(np.array(wd_lex)).to(device)
    rootaffix_num = len(index2rootaffix)
    wd2ra = m.word2feature(data_all_idx, label_size, rootaffix_num, 'root_affix')
    wd_ra = label_multihot(wd2ra, rootaffix_num)
    wd_ra = torch.from_numpy(np.array(wd_ra)).to(device)
    mask_s = m.mask_noFeature(label_size, wd2sem, sememe_num)
    mask_l = m.mask_noFeature(label_size, wd2lex, lexname_num)
    mask_r = m.mask_noFeature(label_size, wd2ra, rootaffix_num)
    
    model = Encoder(
        vocab_size=len(word2index),
        embed_dim=word2vec.shape[1],
        hidden_dim=300,
        layers=1,
        class_num=label_size,
        sememe_num=sememe_num,
        lexname_num=lexname_num,
        rootaffix_num=rootaffix_num,
        device=device,
    )
    model.embedding.weight.data = torch.from_numpy(word2vec)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam
    best_valid_accu = 0
    DEF_UPDATE = True
    for epoch in range(epoch_num):
        print('epoch: ', epoch)
        model.train()
        train_loss = 0
        label_list = list()
        pred_list = list()
        for words_t, definition_words_t in tqdm(train_dataloader, disable=quiet):
            optimizer.zero_grad()
            loss, _, indices = model('train', x=definition_words_t, w=words_t, ws=wd_sems, wl=wd_lex, wr=wd_ra, msk_s=mask_s, msk_l=mask_l, msk_r=mask_r, mode=mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            predicted = indices[:, :100].detach().cpu().numpy().tolist()
            train_loss += loss.item()
            label_list.extend(words_t.detach().cpu().numpy())
            pred_list.extend(predicted)
        train_accu_1, train_accu_10, train_accu_100 = evaluate(label_list, pred_list)
        del label_list
        del pred_list
        gc.collect()
        train_loss_mean = train_loss / len(train_dataset)
        print('train_loss: ', train_loss_mean)
        print('train_accu(1/10/100): %.2f %.2F %.2f'%(train_accu_1, train_accu_10, train_accu_100))
        history.train.loss.append(train_loss_mean)
        history.train.add_accuracy(train_accu_1, train_accu_10, train_accu_100)
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            label_list = []
            pred_list = []
            for words_t, definition_words_t in tqdm(valid_dataloader, disable=quiet):
                loss, _, indices = model('train', x=definition_words_t, w=words_t, ws=wd_sems, wl=wd_lex, wr=wd_ra, msk_s=mask_s, msk_l=mask_l, msk_r=mask_r, mode=mode)
                predicted = indices[:, :100].detach().cpu().numpy().tolist()
                valid_loss += loss.item()
                label_list.extend(words_t.detach().cpu().numpy())
                pred_list.extend(predicted)
            valid_accu_1, valid_accu_10, valid_accu_100 = evaluate(label_list, pred_list)
            valid_loss_mean = valid_loss / len(valid_dataset)
            print('valid_loss: ', valid_loss_mean)
            print('valid_accu(1/10/100): %.2f %.2F %.2f'%(valid_accu_1, valid_accu_10, valid_accu_100))
            history.valid.loss.append(valid_loss_mean)
            history.valid.add_accuracy(valid_accu_1, valid_accu_10, valid_accu_100)
            del label_list
            del pred_list
            gc.collect()
            
            if valid_accu_10>best_valid_accu:
                best_valid_accu = valid_accu_10
                print('-----best_valid_accu-----')
                checkpointer.checkpoint(model, epoch)
                #torch.save(model, 'saved.model')
                test_loss = 0
                label_list = []
                pred_list = []
                for words_t, definition_words_t in tqdm(test_dataloader, disable=quiet):
                    indices = model('test', x=definition_words_t, w=words_t, ws=wd_sems, wl=wd_lex, wr=wd_ra, msk_s=mask_s, msk_l=mask_l, msk_r=mask_r, mode=mode)
                    predicted = indices[:, :1000].detach().cpu().numpy().tolist()
                    label_list.extend(words_t.detach().cpu().numpy())
                    pred_list.extend(predicted)
                test_accu_1, test_accu_10, test_accu_100, median, variance = evaluate_test(label_list, pred_list)
                print('test_accu(1/10/100): %.2f %.2F %.2f %.2f %.2f'%(test_accu_1, test_accu_10, test_accu_100, median, variance))
                if epoch>5:
                    json.dump((index2word[label_list]).tolist(), open(mode+'_label_list.json', 'w'))
                    json.dump((index2word[np.array(pred_list)]).tolist(), open(mode+'_pred_list.json', 'w'))
                del label_list
                del pred_list
                gc.collect()
            
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def _run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--frequency', type=int, default=20) # 5~25
    parser.add_argument('-b', '--batch_size', type=int, default=128) # 64
    parser.add_argument('-e', '--epoch_num', type=int, default=25) # 10
    parser.add_argument('-q', '--quiet', default=False, action='store_true')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('-m', '--mode', type=str, default='b')
    parser.add_argument('-sd', '--seed', type=int, default=543624)
    parser.add_argument("-d", "--data-path", metavar="DIR")
    parser.add_argument("-o", "--output", metavar="DIR", help="output directory for models and results")
    args = parser.parse_args()
    setup_seed(args.seed)
    data_path = args.data_path or "./data"
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output or os.path.join("checkpoints", multird.utils.timestamp()))
    main(data_path=data_path,
         device=device,
         mode=args.mode,
         frequency=args.frequency,
         batch_size=args.batch_size,
         epoch_num=args.epoch_num,
         quiet=args.quiet,
         output_dir=output_dir)
    return 0


if __name__ == '__main__':
    sys.exit(_run())
