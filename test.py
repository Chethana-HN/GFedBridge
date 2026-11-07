#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.data.multimodaldataset import BreastWeakDataset
from src.metrics import per_sample_segmentation_metrics,summarize_metric

CLIENTS=('S1','S2','S3','S4')

def display(values,multiplier):
    mu,sd,_=summarize_metric(values)
    return f'{mu*multiplier:.2f} ± {sd*multiplier:.2f}'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset_root',required=True)
    ap.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--batch_size',type=int,default=1)
    ap.add_argument('--num_workers',type=int,default=0)
    ap.add_argument('--metadata',default='metadata.json')
    a=ap.parse_args()
    device=torch.device(a.device)
    metadata_path=Path(a.metadata).resolve()
    cfg=json.loads(metadata_path.read_text())
    if cfg['format']!='gfedbridge_jit_eval_v1':raise RuntimeError('Incompatible release metadata')
    torch.manual_seed(cfg['seed'])
    values={}
    for public in CLIENTS:
        c=cfg['clients'][public]
        model=torch.jit.load(str(metadata_path.parent/c['checkpoint']),map_location=device).eval()
        dataset=BreastWeakDataset(dataset_root=a.dataset_root,
            client_id=c['dataset_client'],split=c['test_split'],sparse_type=c['sparse_type'],
            mask_suffix=c['mask_suffix'],image_size=(cfg['image_size'],cfg['image_size']),
            base_seed=cfg['seed'],cache_in_memory=True,return_dense_mask=True)
        loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers)
        client={'dice':[],'iou':[],'hd95':[]}
        with torch.inference_mode():
            for raw in loader:
                def tensor(k):
                    x=raw[k]
                    if not torch.is_tensor(x):x=torch.as_tensor(x)
                    if x.ndim==3:x=x.unsqueeze(1)
                    return x.float().to(device,non_blocking=True)
                x=tensor('image');cond=tensor('condition');pos=tensor('positive');neg=tensor('negative')
                box=tensor('box_region');gt=tensor('dense_mask')
                probability=model(x,cond,pos,neg,box)
                current=per_sample_segmentation_metrics(probability=probability,target=gt,
                             threshold=c['threshold'],spacing=(1.0,1.0))
                for metric in client:client[metric].extend(current[metric])
        values[public]=client
        del model
    print('\nGFedBridge | Multimodal [B,B,B,L] | Final Test')
    print(f'{"Client":<12}{"Dice (%)":<22}{"IoU (%)":<22}{"HD95 (px)":<22}')
    print('-'*78)
    for cid in CLIENTS:
        v=values[cid]
        print(f'{cid:<12}{display(v["dice"],100):<22}{display(v["iou"],100):<22}{display(v["hd95"],1):<22}')
    macro={k:[] for k in ('dice','iou','hd95')}
    for cid in CLIENTS:
        for k in macro:
            mu,_,_=summarize_metric(values[cid][k]);macro[k].append(mu)
    print('-'*78)
    print(f'{"Average":<12}{display(macro["dice"],100):<22}{display(macro["iou"],100):<22}{display(macro["hd95"],1):<22}')
    for cid in CLIENTS:
        _,_,n=summarize_metric(values[cid]['hd95'])
        missing=len(values[cid]['hd95'])-n
        if missing:print(f'HD95 warning: {cid} has {missing} infinite cases excluded from finite mean/SD.')

if __name__=='__main__':main()
