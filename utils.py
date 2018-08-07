import os
import torch
import csv

def shuffle(ts, dim=0, inv=False):
    if inv:
        idx = torch.arange(ts.size(dim)-1,-1,step=-1,device=ts.device)
    else:
        idx = torch.randperm(ts.size(dim)).to(ts.device)
    return ts[idx.long()]


def save_log(log, config):
    log_path = os.path.join(config.log_path, 'log.csv')
    write_header = not os.path.exists(log_path)

    with open(log_path, 'a+') as f: 
        f_csv = csv.DictWriter(f, log.keys())
        if write_header:
            f_csv.writeheader()
        f_csv.writerows([log])

    if config.print_log:
        logg = ''
        logg += 'epoch:[{}/{}], iter:[{}/{}], time:{:.7f}\n'.format(log['e'], config.num_epochs, log['iter'], log['niter'], log['time_elapse'])
        logg += 'd_real:{:.4f}, d_fake:{:.4f}, g_fake:{:.4f}\n'.format(log['loss/d_real'], log['loss/d_fake'], log['loss/g_fake'])
        logg += 'zero:{:.4f}, recon:{:.4f}, paral:{:.4f}\n'.format(log['loss/zero'], log['loss/recon'], log['loss/paral'])
        print(logg)

def denorm(x):
    x = ((x+1)/2)
    if torch.is_tensor(x):
        x = x.clamp(0,1)
    return x

def save_model(model, config, log):
    for key, net in model.items():
        torch.save(net.state_dict(), os.path.join(config.model_path, '%s-%d.cpkt'%(key, log['e'])))


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

