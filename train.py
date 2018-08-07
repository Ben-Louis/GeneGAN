from torchvision.utils import save_image
import torch.nn.functional as F
import time
from utils import *


def train(model, data, config):

    # load model / model to device
    for net in model.values():
        net.to(config.device)
    if config.pretrained_model > 0:
        load_model(model, config)

    # optimizor
    opts = {}
    for key, net in model.items():
        opts[key] = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr, betas=(config.beta, 0.999))

    # data_loader
    data.set_mode('train')
    data_loader = data.get_loader(batch_size=config.batch_size, num_workers=config.batch_size//4, shuffle=True, drop_last=True)

    # constants
    niter = len(data_loader)
    batch_size = config.batch_size
    start_time = time.time()

    log = {}
    log['niter'] = niter

    for e in range(config.num_epochs):
        log['e'] = e+1
        for i, (imgs_pos, imgs_neg) in enumerate(data_loader):
            imgs_pos, imgs_neg = imgs_pos.to(config.device), imgs_neg.to(config.device)

            ####################### train Discriminators #######################
            # get fake images
            feats = model['E'](torch.cat([imgs_pos, imgs_neg]))
            feats[0] = shuffle(feats[0])
            fake_imgs = model['G'](feats)
            fake_imgs_pos, fake_imgs_neg = fake_imgs.split(batch_size)

            # compute loss
            dout_real_pos = model['D_pos'](imgs_pos)
            dout_fake_pos = model['D_pos'](fake_imgs_pos.detach())
            dout_real_neg = model['D_neg'](imgs_neg)
            dout_fake_neg = model['D_neg'](fake_imgs_neg.detach())            

            d_loss_real = F.relu(1.0 - dout_real_pos).mean() + F.relu(1.0 - dout_real_neg).mean()            
            d_loss_fake = F.relu(1.0 + dout_fake_pos).mean() + F.relu(1.0 + dout_fake_neg).mean()

            d_loss = d_loss_real + d_loss_fake

            for key in ['D_pos', 'D_neg']:
                opts[key].zero_grad()
            d_loss.backward()
            for key in ['D_pos', 'D_neg']:
                opts[key].step()  
            #################################################################

            ####################### train Autoencoder #######################
            if (i+1) % config.d_train_repeat == 0:                
                feats = model['E'](torch.cat([imgs_pos, imgs_neg]))

                # constrain
                e_loss_zero = feats[1][batch_size:].abs().mean()

                # generate
                half_batch = batch_size // 2
                feats1 = feats[1].split(half_batch)
                feats[1] = torch.cat([feats1[0], shuffle(feats1[2], inv=True), shuffle(feats1[1], inv=True), feats1[3]])
                #feats[1][half_batch:half_batch+batch_size] = shuffle(feats[1][half_batch:half_batch+batch_size], inv=True)
                fake_imgs = model['G'](feats).split(half_batch)

                gout_fake_pos = model['D_pos'](torch.cat([fake_imgs[0], fake_imgs[2]])).mean()
                gout_fake_neg = model['D_neg'](torch.cat([fake_imgs[1], fake_imgs[3]])).mean()

                # compute loss
                ae_loss_rec = (fake_imgs[0]-imgs_pos[:half_batch]).abs().mean() + \
                              (fake_imgs[-1]-imgs_neg[half_batch:]).abs().mean()
                g_loss_fake = gout_fake_pos + gout_fake_neg
                ae_loss_paral = (imgs_pos[half_batch:]+shuffle(imgs_neg[:half_batch], inv=True)-fake_imgs[1]-shuffle(fake_imgs[2])).abs().mean()

                ae_loss = ae_loss_rec * config.lambda_rec + e_loss_zero * config.lambda_zero \
                          - g_loss_fake + ae_loss_paral * config.lambda_paral

                for key in ['E', 'G']:
                    opts[key].zero_grad()
                ae_loss.backward()
                for key in ['E', 'G']:
                    opts[key].step()  
            #################################################################   

            ### save log ###
            if (i+1) % (config.log_step) == 0:                            

                log['loss/d_real'] = d_loss_real.item()
                log['loss/d_fake'] = d_loss_fake.item()
                log['loss/zero'] = e_loss_zero.item()
                log['loss/recon'] = ae_loss_rec.item()
                log['loss/g_fake'] = g_loss_fake.item()
                log['loss/paral'] = ae_loss_paral.item()

                log['gan/d_real(pos)'] = dout_real_pos.mean().item()
                log['gan/d_real(neg)'] = dout_real_neg.mean().item()            
                log['gan/d_fake(pos)'] = dout_fake_pos.mean().item()
                log['gan/d_fake(neg)'] = dout_fake_neg.mean().item()
                log['gan/g_fake(pos)'] = gout_fake_pos.item()
                log['gan/g_fake(neg)'] = gout_fake_neg.item()
                
                log['iter'] = i + 1                
                log['time_elapse'] = time.time() - start_time

                save_log(log, config)

            ### save images ###
            if (i+1) % (config.sample_step) == 0:
                n = config.num_sample
                pos_imgs, neg_imgs = data.get_test((n*(i+1)//config.sample_step)%(100-n))
                pos_imgs, neg_imgs = pos_imgs.to(config.device), neg_imgs.to(config.device)

                imgs = [torch.cat([torch.ones_like(neg_imgs[0:2]).cpu(),neg_imgs.cpu()])]
                for j in range(len(pos_imgs)):
                    pos_img = pos_imgs[j:j+1]

                    with torch.no_grad():
                        feat_pos = model['E'](pos_img)
                        feats_neg = model['E'](neg_imgs)
                        # use pos to guide neg
                        feats = (feats_neg[0], feat_pos[1].repeat(n,1,1,1))
                        fake_imgs = model['G'](feats)
                        # convert pos
                        feat_pos[1].data.zero_()
                        fake_img = model['G'](feat_pos)

                        fake_imgs = torch.cat([pos_img, fake_img, fake_imgs]).cpu()
                        imgs.append(fake_imgs)

                imgs = torch.cat(imgs, dim=3)
                save_image(denorm(imgs), os.path.join(config.log_path, 'gen_imgs_%d_%d.png'%(e+1,i+1)), nrow=1, padding=0)
                print('saving images successfully!')

        ### save models ###
        save_model(model, config, log)
        print('saving models successfully!')

        ### update lr ###
        if (e+1) > (config.num_epochs - config.num_epochs_decay):
            lr = config.lr * (config.num_epochs - e) / config.num_epochs_decay
            for opt in opts.values():
                for param in opt.param_groups:
                    param['lr'] = lr
            print('update learning rate to {:.6f}'.format(lr))






























