# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 11:34:04 2021

@author: ppymcg
"""
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from skimage.color import label2rgb
from skimage import data
from glob import glob
import copy
import datetime
import itertools

class metsat_img:
    def __init__(self,dt,area=None):
        # example for area = (slice(2300,2500),slice(2200,2400))
        self.dt = dt
        if area is not None:
            self.area = area
        files_init = sorted(glob('./**/**IR16**'+str(dt)+'**.jpg'),key = lambda s: s[19:])[0]
        self.dt_initial = files_init[38:-4]
        self.dt_initial = datetime.datetime.strptime(self.dt_initial, '%Y%m%d%H%M%S')
        self.ir = plt.imread(sorted(glob('./**/**IR16**'+str(dt)+'**.jpg'),key = lambda s: s[19:])[0])
        self.v8 = plt.imread(sorted(glob('./**/**VIS8**'+str(dt)+'**.jpg'),key = lambda s: s[19:])[0])
        self.v6 = plt.imread(sorted(glob('./**/**VIS6**'+str(dt)+'**.jpg'),key = lambda s: s[19:])[0])
        self.rgb = np.dstack((self.ir,self.v8,self.v6))
        print(self.dt_initial)
        #print(glob('./**/**IR16**'+str(dt)+'**.jpg')[0])
        
    def day(self):
        try:
            n_days = len(glob('./**/**IR16**'+str(self.dt)[0:8]+'**.jpg'))
            files_day = sorted(glob('./**/**IR16**'+str(int(str(self.dt)[0:8]))+'**.jpg'),key = lambda s: s[19:])
            self.dt_day = []
            for i in range(n_days):
                self.dt_day.append(files_day[i][38:-4])
                
            self.dt_day = [datetime.datetime.strptime(i, '%Y%m%d%H%M%S') for i in self.dt_day]    
                
            self.ir_day = [plt.imread(sorted(glob('./**/**IR16**'+str(self.dt)[0:8]+'**.jpg'),key = lambda s: s[19:])[x]) for x in np.arange(n_days)]
            self.v8_day = [plt.imread(sorted(glob('./**/**VIS8**'+str(self.dt)[0:8]+'**.jpg'),key = lambda s: s[19:])[x]) for x in np.arange(n_days)]
            self.v6_day = [plt.imread(sorted(glob('./**/**VIS6**'+str(self.dt)[0:8]+'**.jpg'),key = lambda s: s[19:])[x]) for x in np.arange(n_days)]
            # stacks them next to each other self.rgb_day = np.dstack((self.ir_day,self.v8_day,self.v6_day))
            self.rgb_day = np.moveaxis(np.stack((self.ir_day,self.v8_day,self.v6_day)),0,-1)
        except:
            print('Only one image in day')
            self.ir_day = self.ir
            self.v8_day = self.v8
            self.v6_day = self.v6
            self.rgb_day = self.rgb
        
    def month(self):
        # first ten days of month
#        month = []
#        for i in range(10):
#            month.append(glob('./**/**IR16**'+str(int(str(self.dt)[0:8])+i)+'**.jpg'))
        n_mon = len(glob('./**/**IR16**'+str(self.dt)[0:6]+'**.jpg'))
        files_mon = sorted(glob('./**/**IR16**'+str(int(str(self.dt)[0:6]))+'**.jpg'),key = lambda s: s[19:])
        self.dt_mon = []
        for i in range(n_mon):
            self.dt_mon.append(files_mon[i][38:-4])
        
        self.dt_mon = [datetime.datetime.strptime(i, '%Y%m%d%H%M%S') for i in self.dt_mon]
        
        self.ir_mon = [plt.imread(sorted(glob('./**/**IR16**'+str(self.dt)[0:6]+'**.jpg'),key = lambda s: s[19:])[x]) for x in np.arange(n_mon)]
        self.v8_mon = [plt.imread(sorted(glob('./**/**VIS8**'+str(self.dt)[0:6]+'**.jpg'),key = lambda s: s[19:])[x]) for x in np.arange(n_mon)]
        self.v6_mon = [plt.imread(sorted(glob('./**/**VIS6**'+str(self.dt)[0:6]+'**.jpg'),key = lambda s: s[19:])[x]) for x in np.arange(n_mon)]
        self.rgb_mon = np.moveaxis(np.stack((self.ir_mon,self.v8_mon,self.v6_mon)),0,-1)
        
    def select(self,selected='initial',area=None):
        if area == None:
            try:
                area = self.area
            except:
                area = input('Please input a selection area in format (slice(yyyy,yyyy),slice(xxxx,xxxx)): ')
        self.area = area
        if selected == 'initial':
            self.select_ir = self.ir[self.area]
            self.select_v8 = self.v8[self.area]
            self.select_v6 = self.v6[self.area]
            self.select_rgb = self.rgb[self.area]
        elif selected == 'day':
            try:
                n_days = len(self.ir_day)
            except:
                self.day()
            if len(self.ir_day)==1:
                self.select_ir_day = self.ir_day[self.area]
                self.select_v8_day = self.v8_day[self.area]
                self.select_v6_day = self.v6_day[self.area]
                self.select_rgb_day = np.stack((self.select_ir_day,self.select_v8_day,self.select_v6_day))
            else:
                self.select_ir_day = [x[self.area] for x in self.ir_day]
                self.select_v8_day = [x[self.area] for x in self.v8_day]
                self.select_v6_day = [x[self.area] for x in self.v6_day]
                self.select_rgb_day = np.moveaxis(np.stack((self.select_ir_day,self.select_v8_day,self.select_v6_day)),0,-1)
        elif selected == 'month':
            try:
                n_mon = len(self.ir_mon)
            except:
                self.month()
            self.select_ir_mon = [x[self.area] for x in self.ir_mon]
            self.select_v8_mon = [x[self.area] for x in self.v8_mon]
            self.select_v6_mon = [x[self.area] for x in self.v6_mon]
            self.select_rgb_mon = np.moveaxis(np.stack((self.select_ir_mon,self.select_v8_mon,self.select_v6_mon)),0,-1)
        else:
            print('No image/s selected')
    
    
    # cloud detection algorithms
    
    # useful functions (for cloud detection algorithms)
    def threshold(self,img, r_up, r_low, g_up, g_low, b_up, b_low):
        # func to return thresholded mask
        mask = ((img[:,:,0] < r_up) & (img[:,:,0] > r_low) & (img[:,:,1] < g_up) & (img[:,:,1] > g_low) & (img[:,:,2] < b_up) & (img[:,:,2] > b_low)).astype(int)
        mask3d=np.stack((mask,mask,mask),axis=2)
        return mask3d

    def iterav(self,selected='initial',base=np.zeros((200,200,3))):

        self.imgs = []
        
        if selected == 'initial':
            try:
                sel = self.select_rgb
            except:
                try:
                    self.select()
                except:
                    self.area = input('Please input a selection area in format (slice(yyyy,yyyy),slice(xxxx,xxxx)): ')
                    self.select()
                sel = self.select_rgb
                
        elif selected == 'day':
            try:
                sel = self.select_rgb_day
            except:
                try:
                    self.select('day')
                except:
                    self.area = input('Please input a selection area in format (slice(yyyy,yyyy),slice(xxxx,xxxx)): ')
                    self.select('day')
                sel = self.select_rgb_day

        elif selected == 'month':
            try:
                sel = self.select_rgb_mon
            except:
                try:
                    self.select('month')
                except:
                    self.area = input('Please input a selection area in format (slice(yyyy,yyyy),slice(xxxx,xxxx)): ')
                    self.select('month')
                sel = self.select_rgb_mon
                
        if sel.ndim==3:
            if np.all(base) == False:
                mask = self.threshold(sel,255,0,255,80,255,80)
                mask2 = self.threshold(sel,100,65,90,50,90,60)
                c = (1-(mask[:,:,0]+mask2[:,:,0]))/255
                c[c<0]=0
                c[c>0]=1
                new_img = (sel*(1-(mask+mask2)))/255
            else:
                mask = np.sum(((sel/255)-base),axis=-1)
                mask[mask<0.1] = 0
                mask[mask>=0.1] =1
                c = (1-(mask))/255
                c[c<0]=0
                c[c>0]=1
                new_img = (sel*(1-np.stack((mask,mask,mask),axis=2))/255)
            new_img = new_img.astype('float')
            new_img[new_img == 0] = np.nan

            self.imgs= new_img
        else:
            c = []
            for s in sel:
                if np.all(base) == False:
                    mask = self.threshold(s,255,0,255,80,255,80)
                    mask2 = self.threshold(s,100,65,90,50,90,60)
                    ci = (1-(mask[:,:,0]+mask2[:,:,0]))/255
                    ci[ci<0]=0
                    ci[ci>0]=1
                    new_img = (s*(1-(mask+mask2)))/255
                else:
                    mask = np.sum(((s/255)-base),axis=-1)
                    mask[mask<0.1] = 0
                    mask[mask>=0.1] =1
                    ci = (1-(mask))/255
                    ci[ci<0]=0
                    ci[ci>0]=1
                    new_img = (s*(1-np.stack((mask,mask,mask),axis=2))/255)
                new_img = new_img.astype('float')
                new_img[new_img == 0] = np.nan
                c.append(ci)
                self.imgs.append(new_img)
        
        if selected == 'initial':
            self.c = c
        elif selected == 'day':
            self.c_day = c
        elif selected == 'month':
            self.c_mon = c
        
        img4av = np.stack(self.imgs,axis=-1)
        self.av = np.nanmean(img4av,axis=-1)

        return self.imgs, self.av

            
    def o_cloud(self,image,area,terrain='neither'):

        if terrain=='sea':
            mask = (plt.imread('earth_landmask.gif')[area][:,:,0])/255
            img = image*mask
            adjust = -10
        
        elif terrain=='land':   
            mask = np.invert(plt.imread('earth_landmask.gif')[area][:,:,0])/255
            img = image*mask
            adjust = -20

        elif terrain == 'neither':
            img = image
            adjust = 0
        else:
            print('No suitable terrain selected, will not apply terrain mask')
            img = image
            adjust = 0

        img = img.astype('float')
        img2 = copy.copy(img)
        img[img==0]=np.nan


        flat = np.ravel(img)
        keep = ~np.isnan(flat)
        flat = flat[keep].astype('int')

        fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
        ax = axes.ravel()
#
        n, bins, patches = ax[1].hist(flat,256)

        m_g = np.mean(flat)
        p_i = n/np.sum(n)
        P_i = np.cumsum(p_i)
        ip_i = np.zeros(len(n))
        for i in range(len(n)):
            ip_i[i] = i*p_i[i]
        mk = np.cumsum(ip_i)
        self.sig_2 = (np.square(m_g*P_i-mk))/(P_i*(1-P_i))
        self.thresh = []
        for i in np.arange(1,len(self.sig_2)-1):
            if self.sig_2[i] > self.sig_2[i+1]:                
                self.thresh.append(i)
        #self.sig_2[self.sig_2>1200]=0
        if len(self.thresh) == 0:
            self.thresh = np.argmax(self.sig_2[10:-10])+adjust
            if self.thresh>200:
                self.thresh = np.rint(self.thresh/2)+adjust
        else:
            if self.thresh[0]>200:
                self.thresh = np.rint(self.thresh[0]/2)
            else:
                self.thresh = self.thresh[0]+adjust

        self.binary = img2 > self.thresh

        if terrain == 'sea':
            self.c_sea = copy.copy(self.binary.astype(int))
            self.c_sea[mask==0]=0
            self.binary = self.c_sea
        else:
            self.binary = self.binary.astype(int)

        ax[0] = plt.subplot(1, 3, 1)
        ax[1] = plt.subplot(1, 3, 2)
        ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

        im_in = ax[0].imshow(img, cmap=plt.cm.gray)
        ax[0].set_title('Original')
        ax[0].axis('off')

        im_hist = ax[1].hist(flat,256)
        ax[1].set_title('Histogram')
        im_hist_line = ax[1].axvline(self.thresh, color='r')

        im_mask = ax[2].imshow(self.binary, cmap=plt.cm.gray)
        ax[2].set_title('Thresholded')
        ax[2].axis('off')

        #plt.show()
        return self.binary.astype(int), self.sig_2, im_in, im_hist, im_hist_line, im_mask

    def g_otsu(self,selected='initial',terrain = 'neither'):

        if selected == 'initial':
            try:
                image_sea = self.select_ir
            except:
                self.select()
#                try:
#                    self.select()
#                except:
#                    self.area = input('Please input a selection area in format (slice(yyyy,yyyy),slice(xxxx,xxxx)): ')
#                    self.select()
            if terrain == 'both':
                image_sea = self.select_v8
                image_land = self.select_v6
                c_sea,ssea,sim_in, sim_hist, sim_hist_line, sim_mask = self.o_cloud(image_sea,self.area,'sea')
                c_land,sland,lim_in, lim_hist, lim_hist_line, lim_mask = self.o_cloud(image_land,self.area,'land')
                c = np.add(c_land,c_sea)
                self.c = (c-1)*(-1)
                im_in = np.array((lim_in, sim_in))
                im_hist = np.array((lim_hist, sim_hist))
                im_hist_line = np.array((lim_hist_line, sim_hist_line))
                im_mask = np.array((lim_mask, sim_mask))
                
            elif terrain == 'neither':
                self.image = np.sqrt((np.square(self.select_ir.astype(float))+np.square(self.select_v8.astype(float))+np.square(self.select_v6.astype(float)))/3)
                cn, sn, im_in, im_hist, im_hist_line, im_mask = self.o_cloud(self.image,self.area)
                self.c = (cn-1)*(-1)

            else:
                print('No suitable terrain selected, will not apply terrain mask')
                image = np.sqrt((self.select_ir)^2+(self.select_v8)^2+(self.select_v6)^2)
                cn, sn, im_in, im_hist, im_hist_line, im_mask = self.o_cloud(image,self.area)
                self.c = (cn-1)*(-1)

            c = self.c    
            mask3d=np.stack((self.c,self.c,self.c),axis=2)
            self.imgs = self.select_rgb*mask3d
            
        elif selected == 'day':
            try:
                image_sea = self.select_ir_day
            except:
                self.select('day')
#                try:
#                    self.select('day')
#                except:
#                    self.area = input('Please input a selection area in format (slice(yyyy,yyyy),slice(xxxx,xxxx)): ')
#                    self.select('day')
            image_sea = self.select_v8_day
            image_land = self.select_v6_day
            self.imgs = []
            c = []
            if terrain == 'both': 
                lim_in = [None] * len(image_land)
                lim_hist = [None] * len(image_land)
                lim_hist_line = [None] * len(image_land)
                lim_mask = [None] * len(image_land)
                sim_in = [None] * len(image_land)
                sim_hist = [None] * len(image_land)
                sim_hist_line = [None] * len(image_land)
                sim_mask = [None] * len(image_land)
                for i in range(len(image_land)):
                    c_land,sland,lim_in[i], lim_hist[i], lim_hist_line[i], lim_mask[i] = self.o_cloud(image_land[i],self.area,'land')
                    c_sea,ssea,sim_in[i], sim_hist[i], sim_hist_line[i], sim_mask[i] = self.o_cloud(image_sea[i],self.area,'sea')
                    ci = np.add(c_land,c_sea)
                    ci = (ci-1)*(-1)
                    c.append(ci)
                    mask3d=np.stack((ci,ci,ci),axis=2)
                    self.imgs.append(self.select_rgb_day[i]*mask3d)
                im_in = np.concatenate((lim_in, sim_in), axis=0)
                im_hist = np.concatenate((lim_hist, sim_hist), axis=0)
                im_hist_line = np.concatenate((lim_hist_line, sim_hist_line), axis=0)
                im_mask = np.concatenate((lim_mask, sim_mask), axis=0)
            elif terrain == 'neither':
                im_in = [None] * len(image_land)
                im_hist = [None] * len(image_land)
                im_hist_line = [None] * len(image_land)
                im_mask = [None] * len(image_land)
                for i in range(len(image_land)):
                    self.image = np.sqrt((np.square(self.select_ir_day[i].astype(float))+np.square(self.select_v8_day[i].astype(float))+np.square(self.select_v6_day[i].astype(float)))/3)
                    cn, sn, im_in[i], im_hist[i], im_hist_line[i], im_mask[i] = self.o_cloud(self.image,self.area)
                    ci = (cn-1)*(-1)
                    c.append(ci)
                    mask3d=np.stack((ci,ci,ci),axis=2)
                    self.imgs.append(self.select_rgb_day[i]*mask3d)
            self.c_day = c
        
        elif selected == 'month':
            try:
                image_sea = self.select_ir_mon
            except:
                self.select('month')
#                try:
#                    self.select('month')
#                except:
#                    self.area = input('Please input a selection area in format (slice(yyyy,yyyy),slice(xxxx,xxxx)): ')
#                    self.select('month')
            image_sea = self.select_v8_mon
            image_land = self.select_v6_mon
            self.imgs = []
            c = []
            if terrain == 'both': 
                lim_in = [None] * len(image_land)
                lim_hist = [None] * len(image_land)
                lim_hist_line = [None] * len(image_land)
                lim_mask = [None] * len(image_land)
                sim_in = [None] * len(image_land)
                sim_hist = [None] * len(image_land)
                sim_hist_line = [None] * len(image_land)
                sim_mask = [None] * len(image_land)
                for i in range(len(image_land)):
                    c_land,sland,lim_in[i], lim_hist[i], lim_hist_line[i], lim_mask[i] = self.o_cloud(image_land[i],self.area,'land')
                    c_sea,ssea,sim_in[i], sim_hist[i], sim_hist_line[i], sim_mask[i] = self.o_cloud(image_sea[i],self.area,'sea')
                    ci = np.add(c_land,c_sea)
                    ci = (ci-1)*(-1)
                    c.append(ci)
                    mask3d=np.stack((ci,ci,ci),axis=2)
                    self.imgs.append(self.select_rgb_mon[i]*mask3d)
                im_in = [None] * len(image_land)
                im_hist = [None] * len(image_land)
                im_hist_line = [None] * len(image_land)
                im_mask = [None] * len(image_land)
                    
            elif terrain == 'neither':
                im_in = np.zeros(len(image_land))
                im_hist = np.zeros(len(image_land))
                im_hist_line = np.zeros(len(image_land))
                im_mask = np.zeros(len(image_land))
                for i in range(len(image_land)):
                    self.image = np.sqrt((np.square(self.select_ir_mon[i].astype(float))+np.square(self.select_v8_mon[i].astype(float))+np.square(self.select_v6_mon[i].astype(float)))/3)
                    cn, sn, im_in[i], im_hist[i], im_hist_line[i], im_mask[i] = self.o_cloud(self.image,self.area)
                    ci = (cn-1)*(-1)
                    c.append(ci)
                    mask3d=np.stack((ci,ci,ci),axis=2)
                    self.imgs.append(self.select_rgb_mon[i]*mask3d)
            self.c_mon = c
        
                
        try:
            if self.imgs.ndim==3:
                self.av = self.imgs
        except:
            for i in range(len(self.imgs)):
                self.imgs[i] = self.imgs[i].astype('float')
                self.imgs[i][self.imgs[i] == 0] = np.nan
            img4av = np.stack(self.imgs,axis=-1)
            self.av = np.nanmean(img4av,axis=-1)

        return self.imgs, self.av, im_in, im_hist, im_hist_line, im_mask

    def cloud_percent(self,selected='initial'):
        if selected == 'initial':
            try:
                c = self.c
            except:
                self.iterav()
                c = self.c
            flat_list = [item for sublist in c for item in sublist]
            count_cloud = flat_list.count(0)
            count_all = c.size
            percent = (count_cloud/count_all)*100
        else:
            if selected=='day':
                try:
                    c = self.c_day
                except:
                    self.iterav('day')
                    c = self.c_day
                labels = self.dt_day
                title = "Change in cloud cover over "+str(self.dt)[0][6:8]+'/'+str(self.dt)[4:6]+'/'+str(self.dt)[0:4]
#                v = [int(x) for x in self.dt_day]
                            
            elif selected=='month':
                try:
                    c = self.c_mon
                except:
                    self.iterav('month')
                    c = self.c_mon
                labels = self.dt_mon
                title = "Change in cloud cover over "+str(self.dt)[4:6]+'/'+str(self.dt)[0:4]
#                v = [int(x) for x in self.dt_mon]
                
            count_cloud = []
            percent = []
            count_all = c[0].size
            for i in range(len(c)):
                self.flat_list = [item for sublist in c[i] for item in sublist]
                count_cloud.append(self.flat_list.count(0))
                percent.append((count_cloud[i]/count_all)*100)
        
        fig, ax = plt.subplots(1)
        ax.plot_date(labels,percent,'o--')
        ax.set_xlabel("Datetime")
        ax.set_ylabel("Percentage Cloud Cover")
        ax.set_title(title)
#        ax.set_xticks(v)
#        ax.set_xticklabels(labels)
        ax.tick_params(labelrotation=45)
        
        plt.show()    
        return percent

# try local thresholding (iterate threshold over local areas in for loop)
#
#    def fill_in(self):
#        for i in range(len(imgs2[:])):
#            fig, (ax1,ax2) = plt.subplots(2)
#            imgs2[i] = np.nan_to_num(imgs2[i])
#            imgs2[i][imgs2[i]==0]=av2[imgs2[i]==0]
#
#            IR_img = plt.imread(IR_files[i])
#            VIS8_img = plt.imread(VIS8_files[i])
#            VIS6_img = plt.imread(VIS6_files[i])
#            RGB_img = np.dstack((IR_img,VIS8_img,VIS6_img))
#            select = RGB_img[2300:2500,2200:2400,:]
#
#            ax1.imshow(select)
#
#            a = av2*255-select
#
#            b = copy.copy(a)
#
#            a[a<0] = 0
#            a[a>0] = 1
#
#            b[b>0] = 0
#            b[b<0] = 1
#
#            arr1 = av2*255*b
#            arr2 = select*a
#
#            t_arr = arr1+arr2
#
#            ax2.imshow(t_arr/255)