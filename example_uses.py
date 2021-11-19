# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:44:52 2021

@author: ppymcg
"""

from metsat_img import *
import numpy as np
import matplotlib.pyplot as plt
import copy
import math

#%%
"A: Initialise the object"

dt = 20190705 # date range for jan 2019
test = metsat_img(dt,(slice(2300,2500),slice(2200,2400))) # pick desired earth surface area using (slice(yyyy,yyyy),slice(xxxx,xxxx))
#%%
"B: Show rgb image of first day/day of datetime string"

fig = plt.figure()
plt.imshow(test.rgb)
plt.axis('off')
#plt.title('RGB Image: '+str(test.dt_initial))
plt.show()

#%%
"C: Show averaged image over month without cloud removal"

test.month()
av_o = np.mean(test.rgb_mon, axis=0)/255
fig = plt.figure()
plt.imshow(av_o)
plt.axis('off')
plt.show()
#%%
"D: Use previously defined area to select desired area for initial image and plot"
test.select()

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)

#fig.suptitle('~15 percent cloud cover day: '+str(test.dt_initial))

ax1.imshow(test.select_ir,cmap='gray')
ax1.set_title('Infrared Band')
ax1.axis('off')
ax2.imshow(test.select_v8,cmap='gray')
ax2.set_title('VIS8 Band')
ax2.axis('off')
ax3.imshow(test.select_v6,cmap='gray')
ax3.set_title('VIS6 Band')
ax3.axis('off')
ax4.imshow(test.select_rgb)
ax4.set_title('RGB Image')
ax4.axis('off')

plt.show()

#%%
"E: Remove cloudy pixel values using threshold values determined via inspection"
img, av = test.iterav()
#%%
"F: Remove cloudy pixel values using global otsu algorithm and save max variance"
img_otsu, av_otsu, im_in, im_hist, im_hist_line, im_mask = test.g_otsu()
sel_sig_2_otsu = copy.copy(test.sig_2[copy.copy(test.thresh)])
#%%
"G: Remove cloudy pixel values using land masks and iteration of otsu algorithm and save max variance"
bimg_otsu, bav_otsu, bim_in, bim_hist, bim_hist_line, bim_mask = test.g_otsu(terrain='both')
sel_sig_2_botsu = copy.copy(test.sig_2[int(copy.copy(test.thresh))])
mask3d_l=test.c
#%%
"H: Calculate change in cloud cover percentage over month of selected datetime. If no cloud removal method previously run,"
"automatic cloud removal via inspection thresholds is computed. For specific desired cloud removal methods, run the above"
"desired section (either E, F or G) before this one"
per_cloud_mon = test.cloud_percent('month')

#%%
"I: Plot resulting images of different cloud cover removal methods alongside intial RGB image"
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4)
ax1.imshow(test.select_rgb)
ax1.axis('off')
ax2.imshow(img)
ax2.axis('off')
ax3.imshow(img_otsu)
ax3.axis('off')
ax4.imshow(bimg_otsu)
ax4.axis('off')
plt.show()

#%%
"J: Calculate the separability measure of the threshold values determined using otsu's method for global and masked methods"
sig_2_otsu = np.var(np.sqrt((np.square(test.select_ir.astype(float))+np.square(test.select_v8.astype(float))+np.square(test.select_v6.astype(float)))/3))
eta_otsu = sel_sig_2_otsu/sig_2_otsu

imgs4sig_2 = test.select_v6*mask3d_l.astype('float')
imgs4sig_2[imgs4sig_2 == 0] = np.nan

sig_2_botsu = np.nanvar(imgs4sig_2)
eta_botsu = sel_sig_2_botsu/sig_2_botsu

#%%
"K: Get otsu thresholded images using a land mask over the course of a month"
imgs, av_otsu, ims_in, ims_hist, ims_hist_line, ims_mask = test.g_otsu('month','both')

"Calculate and plot averaged image for above cloud free images"
#imgs_try = np.stack(imgs)
imgs_otsu_av = np.nanmean(imgs, axis=0)/255
fig = plt.figure()
plt.imshow(imgs_otsu_av)
plt.axis('off')
plt.show()

#%%
"L: Get inspection thresholded images over the course of a month"
imgs_iter, av_iter = test.iterav('month')

"Calculate and plot averaged image for above cloud free images"
#imgs_try = np.stack(imgs)
imgs_iter_av = np.nanmean(imgs_iter, axis=0)

#b = [[]]*3
#v = [-1,-1,-1,0,0,1,1,1]
#h = [-1,0,1,-1,1,-1,0,1]
#for i in range(3):
#    b = np.concatenate(np.where(imgs_iter_av[:,:,i] <int(np.nanpercentile(imgs_iter_av[:,:,i],25))/1.1),np.where(imgs_iter_av[:,:,i]==np.nan)) # bad pixels
#    bx = [] # x-coordinates of pixels to replace
#    by = [] # y-coordinates of pixels to replace
#    for j in range(len(b[0])):
#        bx.append(b[i][0][j])
#        by.append(b[i][1][j])
#    for l in range(len(bx)):
#        kern = []
#        for vi,hi in zip(v,h):
#            try: # check if adjacent pixels exist/have none nan value
#                if math.isnan(imgs_iter_av[:,:,i][int(bx[l])+vi][int(by[l])+hi]) == True:
#                    print('no pixel at ('+str(bx[l]+vi)+','+str(by[l]+hi)+')')
#                else:
#                    kern.append(imgs_iter_av[:,:,i][bx[l]+vi][by[l]+hi])
#            except:
#                print('no pixel at ('+str(bx[l]+vi)+','+str(by[l]+hi)+')')
#        # set bad pixel value as mean value of surrounding good pixels
#        imgs_iter_av[:,:,i][bx[l]][by[l]] = np.array(kern).sum()/len(kern)


fig = plt.figure()
plt.imshow(imgs_iter_av)
plt.axis('off')
plt.show()

#%%
"L: "
filled_imgs = copy.copy(imgs)
for i in range(len(imgs[:])):
    filled_imgs[i] = np.nan_to_num(filled_imgs[i])
    filled_imgs[i][filled_imgs[i]==0]=imgs_otsu_av[filled_imgs[i]==0]*255
    
#%%
"M: Plot filled image value for a day"
fig, ax = plt.subplots(1)
ax.imshow(filled_imgs[8]/255)
ax.axis('off')
plt.show()

#%%
"N: Calculate NDVI for monthly average image and filled day"

imgs4ndvi = imgs_otsu_av
#imgs4ndvi = imgs_iter_av

R_NIR_av = imgs4ndvi[:,:,2]
R_VIS_av = imgs4ndvi[:,:,1]

ndvi_av = (R_NIR_av-R_VIS_av)/(R_NIR_av+R_VIS_av)
#
#x = [131,133,170,199,179,178,181,185,171,198,194,198,73,73,156,162,186,194,198,113,109]
#y = [50,56,54,47,67,35,36,33,31,45,47,41,57,58,55,56,45,47,45,66,65]
#n = [0.24,0.28,0.22,0.2,0,0.26,0.2,0.09,0.16,0.15,0.17,0.14,0.14, 0.12, 0.2, 0.2, 0.17, 0.15, 0.17, 0.17, 0.12]
#
#for xi,yi,ni in zip(x,y,n):
#    ndvi_av[x,y]=n

fig, (ax1, ax2) = plt.subplots(1,2)
ax1.imshow(imgs4ndvi )
ax1.axis('off')
ax2.imshow(ndvi_av, cmap='Greens_r')
ax2.axis('off')
plt.show()
#%%
"Over sea Jan"
seatest = metsat_img(dt,(slice(3000,3200),slice(2000,2200)))

#%%
"Over sea July"
seatest1 = metsat_img(20190111,(slice(3000,3200),slice(2000,2200)))

#%%
fig, (ax1, ax2) = plt.subplots(2)
ax1.imshow(imgs[i]/255)
ax1.axis('off')
ax2.imshow(filled_imgs[i]/255)
ax2.axis('off')
plt.show()
#%%



#%%
#"Select desired area for all images in day"
#test.select('day')

"Select desired area for all images in month and cloud percent"
#seatest1.select('month')
jan_otsu = seatest1.cloud_percent('month')

#%%
"If you want to get cloud percent using otsu run this then the above section"
img, avseas = seatest1.g_otsu('month')

#%%
"Plot IR, VIS8, VIS6 and RGB images for initial image"

#%%
import matplotlib.animation as animation
fig, ax = plt.subplots()
ims = []
for i in range(len(test.rgb_mon)):
    im = ax.imshow(test.rgb_mon[i], animated=True)
    if i == 0:
        ax.imshow(test.rgb_mon[0])  # show an initial one first
    ims.append([im])
ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

plt.show()





























