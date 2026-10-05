import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors, cm
from cloudfield import CloudField
import img as image
from PIL import Image
from scipy.optimize import curve_fit
from scipy.stats import linregress


colors_list = ['blue', 'orange', 'red', 'green', 'darkmagenta', 'magenta', 'darkred']
plt.rcParams['font.size'] = 11

def corr_simple_cases(save=False):

    r_max=100

    fig, ax = plt.subplots(4, 4, figsize=(22,27))

    thresholds = [150, 150, 150, 130, 130]

    norm=colors.TwoSlopeNorm(vmin=-1., vcenter=0., vmax=1.)
    img_norm = colors.TwoSlopeNorm(vmin=0, vcenter=0.5, vmax=1)

    cf_list = [
        CloudField(image.image_to_binary_array('Nuages/convective_rolls.jpg',150), r_max=r_max),
        CloudField(np.ones((500,500), dtype=bool), r_max=r_max),
        CloudField(image.disk(size=500, r=25), r_max=r_max),
        CloudField(image.poisson_field(size=500, cloud_cover=0.3), r_max=r_max)
    ]

    for i, cf in enumerate(cf_list):
        C_xy, C_r, C_angle = [], [], []

        N = 10
        for _ in range(N):
            C_xy_, C_r_, C_angle_ = cf.C(n_draws=30000)
            C_xy.append(C_xy_)
            C_r.append(C_r_)
            C_angle.append(C_angle_)

        C_xy = np.array(C_xy)
        C_r = np.array(C_r)
        C_angle = np.array(C_angle)

        C_xy_mean = np.mean(C_xy, axis=0)
        C_r_mean = np.mean(C_r, axis=0)
        C_angle_mean = np.mean(C_angle, axis=0)

        C_xy_std = np.std(C_xy, axis=0)
        C_r_std = np.std(C_r, axis=0, ddof=1)
        C_angle_std = np.std(C_angle, axis=0)

        ax[i,0].imshow(cf.img, cmap='binary_r', norm=img_norm, origin='lower')
        ax[i,1].imshow(C_xy_mean, cmap='RdBu_r', norm=norm, origin='lower',
                    extent=[-cf.r_max,cf.r_max,-cf.r_max,cf.r_max])
        ax[i,2].plot(np.arange(len(C_r_mean)),
                     C_r_mean, color='black', label=r'C(r)')
        ax[i,2].fill_between(np.arange(len(C_r_mean)),
                             C_r_mean+C_r_std/np.sqrt(N),
                             C_r_mean-C_r_std/np.sqrt(N),
                             color='black', alpha=0.4)
        ax[i,3].plot(np.arange(len(C_angle_mean)), 
                     C_angle_mean, color='black', label=r'$C(\theta)$')
        ax[i,3].fill_between(np.arange(len(C_angle_mean)),
                        C_angle_mean+C_angle_std/np.sqrt(N),
                        C_angle_mean-C_angle_std/np.sqrt(N),
                        color='black', alpha=0.4)  
        if i >= 2:
            ax[i,3].plot(np.arange(len(C_angle_mean)), 
                        np.ones(len(C_angle_mean))*np.mean(C_r_mean),
                        '--', color='red', label=r'$\langle C(r) \rangle$')          

        side = cf.size - 2*cf.r_max
        square = patches.Rectangle((cf.r_max, cf.r_max), side, side, 
                                edgecolor='red', facecolor='none')
        ax[i,0].add_patch(square)


        # LABELS
        ax[i,3].legend()
        ax[i,2].legend()
        ax[i,2].grid()
        if i==3:
            ax[i,1].set_xlabel(r'$r_x$ [px]')
            ax[i,0].set_xlabel(r'$x_x$ [px]')
            ax[i,2].set_xlabel('Radius r [px]')
            ax[i,3].set_xlabel(r'Angle $\theta$ [deg]')

        if i==1: ax[i,1].set_ylabel(r'$r_y$ [px]')
        ax[i,0].set_ylabel(r'$x_y$ [px]')
        ax[i,3].grid()

        ax[i,1].text(-95,45,
                     f'Mean = {C_xy_mean.mean():.2f}\nMax = {C_xy_mean.max():.2f}'  
                    + f'\nMin = {C_xy_mean.min():.2f}',
                    backgroundcolor='white', fontsize=20)
        ax[i,0].text(15,455, f'c.c. = {cf.cloud_cover:.2f}', 
                     backgroundcolor='white', fontsize=20)

        if i==0:
            ax[i,1].set_title(r'Spatial $C(\bf{r})$')
            ax[i,2].set_title(r'Radial $C(r)$')
            ax[i,3].set_title(r'Angular $C(\theta)$')


    fig.colorbar(cm.ScalarMappable(norm=norm, cmap='RdBu_r'), ax=ax, 
                location='top', shrink=0.4, 
                label=r'Spatial correlation $C(\bf{r})$')

    if save:
        plt.savefig('output_figures/corr_simple_cases.pdf', 
                    bbox_inches='tight')

def corr_resolution(save=False):

    #### RESOLUTION SENSITIVITY ANALYSIS ####

    # Image to analyse
    cf_name = 'Nuages/nuagesOI_L_60m.jpg'
    scale = 60e-3 #km/px
    thresh = 150
    size = min(image.image_to_binary_array(cf_name, thresh).shape)
    total_area = size**2 * scale**2

    subsampling = [1, 20, 50, 100]

    fig, ax = plt.subplots(1, 2, figsize=(8,3), sharey=True)


    subscale, C_list = [], []

    for step in subsampling:
        print(len(subscale))

        img = image.image_to_binary_array(cf_name, thresh)
        img = img[:size:step,:size:step] # crop to square and subsample
        subsize = img.shape[0]
        r_max = subsize // 7
        cf = CloudField(img, r_max=r_max)

        subscale.append(step*scale)
        C_xy, C_r, C_theta = cf.C(n_draws=10000)
        C_list.append(C_r)

    subscale = np.array(subscale)
    
    for i, C in enumerate(C_list):
        r = np.arange(len(C)) * subscale[i]
        ax[0].plot(r, C, color=colors_list[i], 
                label=f'{subscale[i]:.2f} km/px')
        mask = r < 10
        ax[1].plot(r[mask], C[mask], '+', color=colors_list[i],
                   label=f'{subscale[i]:.2f} km/px')

    ax[0].set_xlabel(r'Radius $r$ [km]')
    ax[0].set_ylabel(r'Radial correlation $C(r)$')
    ax[0].grid()
    ax[0].legend()
    ax[1].set_xlabel(r'Radius $r$ [km]')
    ax[1].grid()


    if save:
        plt.savefig('output_figures/corr_resolution_analysis.pdf', 
                    bbox_inches='tight')
    

    #### SHOW SUBSAMPLED IMAGES ####

    fig, ax = plt.subplots(1, 4, figsize=(13,3))

    for i, step in enumerate(subsampling):

        img = image.image_to_binary_array(cf_name, thresh)
        size = min(img.shape)
        img = img[:size:step,:size:step] # crop to square and subsample
        subsize = img.shape[0]
        r_max = subsize // 7
        cf = CloudField(img, r_max=r_max)

        ax[i].imshow(img, cmap='binary_r')
        ax[i].set_axis_off()
        ax[i].set_title(f'{step*scale:.2f} km/px - {img.shape}')

        side = cf.size - 2*cf.r_max
        square = patches.Rectangle((cf.r_max, cf.r_max), side, side, 
                                edgecolor=colors_list[i], facecolor='none')
        ax[i].add_patch(square)

    if save:
        plt.savefig('output_figures/corr_resolution_subsampled_img.pdf', 
                bbox_inches='tight')

def corr_threshold(save=False):

    #### RESOLUTION SENSITIVITY ANALYSIS ####

    # Image to analyse
    cf_name = 'Nuages/nuagesOI_L_60m.jpg'
    scale = 60e-3 #km/px
    thresh = [120, 140, 160, 180]
    size = min(image.image_to_binary_array(cf_name, 150).shape)
    total_area = size**2 * scale**2

    fig, ax = plt.subplots(1, 2, figsize=(8,3), sharey=True)

    def f(r, delta, cste):
        return (np.exp(-r/delta) + cste) / (1 + cste)

    p0 = [1.5, 0.1]


    cc, C_list = [], []

    for t in thresh:
        print(len(C_list))

        img = image.image_to_binary_array(cf_name, t)
        img = img[:size,:size] # crop to square
        r_max = size // 4
        cf = CloudField(img, r_max=r_max)

        C_xy, C_r, C_theta = cf.C(n_draws=10000)
        C_list.append(C_r)
        cc.append(cf.cloud_cover)
    
    
    for i, C in enumerate(C_list):
        r = np.arange(len(C)) * scale

        # curve fit
        mask_fit = r < 3
        popt, pcov = curve_fit(f, r[mask_fit], C[mask_fit], p0)
        # montecarlo uncertainty propagation
        B_list = np.random.normal(popt[1], np.sqrt(pcov[1,1]), 10000)
        b_list = B_list / (1 + B_list)
        print(f'delta = ({popt[0]:.3f} +/- {np.sqrt(pcov[0,0]):.3f})')
        print(f'b = ({np.mean(b_list):.3f} +/- {np.std(b_list):.3f})')
        print(f'C(r) poissonien {(2*cc[i]-1)**2:.4f} '
              + f'z = {np.abs((2*cc[i]-1)**2-np.mean(b_list))/np.std(b_list):.2f}')
        print('--------------------------------')
        ax[0].plot(r, C, color=colors_list[i], 
                   label=f'{thresh[i]}/256')
        ax[0].plot(r, np.ones_like(r)*(2*cc[i]-1)**2, '--',
                   color=colors_list[i], alpha=0.5)
        mask = r < 6
        ax[1].plot(r[mask], C[mask], color=colors_list[i],
                   label=f'{popt}')
        ax[1].plot(r[mask_fit], f(r[mask_fit], *popt), '--',
                           color='black', alpha=0.5)

    ax[0].set_xlabel(r'Radius $r$ [km]')
    ax[0].set_ylabel(r'Radial correlation $C(r)$')
    ax[0].grid()
    ax[0].legend()
    ax[1].set_xlabel(r'Radius $r$ [km]')
    ax[1].grid()
    # ax[1].legend()


    if save:
        plt.savefig('output_figures/corr_threshold_analysis.pdf', 
                    bbox_inches='tight')
    
    
    #### SHOW SUBSAMPLED IMAGES ####

    fig, ax = plt.subplots(1, 4, figsize=(13,3))

    for i, t in enumerate(thresh[:4]):

        img = image.image_to_binary_array(cf_name, t)
        size = min(img.shape)
        img = img[:size,:size] # crop to square
        r_max = size // 4
        cf = CloudField(img, r_max=r_max)

        ax[i].imshow(img, cmap='binary_r',
                     extent=[0,scale*size,0,scale*size])
        # ax[i].set_axis_off()
        ax[i].set_title(f'{t}/256    cl. cover = {cf.cloud_cover:.2f}')

        side = (cf.size - 2*cf.r_max)*scale
        square = patches.Rectangle((cf.r_max*scale, cf.r_max*scale), 
                                   side, side, 
                                edgecolor=colors_list[i], facecolor='none')
        ax[i].add_patch(square)

    if save:
        plt.savefig('output_figures/corr_threshold_subsampled_img.pdf', 
                bbox_inches='tight')
        
def MF_resolution(save=False):

    #### RESOLUTION SENSITIVITY ANALYSIS ####

    # Image to analyse
    cf_names = ['Nuages/nuagesOI_L_60m.jpg',
                'Nuages/trous_OI_60m.png',
                'Nuages/petitsnuages_OI_60m.png']
    
    scale = 60e-3 #km/px
    thresh = 150
    size = 5000
    total_area = size**2 * scale**2

    s_min, s_max = scale, 10000*scale
    s_list = np.linspace(s_min, s_max, 200)
    p_list = 1 - scale / s_list
    
    fig, ax = plt.subplots(1, 3, figsize=(15,3))


    for i, cf_name in enumerate(cf_names):

        subscale, m0, m1, m2 = [], [], [], []

        for p in p_list:

            seed = np.random.random(size)
            mask = seed >= p

            img = image.image_to_binary_array(cf_name, thresh)
            img = img[:size,:size] # crop to square
            img = img[mask][:,mask] # subsample
            cf = CloudField(img)

            imM0, imM1, imM2 = cf.minkowski()
            # s = size*scale/np.sum(mask)
            s = scale / (1-p)
            subscale.append(s)
            m0.append(cf.cloud_cover)
            # imM0 does not seem to work properly : even when scaled by cf.n_tot,
            # its value shows a decreasing trend as the image gets subsampled.
            # One wonders whether M1 and M2 work properly...
            m1.append(imM1 * s / total_area)
            m2.append(imM2 / total_area)

        subscale = np.array(subscale)
        m0 = np.array(m0) 
        m1 = np.array(m1)
        m2 = np.array(m2)

        ax[0].plot(subscale, m0, color=colors_list[i], alpha=0.5)
        ax[1].plot(subscale, m1, color=colors_list[i], alpha=0.5)
        ax[2].plot(subscale, m2, color=colors_list[i], alpha=0.5)

        #### FIT ####

        mask1 = subscale > 4
        b1, log_a1, r1, p1, se1 = linregress(np.log(subscale[mask1]), 
                                            np.log(m1[mask1]))
        mask2 = subscale > 4
        b2, log_a2, r2, p2, se2 = linregress(np.log(subscale[mask2]),
                                            np.log(-m2[mask2]))

        a1, a2 = np.exp(log_a1), np.exp(log_a2)

        ax[1].plot(subscale[mask1], a1*subscale[mask1]**b1, '--', 
                color=colors_list[i], label=f'a = {a1:.2f}\nb = {b1:.2f}')
        ax[2].plot(subscale[mask2], -a2*subscale[mask2]**b2, '--', 
                color=colors_list[i], label=f'a = {a2:.2f}\nb = {b2:.2f}')
        ax[0].plot(subscale, np.ones_like(subscale) * m0[0], '--',
            color=colors_list[i], label=r'$m_0$ at best resolution')


    ax[0].set_xlabel(r'Resolution $s$ [km/px]')
    ax[0].set_title(r'Cloud cover $m_0$')
    ax[0].grid()

    ax[1].set_xlabel(r'Resolution $s$ [km/px]')
    ax[1].set_title(r'Interface density $m_1$ [km/km²]')
    ax[1].grid()

    ax[2].set_xlabel(r'Resolution $s$ [km/px]')
    ax[2].set_title(r'Euler characteristic $m_2$ [km$^{-2}$]')
    ax[2].grid()

    if save:
        plt.savefig('output_figures/MF_resolution_analysis.pdf', 
                    bbox_inches='tight')
    

    #### SHOW SUBSAMPLED IMAGES ####

    fig, ax = plt.subplots(3, 4, figsize=(13,9))

    p_list = np.linspace(p_min, p_max, 4)

    for i, cf_name in enumerate(cf_names):

        for j, p in enumerate(p_list):

            seed = np.random.random(size)
            mask = seed >= p

            img = image.image_to_binary_array(cf_name, thresh)
            img = img[:size,:size] # crop to square
            img = img[mask][:,mask] # subsample
            cf = CloudField(img)

            s = scale / (1-p)
            ax[i,j].imshow(img, cmap='binary_r')
            ax[i,j].set_axis_off()
            ax[i,j].set_title(f'{s:.2f} km/px - {img.shape}')

    if save:
        plt.savefig('output_figures/MF_resolution_subsampled_img.pdf', 
                bbox_inches='tight')
        
def MF_resolution_poisson(save=False):

    #### RESOLUTION SENSITIVITY ANALYSIS ####

    # Image to analyse
    cf_name = 'Nuages/Poisson_resolution.png'
    scale = 60e-3 #km/px # value is arbitrary since not a satellite image
    size = min(image.image_to_binary_array(cf_name, 150).shape)
    total_area = size**2 * scale**2

    fig, ax = plt.subplots(1, 3, figsize=(13,3))

    subsampling = np.arange(1, 1000, 10)
    subscale, m0, m1, m2 = [], [], [], []

    for step in subsampling:

        img = image.image_to_binary_array(cf_name, 150)
        img = img[:size:step,:size:step] # crop to square and subsample
        cf = CloudField(img)

        imM0, imM1, imM2 = cf.minkowski()
        subscale.append(step*scale)
        m0.append(cf.cloud_cover)
        # imM0 does not seem to work properly : even when scaled by cf.n_tot,
        # its value shows a decreasing trend as the image gets subsampled.
        # One wonders whether M1 and M2 work properly...
        m1.append(imM1 * step * scale / total_area)
        m2.append(imM2 / total_area)

    subscale = np.array(subscale)
    m0 = np.array(m0) 
    m1 = np.array(m1)
    m2 = np.array(m2)

    ax[0].plot(subscale, m0, color='blue')
    ax[1].loglog(subscale, m1, color='blue')
    ax[2].loglog(subscale, m2, color='blue')

    #### CURVE FIT ####

    mask = subsampling < 100
    b1, log_a1, r1, p1, se1 = linregress(np.log(subscale[mask]), np.log(m1[mask]))    
    b2, log_a2, r2, p2, se2 = linregress(np.log(subscale[mask]), np.log(m2[mask]))  
    a1, a2 = np.exp(log_a1), np.exp(log_a2)

    ax[1].loglog(subscale, a1*subscale**b1, '--', color='red', 
               label=f'a = {a1:.5f}\nb = {b1:.5f}')
    ax[2].loglog(subscale, a2*subscale**b2, '--', color='red', 
                label=f'a = {a2:.5f}\nb = {b2:.5f}')
    
    ax[0].set_xlabel(r'Resolution $s$ [km/px]')
    ax[0].plot(subscale, np.ones_like(subscale) * m0[0], '--',
        color='red', label=r'$m_0$ at best resolution')
    ax[0].set_title(r'Cloud cover $m_0$')
    ax[0].grid()
    ax[0].legend()


    ax[1].set_xlabel(r'Resolution $s$ [km/px]')
    ax[1].set_title(r'Interface density $m_1$ [km/km²]')
    # ax[1].yaxis.set_label_position('right')
    ax[1].grid()
    ax[1].legend()

    ax[2].set_xlabel(r'Resolution $s$ [km/px]')
    # ax[2].yaxis.set_label_position('right')
    ax[2].set_title(r'Euler characteristic $m_2$ [$km^{-2}$]')
    ax[2].grid()
    ax[2].legend()

    if save:
        plt.savefig('output_figures/MF_resolution_poisson_analysis.pdf', 
                    bbox_inches='tight')
    

    #### SHOW SUBSAMPLED IMAGES ####

    subsampling = np.logspace(0, np.log10(100), 4).astype(np.int32)
    fig, ax = plt.subplots(1, 4, figsize=(13,3))

    for i, step in enumerate(subsampling):

        img = image.image_to_binary_array(cf_name, 150)
        img = img[:size:step,:size:step] # crop to square and subsample

        ax[i].imshow(img, cmap='binary_r')
        ax[i].set_axis_off()
        ax[i].set_title(f'{step*scale:.2f} km/px - {img.shape}')

    if save:
        plt.savefig('output_figures/MF_resolution_poisson_subsampled_img.pdf', 
                bbox_inches='tight')

def MF_resolution_disk(save=False):

    #### RESOLUTION SENSITIVITY ANALYSIS ####

    # Image to analyse
    cf_name = 'Nuages/disk.png'
    scale = 60e-3 #km/px # value is arbitrary since not a satellite image
    size = min(image.image_to_binary_array(cf_name, 150).shape)
    total_area = size**2 * scale**2
    r_disk = 3000 #px at original resolution
    fig, ax = plt.subplots(1, 3, figsize=(13,3))

    subsampling = np.arange(1, 300, 3)
    subscale, m0, m1, m2 = [], [], [], []

    for step in subsampling:

        img = image.image_to_binary_array(cf_name, 150)
        img = img[:size:step,:size:step] # crop to square and subsample
        cf = CloudField(img)

        imM0, imM1, imM2 = cf.minkowski()
        subscale.append(step*scale)
        m0.append(cf.cloud_cover)
        # imM0 does not seem to work properly : even when scaled by cf.n_tot,
        # its value shows a decreasing trend as the image gets subsampled.
        # One wonders whether M1 and M2 work properly...
        m1.append(imM1 * step * scale / total_area)
        m2.append(imM2)

    subscale = np.array(subscale)
    m0 = np.array(m0) 
    m1 = np.array(m1)
    m2 = np.array(m2)

    ax[0].plot(subscale, m0, color='blue')

    m0_th = np.pi*r_disk**2/size**2
    ax[0].plot(subscale, np.ones_like(subscale) * m0_th, '--',
            color='red', label=r'$m_0$ for continuous disk')
    ax[1].plot(subscale, m1, color='blue')
    m1_th = 2*np.pi*r_disk*scale / total_area
    ax[1].plot(subscale, np.ones_like(subscale) * m1_th, '--',
               color='red', label=r'$m_1$ for continuous disk')
    ax[2].plot(subscale, m2, color='blue')

    ax[0].set_xlabel(r'Resolution $s$ [km/px]')
    ax[0].set_title(r'Cloud cover $m_0$')
    ax[0].grid()
    ax[0].legend()


    ax[1].set_xlabel(r'Resolution $s$ [km/px]')
    ax[1].set_title(r'Interface density $m_1$ [km/km²]')
    # ax[1].yaxis.set_label_position('right')
    ax[1].grid()
    ax[1].legend()

    ax[2].set_xlabel(r'Resolution $s$ [km/px]')
    # ax[2].yaxis.set_label_position('right')
    ax[2].set_title(r'Absolute Euler characteristic $M_2$')
    ax[2].grid()

    if save:
        plt.savefig('output_figures/MF_resolution_disk_analysis.pdf', 
                    bbox_inches='tight')
    

    #### SHOW SUBSAMPLED IMAGES ####

    subsampling = np.linspace(1, np.max(subsampling), 4).astype(np.int32)
    fig, ax = plt.subplots(1, 4, figsize=(13,3))

    for i, step in enumerate(subsampling):

        img = image.image_to_binary_array(cf_name, 150)
        img = img[:size:step,:size:step] # crop to square and subsample

        ax[i].imshow(img, cmap='binary_r')
        ax[i].set_axis_off()
        ax[i].set_title(f'{step*scale:.2f} km/px - {img.shape}')

    if save:
        plt.savefig('output_figures/MF_resolution_disk_subsampled_img.pdf', 
                bbox_inches='tight')
        
def MF_threshold(save=False):

    #### THRESHOLD SENSITIVITY ANALYSIS ####

    # Image to analyse
    cf_name = 'Nuages/nuagesOI_L_60m.jpg'
    scale = 60e-3 #km/px
    size = min(image.image_to_binary_array(cf_name, 150).shape)
    total_area = size**2 * scale**2
    thresh = np.arange(0, 257, step=4)

    fig, axes = plt.subplots(3, 2, figsize=(9, 8), sharex=False)
    ax, ax_norm = axes[:,0], axes[:,1]

    m0, m1, m2 = [], [], []
    for t in thresh:
        cf = CloudField(image.image_to_binary_array(cf_name, t)[:size,:size])
        imM0, imM1, imM2 = cf.minkowski()
        m0.append(cf.cloud_cover)
        m1.append(imM1 * scale / total_area)
        m2.append(imM2 / total_area)
    m0 = np.array(m0)
    m1 = np.array(m1)
    m2 = np.array(m2)

    mask = np.logical_and(m0>0., m0<1.) # atanh(-1) = atanh(1) = inf

    x = (thresh[mask] - 127.5)/127.5
    pm0 = np.arctanh(2*m0[mask]-1)
    pm1 = m1[mask] / (m0[mask] * (1-m0[mask]))
    pm2 = m2[mask] / m1[mask]

    coef0 = np.polyfit(x, pm0, 6)
    coef1 = np.polyfit(x, pm1, 6)
    coef2 = np.polyfit(x, pm2, 6)

    poly0 = np.poly1d(coef0)
    poly1 = np.poly1d(coef1)
    poly2 = np.poly1d(coef2)

    ax[0].plot(thresh, m0, color='blue')
    ax[0].set_ylabel(r'Cloud cover $m_0$')
    ax[0].grid()

    ax[1].plot(thresh, m1, color='blue')
    ax[1].set_ylabel(r'Interface density $m_1$ [km/km²]')
    ax[1].grid()

    ax[2].plot(thresh, m2, color='blue')
    ax[2].set_ylabel(r'Euler characteristic $m_2 [km^{-2}]$')
    ax[2].grid()
    ax[2].set_xlabel(r"Binarization threshold $\rho$")

    ax_norm[0].plot(x, pm0, color='blue')
    ax_norm[0].plot(x, poly0(x), '--', color='red',
                    label=np.array2string(coef0, precision=2))
    ax_norm[0].set_ylabel(r'$p_{m_0} = \mathrm{tanh}^{-1}(2 m_0 - 1)$')
    ax_norm[0].grid()
    ax_norm[0].legend()

    ax_norm[1].plot(x, pm1, color='blue')
    ax_norm[1].plot(x, poly1(x), '--', color='red',
                    label=np.array2string(coef1, precision=2))
    ax_norm[1].set_ylabel(r'$p_{m_1} = \frac{m_1}{m_0 (1-m_0)}$')
    ax_norm[1].legend()
    ax_norm[1].grid()

    ax_norm[2].plot(x, pm2, color='blue')
    ax_norm[2].plot(x, poly2(x), '--', color='red',
                    label=np.array2string(coef2, precision=2))
    ax_norm[2].set_ylabel(r'$p_{m_2} = \frac{m_2}{m_1}$')
    ax_norm[2].grid()
    ax_norm[2].legend()
    ax_norm[2].set_xlabel(r"Normalized threshold")

    if save:
        plt.savefig('output_figures/MF_threshold.pdf', 
                bbox_inches='tight')


    #### VISUALIZATION ####

    thresh = [70, 110, 150, 190]
    fig, ax = plt.subplots(1, 4, figsize=(13,3))

    for i, t in enumerate(thresh):

        img = image.image_to_binary_array(cf_name, t)
        size = min(img.shape)
        img = img[:size,:size] # crop to square and subsample

        ax[i].imshow(img, cmap='binary_r')
        ax[i].set_axis_off()
        ax[i].set_title(f'{t}/255         {(t-127.5)/127.5:.2f}')

    if save:
        plt.savefig('output_figures/MF_threshold_img.pdf', 
                bbox_inches='tight')

def fractalite():

    # Image to analyse
    cf_names = ['Nuages/nuagesOI_L_60m.jpg',
                'Nuages/trous_OI_60m.png',
                'Nuages/petitsnuages_OI_60m.png']
    
    scale = 60e-3 #km/px
    thresh = 150
    size = 5000
    total_area = size**2 * scale**2

    subsampling = np.logspace(1, np.log10(500), 100, dtype=int)

    fig, ax = plt.subplots()


    for i, cf_name in enumerate(cf_names):

        subscale, m1 = [], []

        for step in subsampling:

            img = image.image_to_binary_array(cf_name, thresh)
            img = img[:size:step,:size:step] # crop to square and subsample
            cf = CloudField(img)

            imM0, imM1, imM2 = cf.minkowski()
            subscale.append(step*scale)
            m1.append(imM1 * step * scale / total_area)

        subscale = np.array(subscale)
        m1 = np.array(m1)

        def f(x, a, b):
            return a*x+b

        mask = subscale < 10
        p0 = [1, 1]
        popt, pcov = curve_fit(f, np.log(subscale[mask]), np.log(m1[mask]), p0) 
        a, b = popt

        ax.loglog(subscale, m1, color='blue')
        ax.loglog(subscale[mask], subscale[mask]**a * np.exp(b), '--', color='red')

    ax.set_xlabel(r'Scale $s$ [km/px]')
    ax.set_ylabel(r'Interface density $m_1$ [km/km²]')
    ax.grid()




            