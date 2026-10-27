import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors, cm, ticker
from cloudfield import CloudField
import img as image
from PIL import Image
from scipy.optimize import curve_fit
from scipy.stats import linregress



colors_list = ['blue', 'orange', 'red', 'green', 'darkmagenta', 'magenta', 'darkred']
plt.rcParams['font.size'] = 16

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

def display_cloudfields(save=False):

    #### DISPLAY CLOUD FIELDS ####

    labels = ['A', 'B', 'C', 'D', 'E', 'F']

    cf_names = ['Nuages/20260527_typical.jpeg',
                'Nuages/20260527_convectionfermee.jpeg',
                'Nuages/20260527_petitsnuages.jpeg',
                'Nuages/20260527_moutonneux.jpeg',
                'Nuages/20260527_typeL.jpeg',
                'Nuages/20260527_anisotropic.jpeg']

    fig, ax = plt.subplots(2, 3, figsize=(9,6),
                           gridspec_kw={'wspace':0, 'hspace':0},
                           sharey=True, sharex=True)
    ax = ax.flatten()
    size=5000

    for i, cf_name in enumerate(cf_names):
        
        ax[i].imshow(Image.open(cf_name).resize((size,size)), cmap='gray')
        ax[i].set_axis_off()
        ax[i].text(100,4900, labels[i],
            backgroundcolor='white', fontsize=20)
        # ax[i].set_title(f'{cf_name.split("/")[-1]} - cl. cover = {cf.cloud_cover:.2f}')

    if save:
        plt.savefig('output_figures/display_cloudfields.pdf', 
                bbox_inches='tight')

def corr_length(cf_name='Nuages/20260527_typical.jpeg', plot=False):

    #### CORRELATION LENGTH ANALYSIS ####

    # Image to analyse
    scale = 30e-3 #km/px
    thresh = 150
    size = 5000
    total_area = size**2 * scale**2

    img = image.image_to_binary_array(cf_name, thresh)
    r_max = size // 4
    cf = CloudField(img, r_max=r_max)

    C_xy, C_r, C_theta = cf.C(n_draws=1000)
    r = np.arange(len(C_r)) * scale

    def f(r, delta):
        cste=(2*cf.cloud_cover-1)**2
        B = cste / (1-cste)
        return (np.exp(-r/delta) + B) / (1 + B)

    max_fit = 1
    mask = r < max_fit
    p0 = [1.5]
    popt, pcov = curve_fit(f, r[mask], C_r[mask], p0)
    delta = popt

    if plot:

        fig, ax = plt.subplots(1, 2, figsize=(8,3))

        ax[0].plot(r, C_r, color='black', label=r'C(r)')
        ax[0].plot(r, f(r, *popt), '--', color='red')
        ax[0].set_xlabel(r'Radius $r$ [km]')
        ax[0].set_ylabel(r'Radial correlation $C(r)$')
        ax[0].grid()

        ax[1].plot(r[mask], C_r[mask], color='black', label=r'C(r)')
        ax[1].plot(r[mask], f(r[mask], *popt), '--', color='red')
        ax[1].set_xlabel(r'Radius $r$ [km]')
        ax[1].grid()

    return delta


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

def test_mantz2008():
    cf_names = ['Nuages/20260527_typical.jpeg',
                'Nuages/20260527_convectionfermee.jpeg',
                'Nuages/20260527_petitsnuages.jpeg',
                'Nuages/20260527_moutonneux.jpeg',
                'Nuages/20260527_typeL.jpeg',
                'Nuages/20260527_anisotropic.jpeg']

    scale = 30e-3 #km/px
    thresh = 150
    size = 5000
    total_area = size**2 * scale**2

    resol = np.logspace(np.log10(0.03), np.log10(0.2), 10)

    fig, ax = plt.subplots(3, 1, figsize=(5,12),
                            sharex=True,
                            gridspec_kw={'hspace': 0})

    for i, cf_name in enumerate(cf_names):
    
            cc, m1, m2, mantz0, mantz1, mantz2 = [], [], [], [], [], []
    
            for s in resol:
    
                f = s / scale
                gray_img = image.image_to_float_array(cf_name, 
                                                  factor=f, size=size)
                cf = CloudField(gray_img=gray_img, thresh=thresh)
    
                imM0, imM1, imM2 = cf.minkowski()
                cc.append(cf.cloud_cover)
                m1.append(imM1 * s / total_area)
                m2.append(imM2 / total_area)

                mz0, mz1, mz2 = cf.mantz2008()
                mantz0.append(mz0)
                mantz1.append(mz1)
                mantz2.append(mz2)
    
            cc = np.array(cc)
            m1 = np.array(m1)
            m2 = np.array(m2)
            mantz0 = np.array(mantz0)
            mantz1 = np.array(mantz1)
            mantz2 = np.array(mantz2)


            ax[0].semilogx(resol, cc, color=colors_list[i])
            ax[0].semilogx(resol, mantz0, '--', color=colors_list[i])
            ax[1].loglog(resol, m1, color=colors_list[i])
            ax[1].loglog(resol, mantz1, '--', color=colors_list[i])
            ax[2].semilogx(resol, np.abs(m2), color=colors_list[i])
            ax[2].semilogx(resol, np.abs(mantz2), '--', color=coolors_list[i])

    # ax[1].set_xlabel(r'Resolution $s$ [km/px]')
    ax[1].set_ylabel(r'$m_1$ [km/km²]')
    ax[1].grid()

    ax[2].set_xlabel(r'Resolution $s$ [km/px]')
    ax[2].set_ylabel(r'$m_2$ [km$^{-2}$]')
    ax[2].grid()

    ax[0].set_ylabel(r'$m_0$ [km²/km²]')
    ax[0].grid(which='both')           

# FIGURES RESOLUTION SENSITIVITY ANALYSIS

# fonction générique
def _MF_resolution(s_min, s_max, fit_min=None, fit_max=None, 
                            save=False, fit=None, N_points=20,
                            root_name=''):

    #### RESOLUTION SENSITIVITY ANALYSIS ####

    labels = ['A', 'B', 'C', 'D', 'E', 'F']

    cf_names = ['Nuages/20260527_typical.jpeg',
                'Nuages/20260527_convectionfermee.jpeg',
                'Nuages/20260527_petitsnuages.jpeg',
                'Nuages/20260527_moutonneux.jpeg',
                'Nuages/20260527_typeL.jpeg',
                'Nuages/20260527_anisotropic.jpeg']

    scale = 30e-3 #km/px
    thresh = 150
    size = 5000
    total_area = size**2 * scale**2

    resol = np.logspace(np.log10(s_min), np.log10(s_max), N_points)

    ### SHOW MINKOWSKI FUNCTIONALS ###
    
    fig, ax = plt.subplots(3, 1, figsize=(5,12),
                           sharex=True,
                           gridspec_kw={'hspace': 0})

    if fit is not None:
        fig_err, ax_err = plt.subplots(2, 1, figsize=(5,8), 
                                       sharex=True, sharey=False,
                                       gridspec_kw={'hspace': 0})
        
        if fit=='exponential':
            ax_err[0].set_ylabel(r'$m_1$ residuals [km/km²]')
            ax_err[0].grid()#(which='both')
            ax_err[1].set_ylabel(r'$m_2$ residuals [km$^{-2}$]')
            ax_err[1].grid()#which='both')
            ax_err[1].set_xlabel(r'Resolution $s$ [km/px]')
            ax_err[1].set_xlabel(r'Resolution $s$ [km/px]')

        elif fit=="powerlaw":
            ax_err[0].set_ylabel(r'$m_1$ (log) [km/km²]')
            ax_err[0].grid(which='both')
            # ax_err[0].set_yticks([0.9, 0.95, 1, 1.05, 1.1])
            # ax_err[1].set_yticks([0.9, 0.95, 1, 1.05, 1.1])
            ax_err[1].set_ylabel(r'$|m_2|$ (log) [km$^{-2}$]')
            ax_err[1].grid(which='both')
            ax_err[1].set_xlabel(r'Resolution $s$ (log) [km/px]')

        

    for i, cf_name in enumerate(cf_names):

        cc, padcc, m0, m1, m2 = [], [], [], [], []

        for s in resol:

            f = s / scale
            img = image.image_to_binary_array(cf_name, thresh, 
                                              factor=f, size=size)
            cf = CloudField(img)

            imM0, imM1, imM2 = cf.minkowski()
            cc.append(cf.cloud_cover)
            padcc.append(cf.dpm_cloud_cover)
            m0.append(imM0 / cf.size**2)
            m1.append(imM1 * s / total_area)
            m2.append(imM2 / total_area)

        cc = np.array(cc)
        padcc = np.array(padcc)
        m0 = np.array(m0)
        m1 = np.array(m1)
        m2 = np.array(m2)

        ax[0].loglog(resol, m1, color=colors_list[i])
        ax[1].loglog(resol, np.abs(m2), color=colors_list[i])
        ax[2].semilogx(resol, m0, color=colors_list[i])
        ax[2].semilogx(resol, cc, color='black')
        ax[2].semilogx(resol, padcc, '--', color='black')

        if fit == 'powerlaw':
            #### FIT POISSON ####

            mask = np.logical_and(fit_min<=resol, fit_max>=resol)
            m2_sign = np.sign(m2[mask][0])

            def f(s, alpha, beta):
                return alpha * s**beta

            p0 = [0.1, -1]
            popt1, pcov1 = curve_fit(f, resol[mask], m1[mask], p0=p0)
            popt2, pcov2 = curve_fit(f, resol[mask], m2_sign*m2[mask], p0=p0)

            alpha1, beta1 = popt1
            alpha2, beta2 = popt2

            print(f'{colors_list[i]} : alpha1={alpha1:.3f} | beta1={beta1:.3f} | alpha2={alpha2:.3f} | b2={beta2:.3f}')

            ax[0].plot(resol[mask], alpha1*resol[mask]**beta1, '--', 
                    color=colors_list[i], label=f'a = {alpha1:.2f}\nb = {beta1:.2f}')
            ax[1].plot(resol[mask], m2_sign*alpha2*resol[mask]**beta2, '--', 
                    color=colors_list[i], label=f'a = {alpha2:.3f}\nb = {beta2:.2f}')

            #### SHOW RESIDUE ####

            # M1
            ax_err[0].loglog(resol, m1, color=colors_list[i])
            ax_err[0].loglog(resol[mask], alpha1*resol[mask]**beta1, '--', 
                             color='black')
    
            # M2
            ax_err[1].loglog(resol, m2_sign*m2, color=colors_list[i])
            ax_err[1].loglog(resol[mask], alpha2*resol[mask]**beta2, '--', 
                             color='black')

            if save:
                fig_err.savefig('output_figures/MF_resolution_analysis_err_'+root_name+'.pdf', 
                    bbox_inches='tight')

        elif fit=='exponential':

            mask = np.logical_and(fit_min<=resol, fit_max>=resol)

            def f(s, a, b, c):
                return a * np.exp(-s/b) + c

            p1 = [m1[mask][0] - m1[mask][-1],
                  resol[mask][-1] - resol[mask][0],
                  m1[mask][-1]]
            popt1, pcov1 = curve_fit(f, resol[mask], m1[mask], p0=p1,
                                     bounds=([-10, 0, -10], [10, 100, 10]))
            a1, b1, c1 = popt1
            ax[0].loglog(resol[mask], f(resol[mask], *popt1), '--', 
                         color='black')
            
            p2 = [m2[mask][0] - m2[mask][-1],
                  resol[mask][-1] - resol[mask][0],
                  m2[mask][-1]]
            popt2, pcov2 = curve_fit(f, resol[mask], m2[mask], p0=p2,
                                     bounds=([-10, 0, -10], [10, 100, 10]))
            a2, b2, c2 = popt2
            ax[1].loglog(resol[mask], np.abs(f(resol[mask], *popt2)), '--', 
                         color='black')
            

            print(f'{colors_list[i]} : cc={m0[0]:.2f} | '
                  + f'a1={a1:.3f} | b1={b1:.3f} | c1={c1:.3f} '
                  + f'| a2={a2:.3f} | b2={b2:.3f} | c2={c2:.4f}')

            #### SHOW RESIDUE ####

            # M1
            err1 = m1 - f(resol, *popt1)
            ax_err[0].semilogx(resol[mask], err1[mask], color=colors_list[i])

            # M2
            err2 = m2 - f(resol, *popt2)
            ax_err[1].semilogx(resol[mask], err2[mask], color=colors_list[i])

            if save:
                fig_err.savefig('output_figures/MF_resolution_analysis_err_'+root_name+'.pdf', 
                            bbox_inches='tight')

    ax[0].set_xlabel(r'Resolution $s$ [km/px]')
    ax[0].set_ylabel(r'$m_1$ [km/km²]')
    ax[0].grid()

    ax[1].set_xlabel(r'Resolution $s$ [km/px]')
    ax[1].set_ylabel(r'$m_2$ [km$^{-2}$]')
    ax[1].grid()

    ax[2].set_ylabel(r'$m_0$ [km²/km²]')
    ax[2].grid(which='both')

    if save:
        fig.savefig('output_figures/MF_resolution_analysis_'+root_name+'.pdf', 
                    bbox_inches='tight')

    

    # #### SHOW SUBSAMPLED IMAGES ####

    # fig, ax = plt.subplots(len(cf_names), 4, figsize=(13,3*len(cf_names)))

    # s_list = np.linspace(s_min, s_max, 4)

    # for i, cf_name in enumerate(cf_names):

    #     for j, s in enumerate(s_list):

    #         seed = np.random.random(size)
    #         p = 1 - scale / s
    #         mask = seed >= p

    #         img = image.image_to_binary_array(cf_name, thresh)
    #         img = img[:size,:size] # crop to square
    #         img = img[mask][:,mask] # subsample
    #         cf = CloudField(img)

    #         ax[i,j].imshow(img, cmap='binary_r')
    #         ax[i,j].set_axis_off()
    #         ax[i,j].set_title(f'{s:.2f} km/px - {img.shape}', color=colors_list[i])

    # if save:
    #     plt.savefig('output_figures/MF_resolution_subsampled_img_'+root_name+'.pdf', 
    #             bbox_inches='tight')

# fonction qui produit la figure résolution grossière
def MF_coarse_resolution(save=False):
    _MF_resolution(s_min=.03, s_max=7,
                  fit_min=4, fit_max=7, fit='powerlaw',
                  save=save, root_name='coarse',
                  N_points=400)

# fonction qui produit la figure résolution fine
def MF_fine_resolution(save=False):
    _MF_resolution(s_min=.03, s_max=30,
                  fit_min=.03, fit_max=.18, fit='exponential',
                  save=save, root_name='fine',
                  N_points=100)

def MF_subsampled_resolution(cf_name='Nuages/20260527_petitsnuages.jpeg', save=False):
    fig, ax = plt.subplots(3, 3, figsize=(9,9),
                           gridspec_kw={'wspace':0, 'hspace':0})
    ax = ax.flatten()
    size=5000
    scale=0.03 #km/px
    thresh=150

    resol = np.logspace(np.log10(0.03), np.log10(30), 9)

    for i, s in enumerate(resol):
    
        # seed = np.random.random(size)
        # p = 1 - scale / s
        # mask = seed >= p
        
        img = image.image_to_binary_array(cf_name, thresh, factor=s/scale, size=size)
    

        cf = CloudField(img)

        ax[i].set_axis_off()
        ax[i].imshow(img, cmap='binary_r')

        ax[i].text(0, 0,
                   f'{s:.2f} km/px',
                   backgroundcolor='lightblue', fontsize=16)

    if save:
        plt.savefig('output_figures/MF_subsampled_resolution.pdf', 
                bbox_inches='tight')


def MF_resolution_poisson(save=False):

    #### RESOLUTION SENSITIVITY ANALYSIS ####

    # Image to analyse
    cf_name = 'Nuages/Poisson_resolution.png'
    scale = 30e-3 #km/px # value is arbitrary since not a satellite image
    size = 5000
    total_area = size**2 * scale**2

    fig, ax = plt.subplots(1, 3, figsize=(13,3))

    resol = np.linspace(0.03, 0.1, 100)
    m0, m1, m2 = [], [], []

    for s in resol:

        img = image.image_to_binary_array(cf_name, 
                                          threshold=150, 
                                          size=size, 
                                          factor=s/scale)
        
        cf = CloudField(img)

        imM0, imM1, imM2 = cf.minkowski()
        m0.append(cf.cloud_cover)
        # imM0 does not seem to work properly : even when scaled by cf.n_tot,
        # its value shows a decreasing trend as the image gets subsampled.
        # One wonders whether M1 and M2 work properly...
        m1.append(imM1 * s / total_area)
        m2.append(imM2 / total_area)

    m0 = np.array(m0) 
    m1 = np.array(m1)
    m2 = np.array(m2)

    ax[0].plot(resol, m0, color='blue')
    ax[1].loglog(resol, m1, color='blue')
    ax[2].loglog(resol, m2, color='blue')

    #### CURVE FIT ####

    mask = resol > 0
    b1, log_a1, r1, p1, se1 = linregress(np.log(resol[mask]), np.log(m1[mask]))    
    b2, log_a2, r2, p2, se2 = linregress(np.log(resol[mask]), np.log(m2[mask]))  
    a1, a2 = np.exp(log_a1), np.exp(log_a2)

    ax[1].loglog(resol, a1*resol**b1, '--', color='red', 
               label=f'a = {a1:.5f}\nb = {b1:.5f}')
    ax[2].loglog(resol, a2*resol**b2, '--', color='red', 
                label=f'a = {a2:.5f}\nb = {b2:.5f}')
    
    ax[0].set_xlabel(r'Resolution $s$ [km/px]')
    ax[0].plot(resol, np.ones_like(resol) * m0[0], '--',
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
    

    # #### SHOW SUBSAMPLED IMAGES ####

    # subsampling = np.logspace(0, np.log10(100), 4).astype(np.int32)
    # fig, ax = plt.subplots(1, 4, figsize=(13,3))

    # for i, step in enumerate(subsampling):

    #     img = image.image_to_binary_array(cf_name, 150)
    #     img = img[:size:step,:size:step] # crop to square and subsample

    #     ax[i].imshow(img, cmap='binary_r')
    #     ax[i].set_axis_off()
    #     ax[i].set_title(f'{step*scale:.2f} km/px - {img.shape}')

    # if save:
    #     plt.savefig('output_figures/MF_resolution_poisson_subsampled_img.pdf', 
    #             bbox_inches='tight')

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




            