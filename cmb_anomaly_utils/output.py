import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from . import const

def get_output_path(**kwargs):
    '''inputs:\n
    measure_flag - observable_flag - nside - is_masked - geom_flag - dtheta'''
    measure_flag = kwargs['measure_flag']
    output_fpath = "./output/"
    output_fpath += "{}".format(kwargs['observable_flag'])
    output_fpath += "_{}".format(kwargs['nside'])
    output_fpath += "_{}".format("masked" if kwargs['is_masked'] else "inpainted")
    output_fpath += "_{}".format("cap" if kwargs['geom_flag'] == const.CAP_FLAG else "{}strip".format(kwargs['strip_thickness']))
    output_fpath += "_{}".format("dcorr2" if measure_flag == const.D_CORR2_FLAG else \
                                "dstd2" if measure_flag == const.D_STD2_FLAG else \
                                "std" if measure_flag == const.STD_FLAG else \
                                "corr" if measure_flag == const.CORR_FLAG else "mean")
    # output_fpath += "_{}dtheta".format(kwargs['dtheta'])
    return output_fpath


def save_data_to_txt(measure_result, **kwargs):
    output_fpath = get_output_path(**kwargs)
    fname = output_fpath + "_measure_range" + ".txt"
    with open(fname, "w") as file:
        np.savetxt(file, kwargs['measure_range'])
    fname = output_fpath + "_result" + ".txt"
    with open(fname, "w") as file:
        np.savetxt(file, measure_result)


def save_fig_to_pdf(fig:plt.Figure, **kwargs):
    matplotlib.use('Agg') # for writing to files only
    output_fpath = get_output_path(**kwargs)
    fig.savefig(output_fpath + ".pdf", transparent=True)


def get_plot_fig(measure_result, **kwargs):
    print("- Plotting")
    matplotlib.use('Agg') # for writing to files only
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(8,5)
    # axis numbers font size
    ax.tick_params(axis='both', which='major', labelsize=14)
    # xlabel
    xlabel = get_xlabel_tex(**kwargs)
    ax.set_xlabel(xlabel, fontsize=16)
    # ylabel
    ylabel = get_ylabel_tex(**kwargs)
    ax.set_ylabel(ylabel, fontsize=18)
    # title
    title = get_plot_title(**kwargs)
    ax.set_title(title, fontsize = 20)
    # plot
    ax.plot(kwargs['measure_range'], measure_result, '-k')
    return fig

#---------- TeX style string generators ----------
def get_measure_tex(**kwargs):
    obs, measure_flag, geom_flag = kwargs['observable_flag'], kwargs['measure_flag'], kwargs['geom_flag']
    double_obs = '{'+ obs + obs +'}'
    # TeX style titles
    if kwargs['measure_flag'] == const.D_CORR2_FLAG:
        captitle = r'$\int [C_{double_obs}^{{top}}(\gamma) - C_{double_obs}^{{bottom}}(\gamma)]^2 d\gamma$'.format(double_obs = double_obs)
        strtitle = r'$\int [C_{double_obs}^{{strip}}(\gamma) - C_{double_obs}^{{rest\,of\,sky}}(\gamma)]^2 d\gamma$'.format(double_obs = double_obs)
    elif measure_flag == const.CORR_FLAG:
        captitle = r'$\frac {{ \int [C_{double_obs}^{{top}}(\gamma)]^2 d\gamma }} {{ \int [C_{double_obs}^{{total}}(\gamma)]^2 d\gamma }}  - 1$'.format(double_obs = double_obs)
        strtitle = r'$\frac {{ \int [C_{double_obs}^{{strip}}(\gamma)]^2 d\gamma }} {{ \int [C_{double_obs}^{{total}}(\gamma)]^2 d\gamma }} - 1$'.format(double_obs = double_obs)
    elif measure_flag == const.D_STD2_FLAG:
        captitle = r'$[\sigma_{{top}}({obs}) - \sigma_{{bottom}}({obs})]^2$'.format(obs = obs)
        strtitle = r'$[\sigma_{{strip}}({obs}) - \sigma_{{rest\,of\,sky}}({obs})]^2$'.format(obs = obs)
    elif measure_flag == const.STD_FLAG:
        captitle = r'$\sigma_{{top}}({obs})$'.format(obs = obs)
        strtitle = r'$\sigma_{{strip}}({obs})$'.format(obs = obs)
    elif measure_flag == const.MEAN_FLAG:
        captitle = r'$<{obs}>_{{top}}$'.format(obs = obs)
        strtitle = r'$<{obs}>_{{strip}}$'.format(obs = obs)
    # returns
    if geom_flag == const.CAP_FLAG:
        return captitle
    elif geom_flag == const.STRIP_FLAG:
        return strtitle

def get_xlabel_tex(**kwargs):
    capxlabel = r'Cap angle [$\degree$]'
    strxlabel = r'Stripe Center [$\degree$]'
    xlabel = capxlabel if kwargs['geom_flag'] == const.CAP_FLAG else strxlabel
    return xlabel

unit_dict = {
    const.U:'\mu K',
    const.Q:'\mu K',
    const.P:'\mu K',
    const.T:'\mu K'
}
def get_ylabel_tex(**kwargs):
    ylabel = get_measure_tex(**kwargs)
    ylabel += r'$\>\>\>\>$'
    if kwargs['measure_flag'] == const.STD_FLAG:
        ylabel += r'$[{unit}]$'.format(unit = unit_dict[kwargs['observable_flag']])
    elif kwargs['measure_flag'] == const.D_CORR2_FLAG:
        ylabel += r'$[{unit}]^4$'.format(unit = unit_dict[kwargs['observable_flag']])
    elif kwargs['measure_flag'] == const.CORR_FLAG:
        pass
    return ylabel

def get_plot_title(**kwargs):
    title = get_measure_tex(**kwargs)
    title += r'$\,\,\,\, Vs \,\,\,\,$ {}'.format("Top Cap Size" if kwargs['geom_flag'] == const.CAP_FLAG else "Stripe Center")
    title += r', '
    title += r'{} Map'.format("Masked" if kwargs['is_masked'] else "Inpainted")
    return title
