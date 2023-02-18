import matplotlib.pyplot as plt
import numpy as np

from .dtypes import run_parameters
from . import const

def get_output_path(params:run_parameters):
    output_fpath = "./output/"
    output_fpath += "{}".format(params.nside)
    output_fpath += "_{}".format("masked" if params.is_masked else "inpainted")
    output_fpath += "_{}".format("cap" if params.geom_flag == const.CAP_FLAG else "{}stripe".format(params.stripe_thickness))
    output_fpath += "_{}".format("dcorr2" if params.measure_flag == const.D_CORR2_FLAG else \
                                "dstd2" if params.measure_flag == const.D_STD2_FLAG else \
                                "std" if params.measure_flag == const.STD_FLAG else "corr")
    output_fpath += "_{}dtheta".format(params.dtheta)
    return output_fpath


def save_data_to_txt(params:run_parameters, measure_result):
    output_fpath = get_output_path(params)
    fname = output_fpath + "_sampling_range" + ".txt"
    with open(fname, "w") as file:
        np.savetxt(file, params.sampling_range)
    fname = output_fpath + "_result" + ".txt"
    with open(fname, "w") as file:
        np.savetxt(file, measure_result)


def save_fig_to_pdf(params:run_parameters, fig:plt.Figure):
    output_fpath = get_output_path(params)
    fig.savefig(output_fpath + ".pdf", transparent=True)


def get_plot_fig(params:run_parameters, measure_result):
    print("- Plotting")
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(8,5)
    # xlabel
    xlabel = get_xlabel_tex(params)
    ax.set_xlabel(xlabel, fontsize=11)
    # ylabel
    ylabel = get_ylabel_tex(params)
    ax.set_ylabel(ylabel, fontsize=12)
    title = get_plot_title(params)
    # title
    ax.set_title(title, fontsize = 12)
    # plot
    ax.plot(params.sampling_range, measure_result, '-k')
    return fig

########### TeX style string generators ############
def get_measure_tex(params:run_parameters):
    obs = params.observable_flag
    double_obs = '{'+ obs + obs +'}'
    # TeX style titles
    if params.measure_flag == const.D_CORR2_FLAG:
        captitle = r'$\int [C_{double_obs}^{{top}}(\gamma) - C_{double_obs}^{{bottom}}(\gamma)]^2 d\gamma$'.format(double_obs = double_obs)
        strtitle = r'$\int [C_{double_obs}^{{stripe}}(\gamma) - C_{double_obs}^{{rest\,of\,sky}}(\gamma)]^2 d\gamma$'.format(double_obs = double_obs)
    elif params.measure_flag == const.CORR_FLAG:
        captitle = r'$\frac{\int [C_{double_obs}^{{top}}(\gamma)]^2 d\gamma}{\int [C_{double_obs}^{{total}}(\gamma)]^2 d\gamma} - 1$'.format(double_obs = double_obs)
        strtitle = r'$\frac{\int [C_{double_obs}^{{stripe}}(\gamma)]^2 d\gamma}{\int [C_{double_obs}^{{total}}(\gamma)]^2 d\gamma} - 1$'.format(double_obs = double_obs)
    elif params.measure_flag == const.D_STD2_FLAG:
        captitle = r'$[\sigma_{{top}}({obs}) - \sigma_{{bottom}}({obs})]^2$'.format(obs = obs)
        strtitle = r'$[\sigma_{{stripe}}({obs}) - \sigma_{{rest\,of\,sky}}({obs})]^2$'.format(obs = obs)
    elif params.measure_flag == const.STD_FLAG:
        captitle = r'$\sigma_{{top}}({obs})$'.format(obs = obs)
        strtitle = r'$\sigma_{{stripe}}({obs})$'.format(obs = obs)
    # returns
    if params.geom_flag == const.CAP_FLAG:
        return captitle
    elif params.geom_flag == const.STRIPE_FLAG:
        return strtitle

def get_xlabel_tex(params:run_parameters):
    capxlabel = r'Cap angle [$\degree$]'
    strxlabel = r'Stripe Center [$\degree$]'
    xlabel = capxlabel if params.geom_flag == const.CAP_FLAG else strxlabel
    return xlabel

def get_ylabel_tex(params:run_parameters):
    ylabel = get_measure_tex(params)
    if params.measure_flag != const.T:
        return ylabel
    ylabel += r'$\>\>\>\>$'
    if params.measure_flag == const.STD_FLAG:
        ylabel += r'$[\mu K]$'
    elif params.measure_flag == const.D_CORR2_FLAG:
        ylabel += r'$[\mu K]^4$'
    elif params.measure_flag == const.CORR_FLAG:
        pass
    return ylabel

def get_plot_title(params:run_parameters):
    title = get_measure_tex(params)
    title += r'$\,\,\,\, Vs \,\,\,\,$ {}'.format("Top Cap Size" if params.geom_flag == const.CAP_FLAG else "Stripe Center")
    title += r', '
    title += r'{} Map'.format("Masked" if params.is_masked else "Inpainted")
    return title
