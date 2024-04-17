import os

from . import const

# ------- Path methods -------
def does_path_exist(path):
    return os.path.exists(path)

def ensure_dir(path):
    '''Create path if doesn't exist'''
    if not does_path_exist(path):
        os.makedirs(path)
    return path

def ensure_output_dir(base_path = './', **kwargs):
    path = get_output_path(base_path, **kwargs)
    ensure_dir(path)
    return path

def get_output_path(base_path = './', **kwargs):
    mask_txt:str     = 'masked' if kwargs.get(const.KEY_IS_MASKED) else 'inpainted'
    geom_flag:str    = kwargs.get(const.KEY_GEOM_FLAG)
    measure_flag:str = kwargs.get(const.KEY_MEASURE_FLAG)
    bpath = base_path + "" if base_path[-1] == "/" else "/"
    path = bpath + "{}/{}/{}/".format(mask_txt.lower(),
                                      geom_flag.lower(),
                                      measure_flag.lower())
    return path

# ------- TeX strings -------
tex_geom_dict = {
    const.CAP_FLAG:     (r"\mathrm{top}", r"\mathrm{bottom}"),
    const.STRIPE_FLAG:   (r"\mathrm{stripe}", r"\mathrm{rest\;of\;sky}"),
    const.FULL_SKY_FLAG: r"\mathrm{full\;sky}"
}
tex_measure_dict = {
    const.STD_FLAG:     r'$\sigma_{geom1}({observable})$',
    const.D_STD2_FLAG:  r'$[\sigma_{geom1}({observable}) - \sigma_{geom2}({observable})]^2$',
    const.NORM_CORR_FLAG: \
        r'$\frac {{\int [C_{tpcf_mode}^{{geom1}}(\gamma)]^2 d\gamma }}{{\int [C_{tpcf_mode}^{{full_sky}}(\gamma)]^2 d\gamma }} - 1$',
    const.D_CORR2_FLAG: \
        r'$\int [C_{tpcf_mode}^{{geom1}}(\gamma) - C_{tpcf_mode}^{{geom2}}(\gamma)]^2 d\gamma$',
    const.NORM_STD_FLAG: \
        r'$\sigma_{geom1}({observable}) / \sigma_{full_sky}({observable}) - 1$',
    const.NORM_D_STD2_FLAG: \
        r'$\frac{{ 1 }}{{ \sigma_{full_sky}({observable})^2 }} ' +\
            r'[\sigma_{geom1}({observable}) - \sigma_{geom2}({observable})]^2$'
    # const.MEAN_FLAG:
}
def get_measure_tex(**kwargs):
    measure_flag = kwargs.get(const.KEY_MEASURE_FLAG)
    geom_flag    = kwargs.get(const.KEY_GEOM_FLAG)
    measure_txt  = tex_measure_dict[measure_flag].format(
        observable  = kwargs.get(const.KEY_OBSERVABLE),
        tpcf_mode   = kwargs.get(const.KEY_TPCF_MODE),
        geom1       = tex_geom_dict[geom_flag][0],
        geom2       = tex_geom_dict[geom_flag][1],
        full_sky    = tex_geom_dict[const.FULL_SKY_FLAG]
    )
    return measure_txt

title_text_dict = {
    const.CAP_FLAG:     r'Cap Size',
    const.STRIPE_FLAG:   r'Stripe Center'
}
def get_title_tex(**kwargs):
    measure_text = get_measure_tex(**kwargs)
    geom_flag    = kwargs.get(const.KEY_GEOM_FLAG)
    return  r'\boldmath{measure_txt}'.format(measure_txt = measure_text) + \
            r'\textbf{ vs. }' + \
            r'\textbf{{{title_txt}}}'.format(title_txt = title_text_dict[geom_flag]) + \
            r' (Inpainted Map)'

tex_unit_dict = {
    const.STD_FLAG:         r'$[\mu K]$',
    const.D_STD2_FLAG:      r'$[\mu K]^2$',
    const.VAR_FLAG:         r'$[\mu K]^2$',
    const.NORM_CORR_FLAG:   r'',
    const.D_CORR2_FLAG:     r'$[\mu K]^4$',
    const.NORM_STD_FLAG:    r'',
    const.NORM_D_STD2_FLAG: r'',
    const.MEAN_FLAG:        r'[\mu K]'
}
def get_ylabel_tex(**kwargs):
    return  get_measure_tex(**kwargs) +\
            r'\;' +\
            tex_unit_dict[kwargs.get(const.KEY_MEASURE_FLAG)]

tex_xlabel = {
    const.CAP_FLAG:   r'Cap Radius[$^\circ$]',
    const.STRIPE_FLAG: r'Stripe Center[$^\circ$]'
}
def get_xlabel_tex(**kwargs):
    geom_flag    = kwargs.get(const.KEY_GEOM_FLAG)
    return tex_xlabel[geom_flag]

# ------- Console printing ------- 
class BColors:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKCYAN    = '\033[96m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

def print_inputs(input_dict):
    # Handy functions
    def colorize(txt, col):
        return col + txt + BColors.ENDC
    def print_line(length, color = BColors.OKGREEN):
        print(colorize("*" * length, color))
    # Colors used
    line_col  = BColors.OKCYAN
    txt_col   = BColors.WARNING
    # Max key length
    mkl = 20
    # Fancy header
    print_line(2 * mkl, line_col)
    txt_header = "Parameters"
    half1   = txt_header[:int(len(txt_header)/2)]
    half2   = txt_header[int(len(txt_header)/2):]
    _half1  = colorize("*" * (mkl - len(half1)), line_col)
    _half2  = colorize("*" * (mkl - len(half2)), line_col)
    print(_half1 + colorize(txt_header, txt_col) + _half2)
    # Parameters
    for key, val in zip(input_dict.keys(), input_dict.values()):
        klower = key.lower()
        if "comment" in klower or "range" in klower:
            continue
        txt_before_delim = "-" + " " * (mkl - len(key)) + colorize(str(key), txt_col)
        print(txt_before_delim + " : " + colorize(str(val), txt_col))
    # Fancy line
    for i in range(2) : print_line(2 * mkl, line_col)