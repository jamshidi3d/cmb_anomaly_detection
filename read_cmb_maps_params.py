#
#
# This script provides a faster way to read maps and inputs!
#
#

import json
import cmb_anomaly_utils as cau

cmb_fpath               = "./input/cmb_fits_files/COM_CMB_IQU-commander_2048_R3.00_full.fits"
mask_fpath              = "./input/cmb_fits_files/COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits"
input_params_fpath      = './input/run_parameters.json'

def get_inputs() -> dict:
    json_inputs_file =  open(input_params_fpath,'r')
    _inputs = json.loads(json_inputs_file.read())
    json_inputs_file.close()
    return _inputs

def get_cmb_pixdata(**kwargs):
    sky_pix = cau.file_reader.get_pix_map_from_cmb(cmb_fpath, mask_fpath, **kwargs)
    return sky_pix

def get_mask(**kwargs):
    if not kwargs.get('is_masked', False):
        return None
    mask = cau.file_reader.read_fits_mask(mask_fpath, kwargs.get('nside', 64))
    return mask

class bcolors:
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
    line_col  = bcolors.OKCYAN
    txt_col   = bcolors.WARNING
    # max key length
    mkl = 20
    # handy functions
    def colorize(txt, col):
        return col + txt + bcolors.ENDC
    def print_line(length, color = bcolors.OKGREEN):
        print(colorize("*" * length, color))
    # fancy header
    print_line(2 * mkl, line_col)
    txt_header = "Parameters"
    half1   = txt_header[:int(len(txt_header)/2)]
    half2   = txt_header[int(len(txt_header)/2):]
    _half1  = colorize("*" * (mkl - len(half1)), line_col)
    _half2  = colorize("*" * (mkl - len(half2)), line_col)
    print(_half1 + colorize(txt_header, txt_col) + _half2)
    # parameters
    for key, val in zip(input_dict.keys(), input_dict.values()):
        if "comment" in key.lower():
            continue
        txt_before_delim = "-" + " " * (mkl - len(key)) + colorize(str(key), txt_col)
        print(txt_before_delim + " : " + colorize(str(val), txt_col))
    # fancy line
    for i in range(2) : print_line(2 * mkl, line_col)