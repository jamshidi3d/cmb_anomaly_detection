# ------- HealPix Flags -------
NESTED              = 'NESTED'
RING                = 'RING'

# ------- Grometry flags -------
CAP_FLAG            = 'CAP'
STRIPE_FLAG         = 'STRIPE'
FULL_SKY_FLAG       = 'FULL_SKY'
# ------- Measure flags -------
NORM_CORR_FLAG      = 'NORM_CORR'
D_CORR2_FLAG        = 'DCORR2'
STD_FLAG            = 'STD'
NORM_STD_FLAG       = 'NORM_STD'
D_STD2_FLAG         = 'DSTD2'
NORM_D_STD2_FLAG    = 'NORM_D_STD2'
MEAN_FLAG           = 'MEAN'

# ------- Observables -------
OBS_T               = 'T'
OBS_U               = 'U'
OBS_Q               = 'Q'
OBS_P               = 'P'
OBS_E_MODE          = 'EMODE'
OBS_B_MODE          = 'BMODE'
'''polarization strength which is equal to sqrt(U^2+Q^2)'''

# ------- 2 Point Correlation Funciton -------
TPCF_TT             = 'TT'

# ------- Threshold parameters -------
ANG_THRESHOLD       = 0.0001
DIST_THRESHOLD      = 0.0000001

# ------- Input parameters (keys) -------
KEY_OBSERVABLE          = 'observable' 
KEY_SIMS_PATH           = 'sims_path'
KEY_CMB_FPATH           = 'cmb_fpath'
KEY_MASK_FPATH          = 'mask_fpath'
KEY_NOISE_PATH          = 'noise_fpath'
KEY_SIMS_ANOM_PATH      = 'sims_dir_anom_path'
KEY_CMB_ANOM_FPATH      = 'cmb_dir_anom_fpath'
KEY_NSIDE               = 'nside'
KEY_DIRNSIDE            = 'dir_nside'
KEY_IS_MASKED           = 'is_masked'
KEY_TPCF_MODE           = 'tpcf_mode'
KEY_POLE_LON            = 'pole_lon'
KEY_POLE_LAT            = 'pole_lat'
KEY_MIN_PIX_RATIO       = 'min_pix_ratio'
KEY_MEASURE_FLAG        = 'measure_flag'
KEY_GEOM_FLAG           = 'geom_flag'
KEY_MEASURE_START       = 'measure_start'
KEY_MEASURE_STOP        = 'measure_stop'
KEY_DMEASURE_SAMPLES    = 'delta_measure_samples'
KEY_MEASURE_RANGE       = 'measure_range'
KEY_CUTOFF_RATIO        = 'cutoff_ratio'
KEY_NDATA_CHUNKS        = 'ndata_chunks'
KEY_GEOM_START          = 'geom_start'
KEY_GEOM_STOP           = 'geom_stop'
KEY_DGEOM_SAMPLES       = 'dgeom_samples'
KEY_GEOM_RANGE          = 'geom_range'
KEY_STRIPE_THICKNESS    = 'stripe_thickness'
KEY_MAX_VALID_ANG       = 'max_valid_ang'
KEY_CORR_FULL_INT       = 'corr_full_integral'
KEY_STD_FULL            = 'std_full'
