import numpy as np

from . import const, coords, math_utils as mu

class PixMap:
    def __init__(self, data:np.ndarray,
                 pos:np.ndarray,
                 mask:np.ndarray = None,
                 pole_lat = 90,
                 pole_lon = 0):
        self.raw_data = np.copy(data)
        self.raw_pos  = np.copy(pos)
        self.mask = None if mask is None else np.array(mask, dtype=bool)
        self.pole_lat = pole_lat
        self.pole_lon = pole_lon
    
    # ------ property methods ------
    @property
    def data(self):
        if not self.mask is None:
            vis_filter = self.get_pixels_visibility_filter()
            return self.raw_data[vis_filter]
        return self.raw_data
    
    @data.setter
    def data(self, value):
        self.raw_data = value

    @property
    def pos(self):
        if not self.mask is None:
            vis_filter = self.get_pixels_visibility_filter()
            return self.raw_pos[vis_filter]
        return self.raw_pos

    @pos.setter
    def pos(self, value):
        self.raw_pos = value

    # ------ pixel extraction methods ------
    def copy(self):
        return PixMap(np.copy(self.raw_data),
                      np.copy(self.raw_pos),
                      np.copy(self.mask),
                      self.pole_lat,
                      self.pole_lon)
    
    def extract_selection(self, selection) -> "PixMap":
        selection_mask = None if self.mask is None else self.mask[selection]
        return PixMap(self.raw_data[selection],
                      self.raw_pos[selection],
                      selection_mask,
                      self.pole_lat,
                      self.pole_lon)

    # ------ pixel visibility methods ------
    def get_pixels_visibility_filter(self):
        vis_filter = (self.mask == False)
        return vis_filter

    def get_visible_pixels_ratio(self):
        if self.mask is None:
            return 1.0
        vis_filter = self.get_pixels_visibility_filter()
        return np.sum(vis_filter) / len(vis_filter)

    # ------ pole methods ------
    def set_pole(self, pole_lat, pole_lon):
        self.pos = coords.rotate_pole_to_north(self.raw_pos,
                                               pole_lat,
                                               pole_lon)
        self.pole_lat, self.pole_lon = pole_lat, pole_lon

    def reset_pole(self):
        # Check if it is rotated or not
        if 90 - self.pole_lat >= const.ANG_THRESHOLD:
            # Original north is gone to the same (Theta) and to (180 + Phi) of previous pole
            self.set_pole(self.pole_lat, 180 + self.pole_lon)
        self.pole_lat, self.pole_lon = 90, 0

    def change_pole(self, pole_lat, pole_lon):
        self.reset_pole()
        self.set_pole(pole_lat, pole_lon)

    # ------ modulation methods ------
    def add_modulation(self, pix_mod_arr):
        self.raw_data *= pix_mod_arr
    
    def add_legendre_modulation(self, a_l):
        _modulation = mu.create_legendre_modulation_factor(self.raw_pos, a_l)
        self.add_modulation(_modulation)