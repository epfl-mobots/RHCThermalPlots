'''
Author(s):
    Cyril Monette, EPFL, cyril.monette@epfl.ch

This library implements a class to view a thermal hive composed of two thermal frames.
The class is capable of analysing the thermal data to locate where the cluster(s) is (are).
'''

from thermalframe import ThermalFrame
from matplotlib.contour import QuadContourSet
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

WARNING = '\033[93m'
BOLD = '\033[1m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
HEADER = '\033[95m'
OKCYAN = '\033[96m'
FAIL = '\033[91m'
UNDERLINE = '\033[4m'
ENDC = '\033[0m'

def cprint(string:str, color=OKCYAN):
    """
    Just some colored print for debugging
    """
    print(color + string + ENDC)


class ThermalHive:
    '''
    A class to represent a thermal hive composed of two thermal frames (upper and lower).
    The class can analyze the thermal data to locate the cluster(s) position(s).
    '''

    gap_between_frames_mm = 66  # mm, vertical gap between upper and lower frames, including the electronics bay on top of the lower frame
    HIVE_PADDING = 20  # padding around the hive when plotting

    def __init__(self, isotherm:float, upper_frame:ThermalFrame=None, lower_frame:ThermalFrame=None):
        # Make sure both frames belong to the same hive
        if upper_frame is not None and lower_frame is not None:
            if upper_frame.hive_id is not None and lower_frame.hive_id is not None:
                assert upper_frame.hive_id == lower_frame.hive_id, "Upper and lower frames must belong to the same hive (hive_id mismatch)."
        # Make sure we have at least one frame to work with
        assert upper_frame is not None or lower_frame is not None, "At least one of upper_frame or lower_frame must be provided."
        self.frames = {'upper': upper_frame, 'lower': lower_frame}

        # Temperature isotherm that defines a cluster
        self.isotherm = isotherm 

        # The frame that currently has the cluster
        self.frame_with_cluster = None # 'upper', 'lower' or None

        self.origin_x = {'upper': 0, 'lower': 0}
        self.origin_y = {'upper': ThermalHive.gap_between_frames_mm+ThermalFrame.y_pcb, 'lower': 0}

        self.contours = None
        self._contours_frame = {'upper': None, 'lower': None}

        # Number of contours
        self.n_contours_tot = 0 # aggregate
        self.n_contours = {'upper': 0, 'lower': 0}

        # Box encapsulating the contours
        self.contours_box = None # aggregate
        self.contours_box_frame = {'upper': None, 'lower': None}

        # Contours Center of Mass
        self.contours_CoM = None # aggregate
        self.contours_CoM_frame = {'upper': None, 'lower': None}

        # Areas of the contours
        self.areas = {'upper': None, 'lower': None}

        # The position what is considered the cluster
        # If more than one contour is found, one will be selected and stored here.
        self.position = None  # (x, y)
        self.position_in_hive = None # with reference to the whole hive
        self.position_frame = {'upper': None, 'lower': None}

        self.isPositionFix = False


    #---- PUBLIC METHODS ----

    def update(self, why:str='', verbose:bool=False) -> bool:
        ''' Search for cluster(s) position(s) in the whole hive'''
        return self.find_hive_contours(f'ThermalHive-update|{why}', verbose=verbose)

    def add_offset(self, pos:Tuple[float, float], frame:str):
        assert frame in ['upper', 'lower'], "frame must be either 'upper' or 'lower'"
        if self.origin_x[frame] is None or self.origin_y[frame] is None:
            raise ValueError("Origin not set. Use set_origins() to set the origins of the upper and lower frames.")
        px, py = pos
        px+= self.origin_x[frame]
        py+= self.origin_y[frame]
        return float(px), float(py)

    def find_hive_contours(self, why:str=None, verbose:bool=False) -> bool:
        '''
        Finds all candidate cluster location(s) (warm contours)
        return True iff exactly one contour is found in the whole hive
        '''

        for key in self.frames.keys():
            # find the hotspots for cluster candidates, including CoM and bbox
            self._find_contours_oneframe(frame=key)

        # aggregate relevant sub-values
        self.n_contours_tot = self.n_contours['upper'] + self.n_contours['lower']
        self.contours_CoM = self.contours_CoM_frame['upper'] + self.contours_CoM_frame['lower']
        self.contours_box = self.contours_box_frame['upper'] + self.contours_box_frame['lower']

        if self.n_contours_tot == 1:
            self.isPositionFix = True
            self.position =  tuple(map(float, self.contours_CoM[0]))
            if self.n_contours['upper'] == 1:
                self.frame_with_cluster = 'upper'
                self.position_frame['upper'] = self.position
                self.position_frame['lower'] = None
                self.position_in_hive = self.add_offset(self.position, frame='upper')

            elif self.n_contours['lower'] == 1:
                self.frame_with_cluster = 'lower'
                self.position_frame['lower'] = self.position
                self.position_frame['upper'] = None
                self.position_in_hive = self.add_offset(self.position, frame='lower')

            else:
                raise ValueError("[E] someone can't count!")
            
        else:
            self.isPositionFix = False
            self.frame_with_cluster = None
            self.position = None # this is commented out in parent, why??
            self.position_frame['lower'] = None
            self.position_frame['upper'] = None
            self.position_in_hive = None

        if verbose:
            if self.isPositionFix:
                fixstr = f"frame {str(self.frame_with_cluster)}"
            else:
                fixstr = ""

            s = f"==[D][find_hive_cont({why})][Fix?{self.isPositionFix}|{fixstr}. Ncont: {self.n_contours_tot}"
            cprint(s, BOLD)

        return self.isPositionFix
    
    def plot(self, ax:plt.Axes, contours:bool=True, center:bool=True, box:bool=False, Tgrad:bool=True):
        '''
        Plots the thermal hive with the following options:

        :param contours: whether to plot the contours of the cluster candidates
        :param center: whether to plot the center of mass of the cluster candidates
        :param box: whether to plot the bounding box of the cluster candidates
        :param Tgrad: whether to plot the thermal gradient of the frames ontop
        '''
        # Ensure that both Thermal Frames have their thermal field calculated (for plotting)
        for key in self.frames.keys():
            if self.frames[key] is not None and self.frames[key].thermal_field is None:
                self.frames[key].calculate_thermal_field()

        def _draw_black_frame(y_offset:float):
            ax.add_patch(
                plt.Rectangle(
                    (0, y_offset),
                    ThermalFrame.x_pcb,
                    ThermalFrame.y_pcb,
                    color='black',
                    zorder=0,
                )
            )

        lower_frame = self.frames['lower']
        upper_frame = self.frames['upper']
        thermal_frames = [frame for frame in [lower_frame, upper_frame] if frame is not None]
        if Tgrad and len(thermal_frames) > 0:
            v_min = min(frame.min_temp for frame in thermal_frames)
            v_max = max(frame.max_temp for frame in thermal_frames)
        
        lw = 1.5
        # First draw the frame contours
        # lower frame
        ax.plot([0, ThermalFrame.x_pcb, ThermalFrame.x_pcb, 0, 0],
                    [0, 0, ThermalFrame.y_pcb, ThermalFrame.y_pcb, 0],
                    lw=lw/2, c='black', zorder=2)
        # upper frame
        ax.plot([0, ThermalFrame.x_pcb, ThermalFrame.x_pcb, 0, 0],
                [ThermalFrame.y_pcb + ThermalHive.gap_between_frames_mm,
                 ThermalFrame.y_pcb + ThermalHive.gap_between_frames_mm,
                 2*ThermalFrame.y_pcb + ThermalHive.gap_between_frames_mm,
                 2*ThermalFrame.y_pcb + ThermalHive.gap_between_frames_mm,
                 ThermalFrame.y_pcb + ThermalHive.gap_between_frames_mm],
                lw=lw/2, c='black', zorder=2)
        
        lower_y_offset = 0
        upper_y_offset = ThermalHive.gap_between_frames_mm + ThermalFrame.y_pcb
        show_lower_cb = lower_frame is not None and Tgrad
        show_upper_cb = upper_frame is not None and lower_frame is None and Tgrad

        if lower_frame is None:
            _draw_black_frame(lower_y_offset)
        elif Tgrad:
            lower_frame.plot_thermal_field(ax, cmap='bwr', show_cb=show_lower_cb, v_min=v_min, v_max=v_max)

        if upper_frame is None:
            _draw_black_frame(upper_y_offset)
        elif Tgrad:
            upper_frame.plot_thermal_field(ax, cmap='bwr', show_cb=show_upper_cb, v_min=v_min, v_max=v_max, y_offset=upper_y_offset)

        if not contours and not center and not box and not Tgrad:
            print("Nothing to plot!")
            return

        if self._contours_frame['lower'] is None or self._contours_frame['upper'] is None:
            self.find_hive_contours(f'2FCl-plot')

        if contours:
            lower_cnts = self._contours_frame['lower']
            upper_cnts = self._contours_frame['upper']

            for _, c in enumerate(lower_cnts):
                ax.plot(c[:, 0], c[:, 1], c='m')
            for _, c in enumerate(upper_cnts):
                ax.plot(c[:, 0], c[:, 1]+ThermalHive.gap_between_frames_mm+ThermalFrame.y_pcb, c='m')

        if center:
            CoM_upper = self.contours_CoM_frame['upper']
            CoM_lower = self.contours_CoM_frame['lower']
            for c in CoM_lower:
                ax.scatter(c[0], c[1], marker='x', c='m', s=15)
                ax.scatter(c[0], c[1], marker='o', c='m', s=10)
            for c in CoM_upper:
                ax.scatter(c[0], c[1]+ThermalHive.gap_between_frames_mm+ThermalFrame.y_pcb, marker='x', c='m', s=15)
                ax.scatter(c[0], c[1]+ThermalHive.gap_between_frames_mm+ThermalFrame.y_pcb, marker='o', c='m', s=10)

        if box:
            if (self.contours_box is not None and len(self.contours_box) > 0):
                for b in self.contours_box_frame['lower']:
                    ax.plot(b[0], b[1], c='#913ba8', lw=lw)
                for b in self.contours_box_frame['upper']:
                    ax.plot(b[0], b[1]+ThermalHive.gap_between_frames_mm+ThermalFrame.y_pcb, c='#913ba8', lw=lw)
            else:
                print("No contour box to plot.")

        ax.set_ylim(-ThermalHive.HIVE_PADDING, 2*ThermalFrame.y_pcb + ThermalHive.gap_between_frames_mm + ThermalHive.HIVE_PADDING)
        ax.set_xlim(-ThermalHive.HIVE_PADDING, ThermalFrame.x_pcb + ThermalHive.HIVE_PADDING)

    #---- SETTERS ----

    def set_origins(self, origin_upper:Tuple[float, float], origin_lower:Tuple[float, float]):
        '''
        Set the origins of the upper and lower frames through tuples (x0, y0).
        '''
        self.origin_x['upper'], self.origin_y['upper'] = origin_upper
        self.origin_x['lower'], self.origin_y['lower'] = origin_lower

    def set_isotherm(self, temperature:float) -> None:
        ''' define the cutoff isotherm above which clusters are interpreted'''
        assert temperature > 0, "Temperature isotherm must be > 0"
        self.isotherm = temperature


    #---- GETTERS ----

    def get_num_contours(self, frame:str = None) -> int:
        '''Returns the number of contours found over either or both frames

        `which` is set to
          None -> over all frames
          'upper' -> upper frame only
          'lower' -> lower frame only

        '''
        if frame is None:
            return self.n_contours_tot
        elif frame in ['upper', 'lower']:
            return self.n_contours[frame]
        else:
            raise ValueError("frame must be either 'upper', 'lower' or None")
        
    def get_wholehive_position(self):
        '''Returns the position of the cluster in the whole hive reference frame
        Origin is bottom-left of lower frame.
        If multiple clusters were found, returns None.
        '''
        if self.isPositionFix:
            return self.position_in_hive
        else:
            print(f"Could not define the cluster position (n_contours={self.n_contours}).")
            return None, None
    
    def get_pos_with_offset(self, frame:str, verbose:bool=False):
        assert frame in ['upper', 'lower'], "frame must be either 'upper' or 'lower'"

        if self.isPositionFix:
            if self.position_frame[frame] is not None:
                cprint(f"pos: {self.position_frame[frame]} ; origin: {self.origin_x[frame]}, {self.origin_y[frame]}")
                return self.add_offset(self.position_frame[frame], (self.origin_x[frame], self.origin_y[frame]))
            else:
                if verbose:
                    print(f"No cluster position defined for '{frame}' frame (n_contours={self.n_contours[frame]}).")
                return None

        else:
            print(f"Position is not fixed (n_contours={self.n_contours}).")
            return None

    def get_position(self, frame:str = None):
        assert frame in [None, 'upper', 'lower'], "frame must be either 'upper', 'lower' or None"

        if self.isPositionFix:
            if frame is None:
                return self.position
            elif frame in ['upper', 'lower']:
                return self.position_frame[frame]

        else:
            print(f"Position is not fixed (n_contours={self.n_contours}).")
            return None
        
    def get_contour_box(self):
        '''
        Returns the bounding box of the contour if position is fixed, else None
        '''
        if self.isPositionFix:
            return self.contours_box
        else:
            return None


    # ---- INTERNAL METHODS ----

    def _find_contours_oneframe(self, frame:str) -> None:
        '''
        Find the cluster contour defined by the chosen isotherm
        `frame` should be in [`upper`, `lower`]. Data is directly stored in member variables.
        '''
        assert frame in ['upper', 'lower'], "frame must be either 'upper' or 'lower'"

        tf = self.frames[frame] # ThermalFrame
        if tf is None:
            self.contours_CoM_frame[frame] = []
            self.contours_box_frame[frame] = []
            self.areas[frame] = []
            self.n_contours[frame] = 0
            self._contours_frame[frame] = []
            return

        # # 3. Find contour line(s)
        contours = self._inner_find_contours(tf)

        # Here we get rid of contours with very small areas
        # And find Center of Mass
        CoM, box, areas = self._center_of_mass(contours, method='polygon')
        #CoM, box, areas = self._center_of_mass(contours, method='box')

        # assign results to relevant member variables
        self.contours_CoM_frame[frame] = CoM
        self.contours_box_frame[frame] = box
        self.areas[frame] = areas
        self.n_contours[frame] = len(areas)
        self._contours_frame[frame] = contours

    def _inner_find_contours(self, tf:ThermalFrame) -> QuadContourSet:
        # 1. Get the interpolated thermal field (not temperature sensors data array)
        if tf.thermal_field is None:
            tf.calculate_thermal_field()

        t = tf.thermal_field
        # flip t vertically to match the image orientation
        t = np.flipud(t)
        # 2. Thresholding to the target temperature
        thresholded_field = np.copy(t)

        # Make all values < threshold_temp equals to 0
        thresholded_field[np.where(thresholded_field < self.isotherm)] = 0
        # Make all values > threshold_temp equals to threshold_temp
        thresholded_field[np.where(thresholded_field >= self.isotherm)] = 255
        
        # 3. Find contour line(s)
        _interactive_state = plt.isinteractive() # save state
        plt.figure()
        plt.ion()
        contours = plt.contourf(thresholded_field, extent=tf.extent, levels=1,)
        contours = contours.allsegs
        contours = contours[1:][0]  # Skip the first contour (the whole frame?)

        if not _interactive_state: # only turn off if it was off beforehand
            plt.ioff()
        plt.close()

        return contours

    def _center_of_mass(self, contours:QuadContourSet, method:str='box', min_area:float=300):
        '''
        Compute contour properties for all contours inputted: CoM, area, and bbox
        
        :param method: 'box' (bounding box method = fast) or 'polygon'
        :param min_area: Contours smaller than this value will be ignored
        :return: lists of CoM, bbox, areas in a tuple
        '''
        min_area_size = min_area  # 300mm^2 = 3cm^2

        CoM, box, areas = [], [], []
        for c in contours:
            p = None  # temp CoM
            b = None  # temp box
            A = 0     # temp area
            x = c[:, 0]
            y = c[:, 1]
            if method == 'polygon':
                g = (x[:-1] * y[1:] - x[1:] * y[:-1])
                # Polygon area
                A = 0.5 * g.sum()
                cx = ((x[:-1] + x[1:]) * g).sum()
                cy = ((y[:-1] + y[1:]) * g).sum()
                if abs(A) < 1e-4:
                    continue
                p = np.round(1. / (6 * A) * np.array([cx, cy]), 2)
            elif method == 'box':
                # Bounding box
                b = np.array([
                    [min(x), min(x), max(x), max(x), min(x)],
                    [min(y), max(y), max(y), min(y), min(y)],
                ])
                b = np.round(b, 2)
                A = np.round((max(x) - min(x)) * (max(y) - min(y)), 1)
                # Center of the box
                p = (np.round(min(x) + (max(x) - min(x)) / 2, 2),
                     np.round(min(y) + (max(y) - min(y)) / 2, 2))

            # If the area too small, ignore it
            if A > min_area_size:
                areas.append(A)
                CoM.append(p)

                if method == 'box':
                    box.append(b)

        return CoM, box, areas