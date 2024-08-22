import tkinter.messagebox
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from reproject import reproject_interp
from matplotlib import get_backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Cursor, TextBox, Button, CheckButtons, RadioButtons
from matplotlib.contour import QuadContourSet
import tkinter
from tkinter import filedialog, simpledialog
from itertools import cycle
import argparse
from typing import Callable, Any
import sys, re
from os.path import isfile


class ImageLayer(object):
    ##### CONSTRUCTORS #####
    def __init__(self, fits_path: str, hdu: fits.ImageHDU, name: str=None, image_type: str='auto', is_bkg_layer: bool=True,
                 vmin_pc: float=None, vmax_pc: float=None, levels_factor: float=3, levels_base: float=np.sqrt(2),
                 slice_type: str='none', ra_direction: str='left',
                 plot_args: dict={'cmap': 'inferno', 'colors': None, 'linewidths': 2, 'alpha': 0.7}):
        self.fits_path = fits_path
        self.hdu = hdu
        self.data: np.ndarray = np.squeeze(hdu.data)
        self.wcs = WCS(hdu.header, naxis=2)
        self.fix_ra_direction(ra_direction)
        self.name = name if name is not None else self.fits_path.split('/')[-1]
        self.image_type = image_type
        if self.image_type == 'auto':
            if self.hdu.header['NAXIS'] > 2:
                self.image_type = 'cube'
            elif self.hdu.header['NAXIS'] == 2:
                self.image_type = 'image'
            else:
                raise ValueError(f"Unsupported number of axes in data: {hdu_in.header['NAXIS']}")
        self.is_bkg_layer = is_bkg_layer
        self.vmin_pc = vmin_pc if vmin_pc is not None else 0.5
        self.vmax_pc = vmax_pc if vmax_pc is not None else 99.5
        self.slice_type = slice_type
        self.slice_func = self.slice_func_none
        self.spectral_axis_len = 0
        self.slice_arg1, self.slice_arg2 = 0, 0
        if self.image_type == 'cube':
            self.spectral_axis_len = self.data.shape[0]
            if self.slice_type == 'manual':
                self.slice_arg1, self.slice_arg2 = 0, self.spectral_axis_len - 1
                self.slice_func = self.slice_func_manual
            elif self.slice_type == 'window':
                self.slice_arg1, self.slice_arg2 = self.spectral_axis_len // 2, self.spectral_axis_len // 1
                self.slice_func = self.slice_func_window
                # For example, a width of 100% of the spectral axis length
        self.data_flat = None
        if self.image_type == 'cube':
            self.data_flat = self.slice_func()
        else:
            self.data_flat = self.data
        self.vmin = np.nanpercentile(self.data_flat, self.vmin_pc)
        self.vmax = np.nanpercentile(self.data_flat, self.vmax_pc)
        self.levels_factor = levels_factor
        self.levels_base = levels_base
        self.artist = None
        self.plot_args = plot_args

    ##### DECONSTRUCTOR #####
    def __del__(self):
        # clean up artists
        if self.artist: self.artist.remove()

    ##### DEFAULTS #####
    def __repr__(self):
        return f"""
Image Layer
-----------
File: {self.fits_path}
Data Shape: {self.data.shape}
Name: {self.name}
Image Type: {self.image_type}
Slice Type: {self.slice_type}
IsBackground: {self.is_bkg_layer}
Plot Args: {self.plot_args}
"""
    
    def __str__(self):
        return f"{self.name}"

    ##### INTERNAL METHODS #####
    def fix_ra_direction(self, direction: str='left'):
        ra_left = self.wcs.pixel_to_world(0,0)
        ra_right = self.wcs.pixel_to_world(10,0)
        assert direction in ['left', 'right'], "Preferred RA direction must be either 'right' or 'left'."
        if (ra_left.ra < ra_right.ra and direction == 'left') or (ra_left.ra > ra_right.ra and direction == 'right'):
            print(f"[INFO] Fixing WCS of {self.fits_path} such that RA is increasing to the {direction}.")
            hdr_tmp = self.hdu.header.copy()
            hdr_tmp['CRPIX1'] = hdr_tmp['NAXIS1'] + 1 - hdr_tmp['CRPIX1']
            if 'CD1_1' in hdr_tmp:
                hdr_tmp['CD1_1'] = -hdr_tmp['CD1_1']
            elif 'CDELT1' in hdr_tmp:
                hdr_tmp['CDELT1'] = -hdr_tmp['CDELT1']
            if hdr_tmp['NAXIS'] == 4: hdr_tmp['NAXIS'] = 3

            reproj_data, _ = reproject_interp((self.data, self.wcs), WCS(hdr_tmp, naxis=2), shape_out=self.data.shape)

            self.hdu.header = hdr_tmp
            self.wcs = WCS(self.hdu.header, naxis=2)
            self.hdu.data = reproj_data
            self.data = reproj_data
            
    def slice_func_none(self):
        return np.nanmean(self.data, axis=0)

    def slice_func_manual(self):
        start_idx = min(self.slice_arg1, self.slice_arg2)
        end_idx = max(self.slice_arg1, self.slice_arg2)
        start_idx = max(0, start_idx)
        end_idx = min(self.spectral_axis_len - 1, end_idx)
        return np.nanmean(self.data[start_idx:end_idx+1, :, :], axis=0)

    def slice_func_window(self):
        central_idx = self.slice_arg1
        width = self.slice_arg2
        start_idx = max(0, int(central_idx - width // 2))
        end_idx = min(self.spectral_axis_len - 1, int(central_idx + width // 2))
        return np.nanmean(self.data[start_idx:end_idx+1, :, :], axis=0)

    def calc_flat_rms(self):
        return np.sqrt(np.nanmean(self.data_flat**2))
    
    def calc_flat_std(self):
        return np.nanstd(self.data_flat)

    ##### SET/GET #####
    def set_name(self, name: str):
        self.name = name

    def set_is_bkg_layer(self, is_bkg_layer: bool):
        self.is_bkg_layer = is_bkg_layer
    
    def set_vmin(self, vmin_pc: float):
        self.vmin = np.nanpercentile(self.data_flat, vmin_pc)

    def set_vmax(self, vmax_pc: float):
        self.vmax = np.nanpercentile(self.data_flat, vmax_pc)
    
    def set_levels_factor(self, levels_factor: float):
        self.levels_factor = levels_factor
    
    def set_levels_base(self, levels_base: float):
        self.levels_base = levels_base

    def set_slice_type(self, slice_type: str):
        match slice_type:
            case 'none':
                self.slice_type = slice_type
                self.slice_func = self.slice_func_none
            case 'manual':
                self.slice_type = slice_type
                self.slice_arg1, self.slice_arg2 = 0, self.spectral_axis_len - 1
                self.slice_func = self.slice_func_manual
            case 'window':
                self.slice_type = slice_type
                self.slice_arg1, self.slice_arg2 = self.spectral_axis_len // 2, self.spectral_axis_len // 1
                self.slice_func = self.slice_func_window
            case _:
                raise ValueError(f"{slice_type} is not a supported slicing method.")
        # Update the flat data being stored
        self.data_flat = self.slice_func()
        self.vmin = self.set_vmin(self.vmin_pc)
        self.vmax = self.set_vmax(self.vmax_pc)

    def get_display_data(self) -> np.ndarray:
        return self.data_flat
    
    def get_is_bkg_layer(self) -> bool:
        return self.is_bkg_layer
    
    ##### PLOT METHODS #####
    def plot_layer(self, ax):
        if self.is_bkg_layer:
            self.artist = ax.imshow(self.data_flat, origin='lower', cmap=self.plot_args['cmap'], vmin=self.vmin, vmax=self.vmax)
        else:
            if isinstance(self.artist, QuadContourSet):
                # Clean up unnecessary contours before plotting more
                self.artist.remove()
            
            self.artist = ax.contour(self.data_flat, colors=self.plot_args['colors'], linewidths=self.plot_args['linewidths'],
                                     alpha=self.plot_args['alpha'], negative_linestyles='dashed', transform=ax.get_transform(self.wcs),
                                     levels=self.calc_flat_rms()*self.levels_factor*self.levels_base**np.arange(0,3))


class LayerManager(object):
    ##### CONSTRUCTORS #####
    def __init__(self, fig: plt.Figure, ax, main_ax_region: list=[0.25, 0.2, 0.75, 0.8],
                 colour_wheel: list=['hotpink','fuchsia','darkviolet','mediumslateblue','royalblue','deepskyblue','cyan',
                                     'aquamarine','lime','gold','lightsalmon','indianred','dimgrey','whitesmoke']):
        self.layers: list[ImageLayer] = list()
        self.current_layer: ImageLayer = None
        self.background_layer: ImageLayer = None
        self.fig = fig
        self.ax = ax
        self.main_ax_region = main_ax_region
        self.bbox_skycoords: list[SkyCoord] = None
        self.ra_direction: str = 'left'
        self.API_update_widgets_callback: function = lambda: None
        self.API_update_radio_buttons_layers_callback: function = lambda: None
        self.API_update_spectral_sliders_callback: function = lambda: None
        self.handle_radiobuttons_layers: RadioButtons = None
        self.handle_cursor: Cursor = None
        self.handle_spectral_sliders_textboxes: list[Slider, TextBox] = None

        # Build a colourwheel for ImageLayer contours' plotargs
        self.colour_wheel = colour_wheel
        self.col_iter = cycle(self.colour_wheel)

    ##### DECONSTRUCTORS #####
    def __del__(self):
        # for layer in self.layers:
        #     layer.get_hdu().close()
        del self
    
    ##### DEFAULTS #####
    def __repr__(self):
        return f"""
LayerManager
------------
ImageLayers: {[layer.name for layer in self.layers]}
Current Layer: {self.current_layer}
Background Layer: {self.background_layer}
RA direction: {self.ra_direction}
API_update_widgets: {self.API_update_widgets_callback}
API_update_radiobuttons: {self.API_update_radio_buttons_layers_callback}
API_update_sliders: {self.API_update_spectral_sliders_callback}
handle_radiobuttons: {self.handle_radiobuttons_layers}
handle_cursor: {self.handle_cursor}
handle_sliders_textboxes: {self.handle_spectral_sliders_textboxes}
"""
    
    def __str__(self):
        return f"LayerManager with ImageLayers: {[layer.name for layer in self.layers]}"

    ##### INTERNAL METHODS #####
    def ax_lims_to_bbox_skycoords(self):
        xlims = self.ax.get_xlim()
        ylims = self.ax.get_ylim()
        bkg_wcs = self.background_layer.wcs
        # start with coord at lower-left and work clockwise
        sky_0_0, sky_0_1, sky_1_1, sky_1_0 = bkg_wcs.pixel_to_world([*[xlims[0]]*2, *[xlims[1]]*2],ylims*2)
        return [sky_0_0, sky_0_1, sky_1_1, sky_1_0]
    
    def bbox_skycoords_to_ax_lims(self):
        xmin = np.nanmin([sky.ra.value for sky in self.bbox_skycoords])
        xmax = np.nanmax([sky.ra.value for sky in self.bbox_skycoords])
        ymin = np.nanmin([sky.dec.value for sky in self.bbox_skycoords])
        ymax = np.nanmax([sky.dec.value for sky in self.bbox_skycoords])
        if self.ra_direction == 'left':
            x_tmp = xmin
            xmin = xmax
            xmax = x_tmp
        bkg_wcs = self.background_layer.wcs
        frame = 'icrs'  # default to ICRS
        if 'RADESYS' in self.background_layer.hdu.header: frame = self.background_layer.hdu.header['RADESYS'].strip(' ').lower()
        if 'RADECSYS' in self.background_layer.hdu.header: frame = self.background_layer.hdu.header['RADECSES'].strip(' ').lower()
        sky_min = SkyCoord(xmin*u.deg,ymin*u.deg,frame=frame)
        sky_max = SkyCoord(xmax*u.deg,ymax*u.deg,frame=frame)
        pix_0_0 = bkg_wcs.world_to_pixel(sky_min)
        pix_1_1 = bkg_wcs.world_to_pixel(sky_max)
        xlims = [pix_0_0[0], pix_1_1[0]]
        ylims = [pix_0_0[1], pix_1_1[1]]
        return [xlims, ylims]

    ##### SET/GET #####
    def set_current_layer(self, layer: ImageLayer):
        self.current_layer = layer
        # self.radiobuttons_layers.set_active(self.layers.index(self.current_layer))
        self.API_update_widgets_callback()
        self.handle_spectral_sliders_textboxes = self.API_update_spectral_sliders_callback()
        print(self.current_layer.__repr__())
    
    def set_background_layer(self, layer: ImageLayer):
        if layer:
            layer.is_bkg_layer = True
        if self.background_layer:
            self.background_layer.is_bkg_layer = False
        self.background_layer = layer

    def set_API_widgets_update(self, plugin_widgets_updater: Callable):
        self.API_update_widgets_callback = plugin_widgets_updater
    
    def set_API_radiobuttons_layers_list_update(self, plugin_layers_buttons_updater: Callable):
        self.API_update_radio_buttons_layers_callback = plugin_layers_buttons_updater

    def set_API_spectral_sliders_update(self, plugin_spectral_sliders_updater: Callable):
        self.API_update_spectral_sliders_callback = plugin_spectral_sliders_updater
    
    def set_handle_radiobuttons_layers_list(self, layers_list_radiobuttons: RadioButtons):
        self.handle_radiobuttons_layers = layers_list_radiobuttons

    def set_handle_cursor(self, image_cursor: Cursor):
        self.handle_cursor = image_cursor
    
    def set_handle_spectral_sliders_textboxes(self, spectral_sliders_textboxes: list[Slider, TextBox]):
        self.handle_spectral_sliders_textboxes = spectral_sliders_textboxes

    def get_layers(self):
        return self.layers

    def get_current_layer(self):
        return self.current_layer
    
    def get_background_layer(self):
        return self.background_layer
    
    def get_fig(self):
        return self.fig
    
    def get_ax(self):
        return self.ax

    ##### PLOT METHODS #####
    def update_plot(self):
        if self.layers:
            if self.ax is None:
                self.ax = self.fig.add_subplot(projection=self.background_layer.wcs)
                self.ax.set_xlabel('RA (J2000)')
                self.ax.set_ylabel('Dec. (J2000)',labelpad=-0.5)
                self.ax.coords[0].set_ticklabel(exclude_overlapping=True)
                self.handle_cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
            for layer in self.layers:
                layer.plot_layer(self.ax)
            # Set axes limit to reflect previous background (or new axes') view
            if len(self.layers) == 1:
                self.bbox_skycoords = self.ax_lims_to_bbox_skycoords()
            elif len(self.layers) > 1:
                xlim, ylim = self.bbox_skycoords_to_ax_lims()
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
            plt.subplots_adjust(*self.main_ax_region)
        else:
            self.ax.remove()
            self.ax = None
            self.bbox_skycoords = None
            info_dialog_wrapper(title="INFO",message="There are currently no layers - add one to begin.")

    ##### PUBLIC METHODS #####
    def add_layer(self, fits_path: str, hdu: fits.ImageHDU, name: str=None, is_bkg_layer: bool=True, vmin_pc: float=None,
                  vmax_pc: float=None, levels_factor: float=3, levels_base: float=np.sqrt(2), slice_type: str='none',
                  plot_args: dict={'cmap': 'inferno', 'colors': None, 'linewidths': 2, 'alpha': 0.7}):
        plot_args['colors'] = next(self.col_iter)
        if self.layers == []:
            is_bkg_layer = True
        layer = ImageLayer(fits_path, hdu, name=name, image_type='auto', is_bkg_layer=is_bkg_layer, vmin_pc=vmin_pc, vmax_pc=vmax_pc,
                           levels_factor=levels_factor, levels_base=levels_base, slice_type=slice_type, ra_direction=self.ra_direction,
                           plot_args=plot_args)
        self.layers.append(layer)
        self.set_current_layer(layer)
        if self.current_layer.is_bkg_layer:
            self.current_layer.set_is_bkg_layer(False)
            self.send_layer_to_background(do_update=False)
        if self.ax is None:
            self.ax = self.fig.add_subplot(projection=self.background_layer.wcs)
            self.ax.set_xlabel('RA (J2000)')
            self.ax.set_ylabel('Dec. (J2000)',labelpad=-0.5)
            self.ax.coords[0].set_ticklabel(exclude_overlapping=True)
            self.handle_cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
        self.handle_radiobuttons_layers = self.API_update_radio_buttons_layers_callback()
        self.update_plot()

    def remove_layer(self, layer: ImageLayer):
        if layer in self.layers:
            self.layers.remove(layer)
            if self.current_layer == layer:
                self.set_current_layer(self.layers[0]) if self.layers else self.set_current_layer(None)
            if self.background_layer == layer:
                self.set_background_layer(self.layers[0]) if self.layers else self.set_background_layer(None)
                if self.ax in self.fig.axes:
                    self.ax.remove()
                    self.ax = None
                # self.ax = self.fig.add_subplot(projection=WCS())
                # self.ax.set_xlabel('RA (J2000)')
                # self.ax.set_ylabel('Dec. (J2000)',labelpad=-0.5)
                # self.ax.coords[0].set_ticklabel(exclude_overlapping=True)
                # cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
            layer.__del__()
            self.handle_radiobuttons_layers = self.API_update_radio_buttons_layers_callback()
            self.update_plot()
        else:
            if layer is None:
                layer_name = "(no layer)"
            else:
                layer_name = layer.name
            raise UserWarning(f"Could not remove layer {layer_name}.")

    def send_layer_to_background(self, do_update: bool=True):
        if self.current_layer and self.current_layer.is_bkg_layer == False:
            if len(self.layers) > 1:
                self.bbox_skycoords = self.ax_lims_to_bbox_skycoords()
            self.set_background_layer(self.current_layer)
            # Change axes WCS by removing and re-creating axes with new projection
            if self.ax in self.fig.axes:
                self.ax.remove()
            self.ax = self.fig.add_subplot(projection=self.background_layer.wcs)
            self.ax.set_xlabel('RA (J2000)')
            self.ax.set_ylabel('Dec. (J2000)',labelpad=-0.5)
            self.ax.coords[0].set_ticklabel(exclude_overlapping=True)
            self.handle_cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
            # Reflect change in background layer by updating sliders, vmin/vmax, axes WCS, etc.
            self.API_update_widgets_callback()
            if do_update:
                self.update_plot()
        else:
            raise UserWarning(f"Could not change background layer {str(self.background_layer)} to {str(self.current_layer)}.")
        
    def layer_names(self):
        names = []
        for layer in self.layers:
            names.append(layer.name)
        return names

    



    

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, default='', help="The full file path to the FITS file to use.")
parser.add_argument("-t", "--type", type=str, choices=['image', 'cube', 'auto'], default='auto',
                    help="The type of data contained in the FITS file (2D or 3D).")
parser.add_argument("-s", "--slice-method", type=str, choices=['none', 'manual', 'window'], default='none',
                    help="The method with which to define slices of a 3D cube.")
parser.add_argument("-v", "--vminmax", nargs="+", type=float, default=[0.5,99.5],
                    help="Min/max percentiles when normalising the data displayed.")
parser.add_argument("-b", "--background", type=str, default='',
                    help="The full file path to a background FITS image (2D) to use, changing the main file's data to be overlaid as contours.")
parser.add_argument("-o", "--output-file", default=sys.stdout,
                    help="The full file path to a desired output file. By default the output is send to standard out.")
args = parser.parse_args()
print(args)


##### I/O FUNCTIONS #####
def open_file_dialog():
    root = tkinter.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a file")
    root.destroy()  # Close the Tkinter root window
    if file_path == (): file_path = None
    return file_path

def save_file_dialog():
    root = tkinter.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(title="Select output file")
    root.destroy()
    if file_path == (): file_path = None
    return file_path

def switch_outstream(event):
    global out_stream

    new_out_file = save_file_dialog()
    if out_stream != sys.stdout:
        out_stream.close()
    if new_out_file == ():
        out_stream = sys.stdout
    else:
        if not isfile(new_out_file):    # quickly create and close the file
            out_stream = open(new_out_file, 'x')
            out_stream.close()
        out_stream = open(new_out_file, 'a')
    return

def input_string_dialog(**dialog_kwargs):
    root = tkinter.Tk()
    root.withdraw()
    input_str = simpledialog.askstring(**dialog_kwargs)
    root.destroy()
    if input_str == (): input_str = 'Missing dialog string'
    return input_str

def warning_dialog_wrapper(**dialog_kwargs):
    root = tkinter.Tk()
    root.withdraw()
    tkinter.messagebox.showwarning(**dialog_kwargs)
    root.destroy()
    return

def error_dialog_wrapper(**dialog_kwargs):
    root = tkinter.Tk()
    root.withdraw()
    tkinter.messagebox.showerror(**dialog_kwargs)
    root.destroy()
    return

def info_dialog_wrapper(**dialog_kwargs):
    root = tkinter.Tk()
    root.withdraw()
    tkinter.messagebox.showinfo(**dialog_kwargs)
    root.destroy()
    return


# Assign args
interactive_type = args.type
interaction_method = args.slice_method
vmin, vmax = args.vminmax
fits_file = args.data
fits_file_bkg = args.background
if args.output_file == sys.stdout:
    out_stream = sys.stdout
else:
    out_stream = open(args.output_file)

# Clean args
if fits_file != '' and not isfile(fits_file):
    raise FileNotFoundError(f"{fits_file} does not exist.")
if fits_file_bkg != '' and not isfile(fits_file_bkg):
    raise FileNotFoundError(f"{fits_file_bkg} does not exist.")

# State the back-end
print(f"Matplotlib backend in use: {get_backend()} (QtAgg preferred).")

# Prompt for selecting file if none given
if fits_file == '':
    fits_file = open_file_dialog()


# Create the figure and axis
fig, ax = plt.subplots(subplot_kw={'projection': WCS()})#,gridspec_kw={'left':0.2,'bottom':0.25,'right':0.8}, layout='constrained')
main_ax_region = [0.25, 0.2, 0.75, 0.8]
plt.subplots_adjust(*main_ax_region)  # Adjust figure to accommodate sliders

layer_manager = LayerManager(fig, ax, main_ax_region)

if fits_file != '':
    # Load the FITS file and data
    hdu_in = fits.open(fits_file)[0]
    data_in = np.squeeze(hdu_in.data) # REMOVE ANY DEGENERATE STOKES AXIS
    wcs_in = WCS(hdu_in.header, naxis=2)

    print(f"Input FITS data has shape {data_in.shape}")
    layer_manager.add_layer(fits_file, hdu_in, None, True, vmin, vmax, slice_type=interaction_method)
if fits_file_bkg != '':
    # Load the FITS file and data
    hdu_in = fits.open(fits_file_bkg)[0]
    data_in = np.squeeze(hdu_in.data) # REMOVE ANY DEGENERATE STOKES AXIS
    wcs_in = WCS(hdu_in.header, naxis=2)

    print(f"Input FITS data has shape {data_in.shape}")
    layer_manager.add_layer(fits_file_bkg, hdu_in, None, True, vmin, vmax)


# Initialise some things we need to hold on to
im = None
last_skycoord = None  # Variable to store the last clicked SkyCoord

##### WIDGET AXES LAYOUT ##########################################################################################
ax_button_out_file = plt.axes([0.82, 0.72, 0.15, 0.03])
ax_textbox_coords = plt.axes([0.82, 0.85, 0.15, 0.04], facecolor='lightgoldenrodyellow')
ax_textbox_coords_deg = plt.axes([0.84, 0.80, 0.13, 0.04], facecolor='lightgoldenrodyellow')
ax_button_save = plt.axes([0.82, 0.76, 0.15, 0.03])
ax_radiobutton_layers = plt.axes([0.02, 0.1, 0.16, 0.75], facecolor='lightgoldenrodyellow')
fig.text(0.04, 0.86, "Layers", fontsize='xx-large', fontweight='heavy')
# ax_button_refresh_layers_list = plt.axes([0.1, 0.91, 0.1, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_1 = plt.axes([0.3, 0.1, 0.3, 0.03], facecolor='lightgoldenrodyellow')
ax_slider_2 = plt.axes([0.3, 0.05, 0.3, 0.03], facecolor='lightgoldenrodyellow')
ax_textbox_1 = plt.axes([0.65, 0.1, 0.1, 0.03], facecolor='lightgoldenrodyellow')
ax_textbox_2 = plt.axes([0.65, 0.05, 0.1, 0.03], facecolor='lightgoldenrodyellow')
ax_textbox_vmin = plt.axes([0.82, 0.68, 0.1, 0.03], facecolor='lightgoldenrodyellow')
ax_textbox_vmax = plt.axes([0.82, 0.64, 0.1, 0.03], facecolor='lightgoldenrodyellow')
ax_button_add_layer = plt.axes([0.82, 0.60, 0.1, 0.03], facecolor='lightgoldenrodyellow')
ax_button_remove_layer = plt.axes([0.82, 0.56, 0.1, 0.03], facecolor='lightgoldenrodyellow')
ax_button_send_to_back = plt.axes([0.82, 0.52, 0.1, 0.03], facecolor='lightgoldenrodyellow')
ax_textbox_contour_factor = plt.axes([0.82, 0.48, 0.1, 0.03], facecolor='lightgoldenrodyellow')
ax_textbox_contour_base = plt.axes([0.82, 0.44, 0.1, 0.03], facecolor='lightgoldenrodyellow')
ax_radiobutton_slice_type = plt.axes([0.82, 0.3, 0.1, 0.1], facecolor='lightgoldenrodyellow')


##### WIDGET HANDLES AND FUNCTIONS ################################################################################
# Option to select output file via button + Tkinter prompt
button_out_file = Button(ax_button_out_file, 'Change output stream')
button_out_file.on_clicked(switch_outstream)
# Add a TextBox to display the last-clicked coordinate
textbox_coords = TextBox(ax_textbox_coords, 'SkyCoord:', initial="No coord selected")
textbox_coords_deg = TextBox(ax_textbox_coords_deg, 'SkyCoord (deg):', initial="No coord selected")

# Function to store world coordinates on mouse click
def onclick(event):
    global last_skycoord
    ax = layer_manager.get_ax()

    if event.inaxes == ax and event.button == 1:  # Left mouse button
        x, y = event.xdata, event.ydata
        # Convert pixel coordinates to world coordinates
        world_coords = ax.wcs.pixel_to_world(x, y)
        last_skycoord = world_coords
        # Update the TextBox with the last clicked coordinate
        textbox_coords.set_val(f"{last_skycoord.to_string(style='hmsdms',precision=3)}")
        textbox_coords_deg.set_val(f"{last_skycoord.to_string(style='decimal',precision=5)}")

# Connect the onclick event to the figure
fig.canvas.mpl_connect('button_press_event', onclick)

# Function to write world coordinates
def print_coord(event):
    if last_skycoord is not None:
        _ = out_stream.write(f"{last_skycoord.to_string(style='decimal',precision=5).replace(' ',',')}\n")
        out_stream.flush()

# Button to write world coordinates to out_stream
button_save = Button(ax_button_save, 'Print Coord')
button_save.on_clicked(print_coord)

# Setup RadioButtons for displaying list of layers (to select/switch to)
def create_radiobuttons_layers_list():
    ax_radiobutton_layers.clear()
    layer_names = layer_manager.layer_names()
    current_layer_idx = layer_names.index(layer_manager.get_current_layer().name)
    radiobutton_layers = RadioButtons(ax_radiobutton_layers, layer_names, current_layer_idx,
                                      label_props={'fontsize': [14]}, radio_props={'s': [64]})

    # Update plot to selected layer
    def update_current_layer(label):
        for layer in layer_manager.get_layers():
            if layer.name == label:
                layer_manager.set_current_layer(layer)
                break

    radiobutton_layers.on_clicked(update_current_layer)
    return radiobutton_layers

# Setup first slice sliders
def create_spectral_sliders_textboxes():
    ax_slider_1.clear()
    ax_slider_2.clear()
    ax_textbox_1.clear()
    ax_textbox_2.clear()
    spectral_axis_len = layer_manager.get_current_layer().spectral_axis_len
    slider_1 = slider_2 = textbox_1 = textbox_2 = None
    if layer_manager.get_current_layer().image_type == 'cube' and layer_manager.get_current_layer().slice_type != 'none':
        arg1, arg2 = layer_manager.get_current_layer().slice_arg1, layer_manager.get_current_layer().slice_arg2
        if layer_manager.get_current_layer().slice_type == 'window':
            # Add sliders for central index and width
            slider_1 = Slider(ax_slider_1, 'Central Channel', 0, spectral_axis_len - 1, valinit=arg1, valstep=1)
            slider_2 = Slider(ax_slider_2, 'Window Width', 1, spectral_axis_len, valinit=arg2, valstep=1)
        else:   # assume 'manual' as default
            # Add sliders for start and end indices
            slider_1 = Slider(ax_slider_1, 'Start Channel', 0, spectral_axis_len - 1, valinit=arg1, valstep=1)
            slider_2 = Slider(ax_slider_2, 'End Channel', 0, spectral_axis_len - 1, valinit=arg2, valstep=1)

        # Plot update on slice change
        def update_slice_slider_1(val):
            textbox_1.set_val(str(slider_1.val))
            current_layer = layer_manager.get_current_layer()
            current_layer.slice_arg1 = int(slider_1.val)
            current_layer.data_flat = current_layer.slice_func()
            layer_manager.update_plot()

        def update_slice_slider_2(val):
            textbox_2.set_val(str(slider_2.val))
            current_layer = layer_manager.get_current_layer()
            current_layer.slice_arg2 = int(slider_2.val)
            current_layer.data_flat = current_layer.slice_func()
            layer_manager.update_plot()

        # Connect the sliders to the update function
        slider_1.on_changed(update_slice_slider_1)
        slider_2.on_changed(update_slice_slider_2)


        # Add text boxes for direct input of slider values
        textbox_1 = TextBox(ax_textbox_1, '', initial=str(arg1))
        textbox_2 = TextBox(ax_textbox_2, '', initial=str(arg2))

        def update_from_textbox_arg1(text):
            try:
                value = int(text)
                slider_1.set_val(value)
            except ValueError:
                pass

        def update_from_textbox_arg2(text):
            try:
                value = int(text)
                slider_2.set_val(value)
            except ValueError:
                pass

        textbox_1.on_submit(update_from_textbox_arg1)
        textbox_2.on_submit(update_from_textbox_arg2)

    return slider_1, slider_2, textbox_1, textbox_2


# Add text boxes for vmin and vmax
textbox_vmin = TextBox(ax_textbox_vmin, 'vmin', initial=str(vmin))
textbox_vmax = TextBox(ax_textbox_vmax, 'vmax', initial=str(vmax))

def update_vmin_vmax(text):
    current_layer = layer_manager.get_current_layer()
    current_layer.set_vmin(float(textbox_vmin.text))
    current_layer.set_vmax(float(textbox_vmax.text))
    layer_manager.update_plot()

textbox_vmin.on_submit(update_vmin_vmax)
textbox_vmax.on_submit(update_vmin_vmax)

# Option to add new layer from file via button + Tkinter prompt
button_add_layer = Button(ax_button_add_layer, 'Add layer')

def new_layer_from_file(event):
    file_in = 'notafile'
    while file_in is not None and not isfile(file_in):
        file_in = open_file_dialog()
    if file_in is not None:
        try:
            hdu = fits.open(file_in)[0]
        except FileNotFoundError as e:
            error_dialog_wrapper(title=f"{type(e)}",message=f"{e.strerror}. Could not open {file_in}. Aborting new layer creation.")
            return
        layer_name = None
        while layer_name is None or layer_name in layer_manager.layer_names():
            # try:
            #     layer_append_num = int(re.match('.*?([0-9]+)$',layer_name).group(1))
            #     layer_append_num += 1
            #     layer_append_num = str(layer_append_num)
            #     end_str_gobble_len = len(layer_append_num)
            # except ValueError:
            #     layer_append_num = ' 1'
            #     end_str_gobble_len = 0
            layer_name = input_string_dialog(title=f"New layer from {file_in}",
                                             prompt="Enter a unique layer name")
        layer_manager.add_layer(file_in,hdu,name=layer_name,is_bkg_layer=False)

button_add_layer.on_clicked(new_layer_from_file)

# Option to remove current layer via button + Tkinter prompt
button_remove_layer = Button(ax_button_remove_layer, 'Remove current layer')

def confirm_layer_removal(event):
    root = tkinter.Tk()
    root.withdraw()
    do_remove = tkinter.messagebox.askyesno(title='Confirmation',
                                            message=f"Are you sure you want to remove the {layer_manager.get_current_layer().name} layer?")
    if do_remove:
        try:
            layer_manager.remove_layer(layer_manager.get_current_layer())
        except UserWarning as w:
            warning_dialog_wrapper(title=f"{type(w)}",message=f"{w} Layer removal failed.")
    root.destroy()

button_remove_layer.on_clicked(confirm_layer_removal)

# Option to send current layer to the background via button
button_send_to_back = Button(ax_button_send_to_back, 'Set layer as background')

def send_layer_to_back(event):
    try:
        layer_manager.send_layer_to_background()
    except UserWarning as w:
        extra_msg = ''
        if layer_manager.get_current_layer() is layer_manager.get_background_layer():
            extra_msg = "This layer is already the background layer!"
        warning_dialog_wrapper(title=f"{type(w)}",message=f"{w} {extra_msg}")

button_send_to_back.on_clicked(send_layer_to_back)

# Add textboxes for contour levels factor and base




# Add RadioButtons to switch slice_type of 3D data layer




##### END WIDGET DEFNITIONS ###################################################################################

def update_widgets_visibility():
    current_layer = layer_manager.get_current_layer()
    if current_layer.image_type == 'cube' and current_layer.slice_type != 'none':
        ax_slider_1.set_visible(True)
        ax_slider_2.set_visible(True)
        # and set to the layer's previously stored values
        # layer_manager.handle_spectral_sliders_textboxes[0].set_val(current_layer.slice_arg1)
        # layer_manager.handle_spectral_sliders_textboxes[1].set_val(current_layer.slice_arg2)
        ax_textbox_1.set_visible(True)
        ax_textbox_2.set_visible(True)
    else:
        ax_slider_1.set_visible(False)
        ax_slider_2.set_visible(False)
        ax_textbox_1.set_visible(False)
        ax_textbox_2.set_visible(False)

    if current_layer.get_is_bkg_layer():
        ax_textbox_vmin.set_visible(True)
        ax_textbox_vmax.set_visible(True)
        textbox_vmin.set_val(str(current_layer.vmin_pc))
        textbox_vmax.set_val(str(current_layer.vmax_pc))
    else:
        ax_textbox_vmin.set_visible(False)
        ax_textbox_vmax.set_visible(False)

    fig.canvas.draw_idle()



layer_manager.set_handle_spectral_sliders_textboxes(create_spectral_sliders_textboxes())
layer_manager.set_API_spectral_sliders_update(create_spectral_sliders_textboxes)
layer_manager.set_handle_radiobuttons_layers_list(create_radiobuttons_layers_list())
layer_manager.set_API_radiobuttons_layers_list_update(create_radiobuttons_layers_list)
layer_manager.set_API_widgets_update(update_widgets_visibility)
update_widgets_visibility()

# Show the plot with an interactive cursor
layer_manager.set_handle_cursor(Cursor(layer_manager.get_ax(), useblit=True, color='red', linewidth=1))
plt.get_current_fig_manager().window.showMaximized()    # NOTE: Only for Qt backend!!
plt.show()
