import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from matplotlib import get_backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Cursor, TextBox, Button, CheckButtons, RadioButtons
from matplotlib.contour import QuadContourSet
import tkinter
from tkinter import filedialog
import argparse
import sys
from os.path import isfile

# TODO: Create LayerManager class, a collection of ImageLayer instances, to control which data is
#       the background image, foreground contours, swapping them out, deleting, adding,
#       the slice method of each, the vminmax of each, etc.

class ImageLayer:
    ##### CONSTRUCTORS #####
    def __init__(self, fits_path: str, hdu: fits.ImageHDU, name: str=None, image_type: str='auto', is_bkg_layer: bool=True,
                 vmin_pc: float=None, vmax_pc: float=None, levels_factor: float=3, levels_base: float=np.sqrt(2),
                 slice_type: str='none', plot_args: dict={'cmap': 'inferno', 'colors': 'lime', 'linewidths': 2, 'alpha': 0.7}):
        self.fits_path = fits_path
        self.hdu = hdu
        self.data = np.squeeze(hdu.data)
        self.wcs = WCS(hdu.header, naxis=2)
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
        self.vmin = np.nanpercentile(self.data, self.vmin_pc)
        self.vmax = np.nanpercentile(self.data, self.vmax_pc)
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
            self.data_flat = self.slice_func(self)
        else:
            self.data_flat = self.data
        self.levels_factor = levels_factor
        self.levels_base = levels_base
        self.artist = None
        self.plot_args = plot_args

    ##### DECONSTRUCTOR #####
    def __del__(self):
        # clean up artists, hdul, etc.
        self.hdu.close()
        self.artist.remove()
        del self

    ##### INTERNAL METHODS #####
    def slice_func_none(self):
        self.data_flat = np.nanmean(self.data, axis=0)

    def slice_func_manual(self):
        start_idx = min(self.slice_arg1, self.slice_arg2)
        end_idx = max(self.slice_arg1, self.slice_arg2)
        start_idx = max(0, start_idx)
        end_idx = min(self.spectral_axis_len - 1, end_idx)
        self.data_flat = np.nanmean(self.data[start_idx:end_idx+1, :, :], axis=0)

    def slice_func_window(self):
        central_idx = self.slice_arg1
        width = self.slice_arg2
        start_idx = max(0, int(central_idx - width // 2))
        end_idx = min(self.spectral_axis_len - 1, int(central_idx + width // 2))
        self.data_flat = np.nanmean(self.data[start_idx:end_idx+1, :, :], axis=0)

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


class LayerManager:
    ##### CONSTRUCTORS #####
    def __init__(self, fig: plt.Figure, ax):
        self.layers = [ImageLayer]
        self.current_layer = None
        self.background_layer = None
        self.fig = fig
        self.ax = ax

    ##### DECONSTRUCTORS #####
    def __del__(self):
        for layer in self.layers:
            layer.hdu.close()
        del self
    
    ##### INTERNAL METHODS #####


    ##### SET/GET #####
    def set_current_layer(self, layer, do_update: bool=True):
        self.current_layer = layer
        # Reflect change in current layer by updating sliders, vmin/vmax, contour levels, etc.
        if do_update:
            self.update_plot()
    
    def get_current_layer(self):
        return self.current_layer

    ##### PLOT METHODS #####
    def update_plot(self):
        if self.layers:
            for layer in self.layers:
                layer.plot_layer(self.ax)
        plt.draw()
        self.fig.canvas.draw_idle()

    ##### PUBLIC METHODS #####
    def add_layer(self, fits_path: str, hdu: fits.ImageHDU, name: str=None, is_bkg_layer: bool=True, vmin_pc: float=None,
                  vmax_pc: float=None, levels_factor: float=3, levels_base: float=np.sqrt(2), slice_type: str='none',
                  plot_args: dict={'cmap': 'inferno', 'colors': 'lime', 'linewidths': 2, 'alpha': 0.7}):
        if self.layers == []:
            is_bkg_layer = True
        layer = ImageLayer(fits_path, hdu, name, 'auto', is_bkg_layer, vmin_pc, vmax_pc, levels_factor, levels_base,
                           slice_type, plot_args)
        self.layers.append(layer)
        self.set_current_layer(layer, do_update=False)
        if self.current_layer.is_bkg_layer:
            self.current_layer.set_is_bkg_layer(False)
            self.send_layer_to_background(do_update=False)
        self.update_plot()

    def remove_layer(self, layer):
        if layer in self.layers:
            self.layers.remove(layer)
            if self.current_layer == layer:
                self.current_layer = self.layers[0] if self.layers else None
            del layer
            self.update_plot()
        else:
            raise UserWarning(f"Could not remove layer {layer.name}")

    def send_layer_to_background(self, do_update: bool=True):
        if self.current_layer and self.current_layer.is_bkg_layer == False:
            self.current_layer.is_bkg_layer = True
            if self.background_layer is not None:
                self.background_layer.is_bkg_layer = False
            self.background_layer = self.current_layer
            # Change axes WCS by removing and re-creating axes with new projection
            self.ax.remove()
            self.ax = self.fig.add_subplot(projection=self.background_layer.wcs)
            self.ax.set_xlabel('RA (J2000)')
            self.ax.set_ylabel('Dec. (J2000)',labelpad=-0.5)
            self.ax.coords[0].set_ticklabel(exclude_overlapping=True)
            cursor = Cursor(self.ax, useblit=True, color='red', linewidth=1)
            # Reflect change in background layer by updating sliders, vmin/vmax, axes WCS, etc.
            if do_update:
                self.update_plot()
        else:
            raise UserWarning(f"Could not change background layer {self.background_layer.name} to {self.current_layer.name}")
        


    



    

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

def open_file_dialog():
    root = tkinter.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select a file")
    root.destroy()  # Close the Tkinter root window
    return file_path

def save_file_dialog():
    root = tkinter.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(title="Select output file")
    root.destroy()
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

def switch_data(event):
    global wcs, wcs_in, hdu_in, data_in, im, ax, fig, cursor

    new_data_file = open_file_dialog()
    if new_data_file == ():
        dummy = True
    else:
        # Load the FITS file and data
        hdu_in = fits.open(new_data_file)[0]
        data_in = np.squeeze(hdu_in.data) # REMOVE ANY DEGENERATE STOKES AXIS
        wcs_in = WCS(hdu_in.header, naxis=2)
        print(f"Input FITS data has shape {data_in.shape}")
        im.remove()
        if fits_file_bkg != '':
            bkg_im.remove()

        if fits_file_bkg != '':
            wcs = wcs_bkg
        else:
            wcs = wcs_in
        
        ax.remove()
        ax = fig.add_subplot(projection=wcs)
        if fits_file_bkg != '':
            bkg_im = ax.imshow(data_bkg, origin='lower', cmap='inferno',
                            vmin=np.nanpercentile(data_bkg, vmin), vmax=np.nanpercentile(data_bkg, vmax))
            im = ax.contour(prepare_flat_array(arg1, arg2), levels=np.nanstd(prepare_flat_array(arg1, arg2))*3*np.sqrt(2)**np.arange(0,3),
                            alpha=0.7, linewidths=2, negative_linestyles='dashed', colors='lime', transform=ax.get_transform(wcs_in))
        else:
            data_slice = prepare_flat_array(arg1, arg2)
            im = ax.imshow(data_slice, origin='lower', cmap='inferno',
                        vmin=np.nanpercentile(data_slice, vmin), vmax=np.nanpercentile(data_slice, vmax))

        ax.set_xlabel('RA (J2000)')
        ax.set_ylabel('Dec. (J2000)',labelpad=-0.5)
        ax.coords[0].set_ticklabel(exclude_overlapping=True)

        cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
    fig.canvas.draw_idle()
    update(None)


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

# Load the FITS file and data
hdu_in = fits.open(fits_file)[0]
data_in = np.squeeze(hdu_in.data) # REMOVE ANY DEGENERATE STOKES AXIS
wcs_in = WCS(hdu_in.header, naxis=2)

# [Optional] Load the FITS file and data for background image
if fits_file_bkg != '':
    hdu_bkg = fits.open(fits_file_bkg)[0]
    data_bkg = np.squeeze(hdu_bkg.data)
    wcs_bkg = WCS(hdu_bkg.header, naxis=2)
    assert data_bkg.ndim == 2

if interactive_type == 'auto':
    if hdu_in != None and hdu_in.header['NAXIS'] > 2:
        interactive_type = 'cube'
    elif hdu_in != None and hdu_in.header['NAXIS'] == 2:
        interactive_type = 'image'
    else:
        raise ValueError(f"Unsupported number of axes in data: {hdu_in.header['NAXIS']}")

if interactive_type == 'cube':
    # Ensure the data cube has 3 dimensions
    if data_in.ndim < 3:
        raise ValueError(f"The data must be a 3D cube (data had {data_in.ndim} dimensions).")
elif interactive_type == 'image':
    if data_in.ndim != 2:
        raise ValueError(f"The data must be a 2D image (data had {data_in.ndim} dimensions).")

print(f"Input FITS data has shape {data_in.shape}")

# Set up initial slice
arg1, arg2 = 0, 0
if interactive_type == 'cube':
    spectral_axis_len = data_in.shape[0]
    if interaction_method == 'manual':
        arg1, arg2 = 0, spectral_axis_len - 1
    elif interaction_method == 'window':
        arg1 = spectral_axis_len // 2
        arg2 = spectral_axis_len // 1  # For example, a width of 100% of the spectral axis length

# Function to compute the averaged slice
if interactive_type == 'cube':
    if interaction_method == 'manual':
        def prepare_flat_array(arg1, arg2):
            '''
            Manual spectral range averaging of cube. `arg1` is the lower index and `arg2` is the upper index.
            '''
            start_idx = min(arg1, arg2)
            end_idx = max(arg1, arg2)
            return np.nanmean(data_in[start_idx:end_idx+1, :, :], axis=0)
    elif interaction_method == 'window':
        def prepare_flat_array(arg1, arg2):
            '''
            Spectral window-averaging of cube. `arg1` is the central index and `arg2` is the window width.
            '''
            central_idx = arg1
            width = arg2
            start_idx = max(0, int(central_idx - width // 2))
            end_idx = min(spectral_axis_len - 1, int(central_idx + width // 2))
            return np.nanmean(data_in[start_idx:end_idx+1, :, :], axis=0)
    elif interaction_method == 'none':
        def prepare_flat_array(arg1, arg2):
            '''
            Dummy function. `arg1`, `arg2` do not matter. Returns average of cube (spectrally).
            '''
            return np.nanmean(data_in, axis=0)
else:
    def prepare_flat_array(arg1, arg2):
        '''
        Dummy function. `arg1`, `arg2` do not matter. Returns the already 2D image array.
        '''
        return data_in

# Create the figure and axis
if fits_file_bkg != '':
    wcs = wcs_bkg
else:
    wcs = wcs_in
fig, ax = plt.subplots(subplot_kw={'projection': wcs})
plt.subplots_adjust(left=0.1, bottom=0.25)  # Adjust figure to accommodate sliders

# Plot the initial averaged slice
im = None
last_skycoord = None  # Variable to store the last clicked SkyCoord

if fits_file_bkg != '':
    bkg_im = ax.imshow(data_bkg, origin='lower', cmap='inferno',
                    vmin=np.nanpercentile(data_bkg, vmin), vmax=np.nanpercentile(data_bkg, vmax))
    im = ax.contour(prepare_flat_array(arg1, arg2), levels=np.nanstd(prepare_flat_array(arg1, arg2))*3*np.sqrt(2)**np.arange(0,3),
                    alpha=0.7, linewidths=2, negative_linestyles='dashed', colors='lime', transform=ax.get_transform(wcs_in))
else:
    data_slice = prepare_flat_array(arg1, arg2)
    im = ax.imshow(data_slice, origin='lower', cmap='inferno',
                   vmin=np.nanpercentile(data_slice, vmin), vmax=np.nanpercentile(data_slice, vmax))
    # Flip RA to be increasing to the left if it is not
    ra_left = ax.wcs.pixel_to_world(0,0)
    ra_right = ax.wcs.pixel_to_world(10,0)
    if ra_left.ra < ra_right.ra:
        ax.invert_xaxis()

ax.set_xlabel('RA (J2000)')
ax.set_ylabel('Dec. (J2000)',labelpad=-0.5)
ax.coords[0].set_ticklabel(exclude_overlapping=True)

if interactive_type == 'cube' and interaction_method != 'none':
    if interaction_method == 'manual':
        # Add sliders for start and end indices
        ax_slider_1 = plt.axes([0.25, 0.1, 0.4, 0.03], facecolor='lightgoldenrodyellow')
        ax_slider_2 = plt.axes([0.25, 0.15, 0.4, 0.03], facecolor='lightgoldenrodyellow')

        slider_1 = Slider(ax_slider_1, 'Start Channel', 0, spectral_axis_len - 1, valinit=arg1, valstep=1)
        slider_2 = Slider(ax_slider_2, 'End Channel', 0, spectral_axis_len - 1, valinit=arg2, valstep=1)
    elif interaction_method == 'window':
        # Add sliders for central index and width
        ax_slider_1 = plt.axes([0.25, 0.1, 0.4, 0.03], facecolor='lightgoldenrodyellow')
        ax_slider_2 = plt.axes([0.25, 0.15, 0.4, 0.03], facecolor='lightgoldenrodyellow')

        slider_1 = Slider(ax_slider_1, 'Central Channel', 0, spectral_axis_len - 1, valinit=arg1, valstep=1)
        slider_2 = Slider(ax_slider_2, 'Window Width', 1, spectral_axis_len, valinit=arg2, valstep=1)

    # Add text boxes for direct input of slider values
    ax_textbox_1 = plt.axes([0.72, 0.1, 0.1, 0.03], facecolor='lightgoldenrodyellow')
    ax_textbox_2 = plt.axes([0.72, 0.15, 0.1, 0.03], facecolor='lightgoldenrodyellow')

    textbox_1 = TextBox(ax_textbox_1, '', initial=str(arg1))
    textbox_2 = TextBox(ax_textbox_2, '', initial=str(arg2))

    # Function to update the plot when sliders are changed
    def update(val):
        global im

        textbox_1.set_val(int(slider_1.val))
        textbox_2.set_val(int(slider_2.val))

        vmin = float(textbox_vmin.text)
        vmax = float(textbox_vmax.text)

        arg1 = int(slider_1.val)
        arg2 = int(slider_2.val)

        data_slice = prepare_flat_array(arg1, arg2)

        # Update the image data
        if fits_file_bkg == '':
            im.set_data(data_slice)
            im.set_clim(vmin=np.nanpercentile(data_slice, vmin), vmax=np.nanpercentile(data_slice, vmax))
        else:   # or contours if image data is background image
            im.remove()
            im = ax.contour(data_slice, levels=np.nanstd(data_slice)*3*np.sqrt(2)**np.arange(0,3),
                    alpha=0.7, linewidths=2, negative_linestyles='dashed', colors='lime', transform=ax.get_transform(wcs_in))
        fig.canvas.draw_idle()
    
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

    # Connect the sliders to the update function
    slider_1.on_changed(update)
    slider_2.on_changed(update)

# Add text boxes for vmin and vmax
ax_textbox_vmin = plt.axes([0.72, 0.5, 0.1, 0.03], facecolor='lightgoldenrodyellow')
ax_textbox_vmax = plt.axes([0.72, 0.55, 0.1, 0.03], facecolor='lightgoldenrodyellow')

textbox_vmin = TextBox(ax_textbox_vmin, 'vmin', initial=str(vmin))
textbox_vmax = TextBox(ax_textbox_vmax, 'vmax', initial=str(vmax))

def update_vmin(text):
    try:
        vmin = float(text)
        update(None)
    except ValueError:
        pass

def update_vmax(text):
    try:
        vmax = float(text)
        update(None)
    except ValueError:
        pass

textbox_vmin.on_submit(update_vmin)
textbox_vmax.on_submit(update_vmax)

# Option to select output file via button + Tkinter prompt
ax_button_out_file = plt.axes([0.72, 0.65, 0.15, 0.03])
button_out_file = Button(ax_button_out_file, 'Change output stream')
button_out_file.on_clicked(switch_outstream)

# Option to select new data from file
ax_button_swap_data = plt.axes([0.72, 0.45, 0.15, 0.03])
button_swap_data = Button(ax_button_swap_data, 'Swap FITS data')
button_swap_data.on_clicked(switch_data)

# Add a TextBox to display the last-clicked coordinate
ax_textbox_coords = plt.axes([0.72, 0.85, 0.15, 0.04], facecolor='lightgoldenrodyellow')
textbox_coords = TextBox(ax_textbox_coords, 'SkyCoord:', initial="No coord selected")
ax_textbox_coords_deg = plt.axes([0.74, 0.80, 0.13, 0.04], facecolor='lightgoldenrodyellow')
textbox_coords_deg = TextBox(ax_textbox_coords_deg, 'SkyCoord (deg):', initial="No coord selected")

# Function to print world coordinates on mouse click
def onclick(event):
    global last_skycoord

    if event.inaxes == ax and event.button == 1:  # Left mouse button
        x, y = event.xdata, event.ydata
        # Convert pixel coordinates to world coordinates
        world_coords = wcs.pixel_to_world(x, y)
        last_skycoord = world_coords
        # Update the TextBox with the last clicked coordinate
        textbox_coords.set_val(f"{last_skycoord.to_string(style='hmsdms',precision=3)}")
        textbox_coords_deg.set_val(f"{last_skycoord.to_string(style='decimal',precision=5)}")

# Connect the onclick event to the figure
fig.canvas.mpl_connect('button_press_event', onclick)

def save_to_file(event):
    if last_skycoord is not None:
        _ = out_stream.write(f"{last_skycoord.to_string(style='decimal',precision=5).replace(' ',',')}\n")
        out_stream.flush()

ax_button_save = plt.axes([0.72, 0.7, 0.15, 0.03])
button_save = Button(ax_button_save, 'Save Coord')
button_save.on_clicked(save_to_file)

# Show the plot with an interactive cursor
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
plt.get_current_fig_manager().window.showMaximized()    # NOTE: Only for Qt backend!!
plt.show()
