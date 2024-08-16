import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Cursor, TextBox, Button, CheckButtons
import tkinter
import argparse
import sys
from os.path import isfile

parser = argparse.ArgumentParser()
parser.add_argument("path_to_fits_file", type=str, help="The full file path to the FITS file to use.")
parser.add_argument("-t", "--type", type=str, choices=['image', 'cube'], required=True,
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
    file_path = tkinter.filedialog.askopenfilename(title="Select a file")
    root.destroy()  # Close the Tkinter root window
    return file_path

# Assign args
interactive_type = args.type
interaction_method = args.slice_method
vmin, vmax = args.vminmax
fits_file = args.path_to_fits_file
fits_file_bkg = args.background
if args.output_file == sys.stdout:
    out_stream = sys.stdout
else:
    out_stream = open(args.output_file)

# Clean args
if not isfile(fits_file):
    raise FileNotFoundError(f"{fits_file} does not exist.")
if fits_file_bkg != '' and not isfile(fits_file_bkg):
    raise FileNotFoundError(f"{fits_file_bkg} does not exist.")

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

# Add a TextBox to display the last-clicked coordinate
ax_textbox_coords = plt.axes([0.72, 0.7, 0.2, 0.05], facecolor='lightgoldenrodyellow')
textbox_coords = TextBox(ax_textbox_coords, 'SkyCoord:', initial="No coord selected", readonly=True)

# Function to print world coordinates on mouse click
def onclick(event):
    global last_skycoord

    if event.inaxes == ax and event.button == 1:  # Left mouse button
        x, y = event.xdata, event.ydata
        # Convert pixel coordinates to world coordinates
        world_coords = wcs.pixel_to_world(x, y)
        last_skycoord = world_coords
        # Update the TextBox with the last clicked coordinate
        textbox_coords.set_val(f"{world_coords}")

# Connect the onclick event to the figure
fig.canvas.mpl_connect('button_press_event', onclick)

def save_to_file(event):
    if last_skycoord is not None:
        _ = out_stream.write(f"{last_skycoord}\n")

ax_button_save = plt.axes([0.72, 0.6, 0.2, 0.05])
button_save = Button(ax_button_save, 'Save Coord')
button_save.on_clicked(save_to_file)

# Show the plot with an interactive cursor
cursor = Cursor(ax, useblit=True, color='red', linewidth=1)
plt.get_current_fig_manager().window.showMaximized()    # NOTE: Only for Qt backend!!
plt.show()
