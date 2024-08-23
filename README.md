# FIBMO
`FIBMO` is the **F**ITS **I**nteractive **B**rowser for **M**ulti-wavelength **O**bservations, an in-development, light-weight Python alternative designed for exploring multi-wavelength data simultaneously.
The fundamental objective is to display an image or slice of a cube and overlay a number of images or slices with contours, whilst the user is able to navigate the image plane with the `matplotlib.pyplot` interface.
Additionally, small features are included for quality-of-life analysis of the data being browsed. There are opportunities for such features to be expanded upon, given the request.

> [!IMPORTANT]
> This is a very early development project. What you see should not be judged as complete nor professional.

## Features
- Intelligently reads and interprets FITS files (to some capacity)
- Organises data into `ImageLayers`, controlled by a `LayerManager`
  - Switch between layers to modify their representation
    - Set the vmin/vmax percentage of the background layer
    - Select a slice method and the slice section for collapsing 3D cubes
  - Projects non-background layers onto the WCS of the background layer
  - Get `SkyCoord` information upon clicking the left mouse button at a cursor position
    - Choose the output stream to write the logged `SkyCoord`
  - The plot is in an interactive `matplotlib.pyplot` figure, allowing for zooming and panning

## Requirements
- Python >= 3.10
- Astropy (to-check)
- Matplotlib (to-check)
- Numpy (to-check)
- tkinter/Tk (to-check)
- reproject (to-check)

## TODO
- [ ] Containerisation to ease dependencies/requirements
- [ ] Layer control: name, cmap, contour colour
- [ ] Update layers list display: text wrapping, indicate current background layer, indicate contour colour
- [ ] Toggle layer visibility on/off (button per layer)
- [ ] Get layer stats
- [ ] Function to copy the current view and perform a false-colour image generation, combining selected layers with specified colour values (opacity is split between N layers)
- [ ] Spectrum preview at last SkyCoord (for current layer if it has a spectral axis)
- [ ] Pixel value at last SkyCoord (opt. convert to a common unit?)
- [ ] Support for more matplotlib backends
- [ ] Optimise initialisation without any data
- [ ] Support for more wavelengths

## Known Issues
- [ ] WCS is handled incorrectly for some HST images (but only for overlaying contours). Further investigation required; please raise similar issues with other data.
