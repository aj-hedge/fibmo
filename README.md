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

## TODO
- [ ] Optimise initialisation without any data
- [ ] Support for more wavelengths
