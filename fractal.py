import numpy as np
import matplotlib.pyplot as plt


def mandelbrot(c):
    z = 0
    n = 0
    while abs(z) <= 2 and n < 1000:
        z = z*z + c
        n += 1
    if n == 1000:
        return 0
    else:
        return n

xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
fig, ax = plt.subplots()
x, y = np.meshgrid(np.linspace(xmin, xmax, 1000), np.linspace(ymin, ymax, 1000))
c = x + y*1j
mandelbrot_set = np.array([mandelbrot(i) for i in c.flat]).reshape(x.shape)

im = ax.imshow(mandelbrot_set.T, cmap='hot', extent=[xmin, xmax, ymin, ymax])

def onclick(event):
    global xmin, xmax, ymin, ymax
    xcenter = (event.xdata + xmin)/2
    ycenter = (event.ydata + ymin)/2
    xwidth = (xmax - xmin) / (event.inaxes.get_xlim()[1] - event.inaxes.get_xlim()[0])
    ywidth = (ymax - ymin) / (event.inaxes.get_ylim()[1] - event.inaxes.get_ylim()[0])
    xmin = xcenter - xwidth/2
    xmax = xcenter + xwidth/2
    ymin = ycenter - ywidth/2
    ymax = ycenter + ywidth/2

    x, y = np.meshgrid(np.linspace(xmin, xmax, 2000), np.linspace(ymin, ymax, 2000))
    c = x + y*1j
    mandelbrot_set = np.array([mandelbrot(i) for i in c.flat]).reshape(x.shape)

    im.set_data(mandelbrot_set.T)
    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    fig.canvas.draw()

cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()