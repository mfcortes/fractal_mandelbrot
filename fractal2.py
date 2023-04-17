# construido por chatgtp4
import numpy as np
import matplotlib.pyplot as plt
from numba import cuda, float32
import math

# Definición de la función del fractal de Mandelbrot en CUDA
@cuda.jit(device=True)
def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

# Función para generar el fractal de Mandelbrot en paralelo
@cuda.jit
def mandelbrot_set_cuda(min_x, max_x, min_y, max_y, image, max_iter):
    height, width = image.shape

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    start_x, start_y = cuda.grid(2)
    grid_x = cuda.gridDim.x * cuda.blockDim.x
    grid_y = cuda.gridDim.y * cuda.blockDim.y

    for x in range(start_x, width, grid_x):
        real = min_x + x * pixel_size_x
        for y in range(start_y, height, grid_y):
            imag = min_y + y * pixel_size_y
            color = mandelbrot(complex(real, imag), max_iter)
            image[y, x] = color

def render_mandelbrot(min_x, max_x, min_y, max_y, width, height, max_iter):
    # Configuración de la GPU
    threads_per_block = (32, 32)
    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Crear la imagen y copiarla a la memoria del dispositivo
    image = np.empty((height, width), dtype=np.uint32)
    d_image = cuda.to_device(image)

    # Llamar a la función CUDA para generar el fractal
    mandelbrot_set_cuda[blocks_per_grid, threads_per_block](min_x, max_x, min_y, max_y, d_image, max_iter)

    # Copiar la imagen resultante de la memoria del dispositivo a la memoria del host
    d_image.copy_to_host(image)
    
    

    return image



def main():
    # Configuración inicial
    width, height = 800, 800
    max_iter = 2000

    min_x, max_x = -2, 1
    min_y, max_y = -1.5, 1.5

    plt.ion()
    fig, ax = plt.subplots()
    is_selecting = False
    eclick = None
    rect = plt.Rectangle((0, 0), 0, 0, edgecolor='red', facecolor='none')

    def on_click(event):
        nonlocal eclick, is_selecting
        eclick = event
        if event.inaxes == ax:
            is_selecting = True
            rect.set_width(0)
            rect.set_height(0)
            rect.set_xy((event.xdata, event.ydata))
            ax.add_patch(rect)

    def on_release(event):
        nonlocal eclick, is_selecting, min_x, max_x, min_y, max_y
        if event.inaxes == ax and is_selecting:
            is_selecting = False
            x0, y0 = eclick.xdata, eclick.ydata
            x1, y1 = event.xdata, event.ydata

            # Ordenar las coordenadas
            if x0 > x1:
                x0, x1 = x1, x0
            if y0 > y1:  # Ordenar las coordenadas en el eje Y sin invertir
                y0, y1 = y1, y0

            # Verificación de tolerancia para evitar transformaciones singulares
            if abs(x1 - x0) < 1e-6 or abs(y1 - y0) < 1e-6:
                return

            min_x, max_x = round(x0, 6), round(x1, 6)
            min_y, max_y = round(y0, 6), round(y1, 6)

            # Calcular la relación de aspecto de la selección
            selection_aspect_ratio = abs((max_y - min_y) / (max_x - min_x))

            # Ajustar el tamaño de la imagen en función de la relación de aspecto de la selección
            new_width = width
            new_height = round(width * selection_aspect_ratio)
            if new_height > new_width:
                new_width = round(height / selection_aspect_ratio)
                new_height = height

            image = render_mandelbrot(min_x, max_x, min_y, max_y, new_width, new_height, max_iter)
            im.set_data(image)
            im.set_extent((min_x, max_x, min_y, max_y))

            # Ajustar los límites del eje y mantener la relación de aspecto de la selección
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_aspect(selection_aspect_ratio)

            # Ajustar la relación de aspecto de la figura para que coincida con la de la selección
            fig_aspect_ratio = fig.get_figheight() / fig.get_figwidth()
            if selection_aspect_ratio > fig_aspect_ratio:
                fig.set_figheight(fig.get_figwidth() * selection_aspect_ratio)
            else:
                fig.set_figwidth(fig.get_figheight() / selection_aspect_ratio)

            plt.tight_layout()
            plt.draw()
            ax.patches.remove(rect)



    
    def on_motion(event):
        nonlocal eclick
        if is_selecting and event.inaxes == ax:
            x0, x1 = eclick.xdata, event.xdata
            y0, y1 = eclick.ydata, event.ydata
            rect.set_width(abs(x1 - x0))
            rect.set_height(abs(y1 - y0))
            rect.set_xy((min(x0, x1), min(y0, y1)))
            plt.draw()


    image = render_mandelbrot(min_x, max_x, min_y, max_y, width, height, max_iter)
    # im = ax.imshow(image, cmap='twilight_shifted', extent=(min_x, max_x, min_y, max_y))
    im = ax.imshow(image, cmap='twilight_shifted', extent=(min_x, max_x, min_y, max_y), origin='lower')


    print(f'Mostrando el fractal de Mandelbrot en el rango x: ({min_x}, {max_x}), y: ({min_y}, {max_y})')

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    plt.show(block=True)

if __name__ == "__main__":
    main()