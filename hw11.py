import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import json

def load_calibration_params(filepath):
    """
    Carga los parámetros de calibración de la cámara desde un archivo JSON.

    Args:
        filepath (str): Ruta al archivo JSON que contiene los parámetros de calibración.

    Returns:
        dict: Parámetros de calibración cargados desde el archivo JSON.
    """
    with open(filepath, 'r') as file:
        params = json.load(file)
    return params

def compute_disparity(uL_c, uR_c):
    """
    Calcula la disparidad entre las coordenadas horizontales en las imágenes izquierda y derecha.

    Args:
        uL_c (float): Coordenada horizontal del punto en la imagen izquierda.
        uR_c (float): Coordenada horizontal del punto en la imagen derecha.

    Returns:
        float: Disparidad entre las coordenadas horizontales.
    """
    return uL_c - uR_c

def compute_coordinates(u, v, Z, f):
    """
    Calcula las coordenadas tridimensionales (X, Y, Z) de un punto en el espacio 3D.

    Args:
        u (float): Coordenada horizontal del punto en la imagen.
        v (float): Coordenada vertical del punto en la imagen.
        Z (float): Profundidad del punto en milímetros.
        f (float): Longitud focal de la cámara en píxeles.

    Returns:
        tuple: Coordenadas tridimensionales (X, Y, Z) del punto.
    """
    X = (u * Z) / f
    Y = (v * Z) / f
    return X, Y, Z

def on_mouse_click(event, x, y, flags, params):
    """
    Manejador de eventos para el clic del mouse en una ventana.

    Registra las coordenadas del punto seleccionado cuando se hace clic en la ventana.

    Args:
        event (int): Tipo de evento de OpenCV.
        x (int): Coordenada x del punto seleccionado.
        y (int): Coordenada y del punto seleccionado.
        flags (int): Indicadores de estado del mouse.
        params (dict): Parámetros adicionales pasados al manejador de eventos.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        params['coords'].append((x, y))
        print(f"Selected pixel at ({x}, {y})")

def select_pixels(image, window_name):
    """
    Selecciona píxeles de interés en una imagen utilizando el clic del mouse.

    Muestra la imagen en una ventana y permite al usuario seleccionar píxeles haciendo clic con el mouse.

    Args:
        image (numpy.ndarray): Imagen en la que se seleccionarán los píxeles.
        window_name (str): Nombre de la ventana que muestra la imagen.

    Returns:
        list: Lista de coordenadas de los píxeles seleccionados.
    """
    cv2.namedWindow(window_name)
    coords = {'coords': []}
    cv2.setMouseCallback(window_name, on_mouse_click, coords)
    print(f"Please select pixels in the {window_name} window. Press ESC to finish.")
    while True:
        cv2.imshow(window_name, image)
        if cv2.waitKey(1) & 0xFF == 27:  # Salir con la tecla ESC
            break
    cv2.destroyAllWindows()
    return coords['coords']

def main(left_image_path, right_image_path, calib_params_path):
    """
    Función principal para la reconstrucción 3D a partir de imágenes estéreo.

    Carga las imágenes, selecciona los píxeles de interés en las imágenes izquierda y derecha,
    calcula las coordenadas 3D de los puntos seleccionados y visualiza los resultados en un gráfico 3D.

    Args:
        left_image_path (str): Ruta a la imagen izquierda.
        right_image_path (str): Ruta a la imagen derecha.
        calib_params_path (str): Ruta al archivo de parámetros de calibración de la cámara.
    """
    # Cargar parámetros de calibración
    calib_params = load_calibration_params(calib_params_path)
    fx = float(calib_params['rectified_fx'])
    fy = float(calib_params['rectified_fy'])
    cx = float(calib_params['rectified_cx'])
    cy = float(calib_params['rectified_cy'])
    B = abs(float(calib_params['baseline']))  # Usar valor absoluto de la base

    # Cargar imágenes
    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    if left_image is None or right_image is None:
        raise FileNotFoundError("One or both images not found. Please check the file paths.")

    # Seleccionar píxeles en las imágenes
    left_coords = select_pixels(left_image, 'Left Image')
    right_coords = select_pixels(right_image, 'Right Image')

    if len(left_coords) != len(right_coords):
        raise ValueError("Number of selected points must be the same in both images")

    # Calcular coordenadas 3D
    points_3d = []
    for (uL, vL), (uR, vR) in zip(left_coords, right_coords):
        uL_c = uL - cx
        uR_c = uR - cx
        vL_c = vL - cy  # vR_c == vL_c porque las imágenes están rectificadas

        disparity = compute_disparity(uL_c, uR_c)
        if disparity == 0:
            print(f"Warning: Disparity is zero for points ({uL}, {vL}) and ({uR}, {vR}). Skipping.")
            continue
        Z = compute_depth(fx, B, disparity)
        X, Y, Z = compute_coordinates(uL_c, vL_c, Z, fx)

        points_3d.append((X, Y, Z))

    # Graficar puntos 3D
    if points_3d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        xs, ys, zs = zip(*points_3d)
        ax.scatter(xs, ys, zs)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.show()
    else:
        print("No 3D points to display.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sparse 3D Reconstruction using pre-captured images")
    parser.add_argument('--l_img', type=str, required=True, help='Path to the left image')
    parser.add_argument('--r_img', type=str, required=True, help='Path to the right image')
    parser.add_argument('--calib_params', type=str, required=True, help='Path to the calibration parameters file')
    args = parser.parse_args()

    main(args.l_img, args.r_img, args.calib_params)
