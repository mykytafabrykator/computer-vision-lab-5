import cv2
import numpy as np
from skimage import feature, morphology
from skimage.segmentation import watershed
from scipy.ndimage import distance_transform_edt
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb


# Завдання 1: Виявлення точок
def detect_points(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    threshold = np.max(np.abs(laplacian)) * 0.5
    points = np.abs(laplacian) >= threshold
    cv2.imwrite(output_path, (points * 255).astype(np.uint8))


# Завдання 2: Виявлення ліній
def detect_lines(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    kernels = [
        np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]),  # Горизонтальна
        np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]),  # Вертикальна
    ]
    responses = [cv2.filter2D(image, -1, kernel) for kernel in kernels]
    combined_response = np.max(responses, axis=0)
    cv2.imwrite(output_path, combined_response)


# Завдання 3: Виявлення перепадів
def detect_edges(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    edges = feature.canny(image)
    cv2.imwrite(output_path, (edges * 255).astype(np.uint8))


# Завдання 4: Глобальний поріг
def global_threshold(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    T_prev = np.mean(image)
    T_curr = 0
    threshold_diff = 1e-3  # Критерій зупинки

    while abs(T_prev - T_curr) > threshold_diff:
        if T_curr != 0:
            T_prev = T_curr

        G1 = image[image >= T_prev]
        G2 = image[image < T_prev]

        T_curr = 0.5 * (np.mean(G1) + np.mean(G2))

    _, thresholded = cv2.threshold(image, T_curr, 255, cv2.THRESH_BINARY)

    cv2.imwrite(output_path, thresholded)


# Завдання 5: Вододіли (перетворення відстані)
def watershed_distance(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    distance = distance_transform_edt(binary)
    markers = morphology.label(distance > 20)
    segmented = watershed(-distance, markers, mask=binary)
    cv2.imwrite(output_path, (segmented > 0).astype(np.uint8) * 255)


# Завдання 6: Вододіли (градієнт)
def watershed_gradient(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    gradient = cv2.morphologyEx(
        image, cv2.MORPH_GRADIENT,
        np.ones((3, 3), np.uint8)
    )
    _, binary = cv2.threshold(
        gradient, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    markers = morphology.label(binary)
    segmented = watershed(-gradient, markers, mask=binary)
    cv2.imwrite(output_path, (segmented > 0).astype(np.uint8) * 255)


# Завдання 7: K-середні
def kmeans_segmentation(image_path, output_path, n_clusters=5):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab_image = rgb2lab(image)
    pixels = lab_image.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    clustered_image = kmeans.cluster_centers_[labels].reshape(lab_image.shape)
    clustered_image = lab2rgb(clustered_image)
    clustered_image = (clustered_image * 255).astype(np.uint8)
    cv2.imwrite(output_path, cv2.cvtColor(clustered_image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    detect_points("input/pic.1.tif",
                  "result_images/detect_points.jpg")
    detect_lines("input/pic.2.tif",
                 "result_images/detect_lines.jpg")
    detect_edges("input/pic.3.tif",
                 "result_images/detect_edges.jpg")
    global_threshold("input/pic.4.tif",
                     "result_images/global_threshold.jpg")
    watershed_distance("input/pic.5.tif",
                       "result_images/watershed_distance.jpg")
    watershed_gradient("input/pic.6.tif",
                       "result_images/watershed_gradient.jpg")
    kmeans_segmentation("input/pic.8.jpg",
                        "result_images/kmeans_segmentation.jpg")
