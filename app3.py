import os
import shutil
import numpy as np
import cv2
import sqlite3

from pathlib import Path
from flask import Flask, render_template, request

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- PATH SETUP ---
BASE_DIR = Path(__file__).resolve().parent
TEST_DIR = BASE_DIR / "test"
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = STATIC_DIR / "images"
MODEL_PATH = BASE_DIR / "Convolutional_Neural_Network.h5"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)  # make sure static/images exists

# Flask app
app = Flask(__name__)

# ---------------- AUTH + BASIC PAGES ---------------- #

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = ? AND password = ?"
        cursor.execute(query, (name, password))
        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided, Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)
        cursor.execute("INSERT INTO user VALUES (?, ?, ?, ?)", (name, password, mobile, email))
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')

    return render_template('index.html')


@app.route('/userlog.html')
def userlogg():
    return render_template('userlog.html')


@app.route('/developer.html')
def developer():
    return render_template('developer.html')


# ---------------- IMAGE UPLOAD + PROCESSING ---------------- #

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
        # Clear old uploaded images
        for file in UPLOAD_DIR.glob("*"):
            file.unlink()

        fileName = request.form['filename']
        src_path = TEST_DIR / fileName
        dst_path = UPLOAD_DIR / fileName

        if not src_path.exists():
            return render_template('userlog.html', msg=f"File {fileName} not found in test folder.")

        # Copy selected image to static/images
        shutil.copy(src_path, dst_path)

        image = cv2.imread(str(src_path))

        # ---------- 1. IMAGE PREPROCESSING STEPS ---------- #

        # Grayscale conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Luminosity grayscale
        luminosity_gray = 0.21 * image[:, :, 2] + 0.72 * image[:, :, 1] + 0.07 * image[:, :, 0]
        cv2.imwrite(str(STATIC_DIR / 'luminosity_gray.jpg'), luminosity_gray)

        # Median filter
        median_filtered = cv2.medianBlur(luminosity_gray.astype("uint8"), 5)
        cv2.imwrite(str(STATIC_DIR / 'median_filtered.jpg'), median_filtered)

        # Unsharp masking
        gaussian = cv2.GaussianBlur(median_filtered, (9, 9), 10.0)
        unsharp_image = cv2.addWeighted(gray_image, 1.5, gaussian, -0.5, 0)
        cv2.imwrite(str(STATIC_DIR / 'unsharp_masked.jpg'), unsharp_image)

        # Thresholding
        _, gaussian_threshold = cv2.threshold(unsharp_image, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(str(STATIC_DIR / 'gaussian_thresholded.jpg'), gaussian_threshold)

        # Edge-based segmentation (Canny)
        edges = cv2.Canny(gaussian_threshold, 100, 200)
        cv2.imwrite(str(STATIC_DIR / 'edge_based_segmentation.jpg'), edges)

        # ---------- 2. GRADING USING SHAPE FEATURES (NO COUNTS) ---------- #

        gray_for_contour = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        blurred = cv2.GaussianBlur(gray_for_contour, (5, 5), 0)
        _, binary_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        total_regions = 0
        grade_score = 0  # 1 = low, 2 = intermediate, 3 = high

        for contour in contours:
            area = cv2.contourArea(contour)
            if area <= 0:
                continue

            perimeter = cv2.arcLength(contour, True)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)

            circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
            solidity = area / hull_area if hull_area > 0 else 0

            # Local grade based on shape and compactness
            if solidity > 0.8 and circularity > 0.7:
                local_grade = 1  # Well-organized
            elif 0.5 < solidity <= 0.8 and 0.5 < circularity <= 0.7:
                local_grade = 2  # Moderately organized
            else:
                local_grade = 3  # Disorganized / solid growth

            grade_score += local_grade
            total_regions += 1

        if total_regions > 0:
            average_grade_score = grade_score / total_regions
            if average_grade_score < 1.5:
                grade = "Low Grade (Well-Organized)"
            elif average_grade_score < 2.5:
                grade = "Intermediate Grade (Moderately Organized)"
            else:
                grade = "High Grade (Disorganized/Solid Growth)"
        else:
            grade = "Grade cannot be determined (no valid regions detected)"

        # NOTE: We do not save or display any separate contour image here.

        # ---------- 3. CNN PREDICTION + STAGE TEXT ---------- #

        model = load_model(str(MODEL_PATH))
        path = str(dst_path)

        def prepare_test_image(path):
            img = load_img(path, target_size=(128, 128), color_mode="grayscale")
            x = img_to_array(img)
            x = x / 255.0
            return np.expand_dims(x, axis=0)

        result = model.predict(prepare_test_image(path))
        class_result = np.argmax(result, axis=1)

        if class_result[0] == 0:
            str_label = "Endometrial Adenocarcinoma"
            stage = "Stage 3 - Malignant cancer detected"
        elif class_result[0] == 1:
            str_label = "Endometrial Hyperplasia"
            stage = "Stage 2 - Pre-cancer condition"
        elif class_result[0] == 2:
            str_label = "Endometrial Polyp"
            stage = "Stage 1 - Benign lesion"
        else:
            str_label = "Normal Endometrium"
            stage = "No stage (normal tissue)"

        accuracy = (
            f"The predicted image of {str_label} is with an accuracy of "
            f"{result[0][class_result[0]]*100:.2f}%"
        )

        # ---------- 4. GRAPH OF CLASS PROBABILITIES ---------- #

        dic = {
            "EA": float(result[0][0]),
            "EH": float(result[0][1]),
            "EP": float(result[0][2]),
            "NE": float(result[0][3]),
        }

        fig = plt.figure(figsize=(5, 5))
        plt.bar(dic.keys(), dic.values(), color="maroon", width=0.3)
        plt.xlabel("Comparison")
        plt.ylabel("Accuracy Level")
        plt.title("Accuracy Comparison between Endometrium Classes")
        plt.savefig(STATIC_DIR / "matrix.png")
        plt.close(fig)

        # ---------- 5. IMAGES TO DISPLAY (NO CONTOURS IMAGE) ---------- #

        ImageDisplay = [
            f"/static/images/{fileName}",      # Original
            "/static/luminosity_gray.jpg",     # Gray
            "/static/median_filtered.jpg",     # Median filter
            "/static/unsharp_masked.jpg",      # Unsharp
            "/static/gaussian_thresholded.jpg",# Threshold
            "/static/edge_based_segmentation.jpg",  # Canny edges
            "/static/matrix.png",              # Graph
        ]

        labels = [
            "Original",
            "Gray Image",
            "Median Filter",
            "Unsharp Masking",
            "Gaussian Threshold",
            "Edge Based (Canny)",
            "Graph",
        ]

        return render_template(
            "results.html",
            labels=labels,
            status=str_label,
            accuracy=accuracy,
            grade=grade,
            stage=stage,
            ImageDisplay=ImageDisplay,
            n=len(ImageDisplay),
        )

    return render_template("userlog.html")


@app.route("/logout")
def logout():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
