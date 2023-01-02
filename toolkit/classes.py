import os
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import cv2 as cv

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

plt.rcParams.update({"font.size": 16})


class Environment:
    def __init__(
        self,
        raw_folder="./raw/",
        data_folder="./data/",
        preprocessed_folder="./preprocessed/",
        augmented_folder="./augmented/",
        plot_folder="./plots/",
        PATH=None,
    ):
        if PATH is None:
            PATH = [
                "Train/Non defective/",
                "Train/Defective/",
                "Validation/Non defective/",
                "Validation/Defective/",
                "Test/Non defective/",
                "Test/Defective/",
            ]

        self.raw_folder = raw_folder
        self.data_folder = data_folder
        self.preprocessed_folder = preprocessed_folder
        self.augmented_folder = augmented_folder
        self.plot_folder = plot_folder

        self.RAW_PATH = [os.path.join(self.raw_folder, p) for p in PATH]
        self.DATA_PATH = [os.path.join(self.data_folder, p) for p in PATH]
        self.PREPROCESSED_PATH = [
            os.path.join(self.preprocessed_folder, p) for p in PATH
        ]
        self.AUGMENTED_PATH = [os.path.join(self.augmented_folder, p) for p in PATH[:2]]

        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)

        for p in self.DATA_PATH:
            if not os.path.exists(p):
                os.makedirs(p)

        for p in self.PREPROCESSED_PATH:
            if not os.path.exists(p):
                os.makedirs(p)

        for p in self.AUGMENTED_PATH:
            if not os.path.exists(p):
                os.makedirs(p)

    @staticmethod
    def __del__():
        gc.collect()


class BaseClass:
    @staticmethod
    def __del__():
        gc.collect()

    @staticmethod
    def remove_old_data(folder):
        for root, _, files in os.walk(folder, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))


class ImageProcessor(BaseClass):
    def __init__(self, steps, kwargs, plot_kwargs=None, save_folder="./preprocessed/"):
        if plot_kwargs is None:
            plot_kwargs = {}
        self.df = None
        self.steps = []
        self.pipeline_steps = steps
        self.kwargs = kwargs
        self.plot_kwargs = plot_kwargs
        self.save_folder = save_folder
        self.defect_class = None
        self.img_path = None
        self.imgs = []
        self.lines = []
        self.kp = []
        self.des = []
        self.function = {
            "original": self.load_sample,
            "crop": self.crop,
            "resize": self.resize,
            "grayscale": self.grayscale,
            "binary": self.binary,
            "hist_eq": self.hist_eq,
            "noise_filt": self.noise_filt,
            "stretching": self.stretching,
            "Laplacian": self.Laplacian,
            "Canny": self.Canny,
            "Hough": self.Hough,
            "feat_detect": self.feat_detect,
            "show": self.show,
            "save": self.save_image,
        }

    def load_sample(self, df_entry):
        del self.steps
        del self.imgs
        del self.lines
        del self.kp
        del self.des
        gc.collect()
        self.steps = ["original"]
        self.lines = []
        self.kp = []
        self.des = []
        self.imgs = [cv.imread(df_entry.img)]
        self.defect_class = df_entry.defect_str
        self.img_path = df_entry.img

    def crop(self):
        ratio = self.kwargs.get("crop").get("ratio")
        h, w = self.imgs[-1].shape[:2]
        pad_h = int(ratio * h)
        pad_w = int(ratio * w)
        self.steps.append("crop")
        self.imgs.append(self.imgs[-1][pad_h:-pad_h, pad_w:-pad_w, :])

    def resize(self):
        h, w = self.imgs[-1].shape[:2]
        new_h = self.kwargs.get("resize").get("height")
        new_w = self.kwargs.get("resize").get("width")
        self.steps.append("resize")
        if w < h:
            nh = int(h / w * new_h)
            nw = new_w
            self.imgs.append(
                cv.resize(self.imgs[-1], (nw, nh), interpolation=cv.INTER_AREA)
            )
            pad = int((nh - new_h) / 2)
            if pad > 0:
                self.imgs[-1] = self.imgs[-1][pad : pad + new_h, :]
        else:
            nh = new_h
            nw = int(w / h * new_w)
            self.imgs.append(
                cv.resize(self.imgs[-1], (nw, nh), interpolation=cv.INTER_AREA)
            )
            pad = int((nw - new_w) / 2)
            if pad > 0:
                self.imgs[-1] = self.imgs[-1][:, pad : pad + new_w]

    def grayscale(self):
        self.steps.append("grayscale")
        self.imgs.append(cv.cvtColor(self.imgs[-1], cv.COLOR_BGR2GRAY))

    def binary(self):
        self.steps.append("binary")
        self.imgs.append(cv.threshold(self.imgs[-1], 127, 255, cv.THRESH_BINARY)[1])

    def hist_eq(self):
        eq_type = self.kwargs.get("hist_eq").get("type")
        eq_size = self.kwargs.get("hist_eq").get("size")
        self.steps.append("hist_eq")
        if eq_type == "global":
            self.imgs.append(cv.equalizeHist(self.imgs[-1]))
        elif eq_type == "CLAHE":
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=eq_size)
            self.imgs.append(clahe.apply(self.imgs[-1]))
        else:
            self.imgs.append(self.imgs[-1])
            print("Histogram equalization type not yet implemented!")

    def noise_filt(self):
        filt_type = self.kwargs.get("noise_filt").get("type")
        filt_size = self.kwargs.get("noise_filt").get("size")
        self.steps.append("noise_filt")
        if filt_type == "Gaussian":
            self.imgs.append(cv.GaussianBlur(self.imgs[-1], filt_size, 0))
        elif filt_type == "median":
            self.imgs.append(cv.medianBlur(self.imgs[-1], filt_size))
        else:
            self.imgs.append(self.imgas[-1])

    def stretching(self):
        self.steps.append("stretching")
        self.imgs.append(cv.normalize(self.imgs[-1], None, 0, 1))

    def Laplacian(self):
        size = self.kwargs.get("Laplacian").get("size")
        self.steps.append("Laplacian")
        self.imgs.append(cv.Laplacian(self.imgs[-1], ddepth=cv.CV_64F, ksize=size))

    def Canny(self):
        low = self.kwargs.get("Canny").get("low")
        high = self.kwargs.get("Canny").get("high")
        self.steps.append("Canny")
        self.imgs.append(cv.Canny(self.imgs[-1], low, high))

    def Hough(self):
        minLL = self.kwargs.get("Hough").get("minLineLength")
        maxLG = self.kwargs.get("Hough").get("maxLineGap")
        self.steps.append("Hough")
        self.imgs.append(self.imgs[-1])
        self.lines = cv.HoughLinesP(
            self.imgs[-1], 1, np.pi / 180, 10, minLineLength=minLL, maxLineGap=maxLG
        )
        d = []
        for line in self.lines:
            x1, y1, x2, y2 = line[0]
            d.append((x1 - x2) ** 2 + (y1 - y2) ** 2)
        self.lines = [
            x
            for _, x in sorted(
                zip(d, self.lines), key=lambda pair: pair[0], reverse=True
            )
        ]

    def feat_detect(self):
        self.steps.append("feat_detect")
        detector_type = self.kwargs.get("feat_detect").get("type")
        if detector_type == "ORB":
            detector = cv.ORB_create()
        elif detector_type == "SIFT":
            detector = cv.sift_create()
        else:
            self.imgs.append(self.imgs[-1])
            print("Feature detector type not yet implemented!")
            return
        kp = detector.detect(self.imgs[-1], None)
        self.imgs.append(
            cv.drawKeypoints(self.imgs[-1], kp, None, color=(0, 255, 0), flags=0)
        )

    def save_image(self):
        path = self.img_path.split("/")
        path[1] = self.save_folder
        dst = os.path.join(*path[1:])
        cv.imwrite(dst, self.imgs[-1])

    def process_images(self, df_images):
        self.df = df_images
        for df_entry in self.df.itertuples():
            self.load_sample(df_entry)
            self.defect_class = df_entry.defect_str
            for step in self.pipeline_steps[1:]:
                self.function.get(step)()

    def show(self):
        _, ax = plt.subplots(1, len(self.imgs), figsize=(15, 5), tight_layout=True)
        for i, img in enumerate(self.imgs):
            if plot_kw := self.plot_kwargs.get(self.steps[i]):
                ax[i].imshow(img, **plot_kw)
            else:
                ax[i].imshow(img)
            ax[i].set_xticks(ticks=[])
            ax[i].set_yticks(ticks=[])
            ax[i].set_title(self.steps[i])
            if self.steps[i] == "Hough":
                max_lines = self.kwargs.get("Hough").get("maxLines")
                for line in self.lines[:max_lines]:
                    x1, y1, x2, y2 = line[0]
                    x = np.linspace(x1, x2, 100, endpoint=True)
                    y = np.linspace(y1, y2, 100, endpoint=True)
                    ax[i].plot(x, y, c="green")
        ax[0].set_ylabel(self.defect_class)
        plt.show()


class Augmenter(BaseClass):
    def __init__(self, N, df_images, save_folder="./augmented/"):
        self.N = N
        self.images = df_images[df_images.type == "train"]
        self.save_folder = save_folder
        self.augmented_images = pd.DataFrame(
            {
                "type": pd.Series(dtype="str"),
                "defect": pd.Series(dtype="int"),
                "defect_str": pd.Series(dtype="str"),
                "path": pd.Series(dtype="str"),
                "filename": pd.Series(dtype="str"),
                "img": pd.Series(dtype="str"),
                "height": pd.Series(dtype="int"),
                "width": pd.Series(dtype="int"),
                "components": pd.Series(dtype="int"),
                "R_mean": pd.Series(dtype="float"),
                "G_mean": pd.Series(dtype="float"),
                "B_mean": pd.Series(dtype="float"),
            }
        )
        self.augment_functions = [
            self.flip,
            self.random_rotate,
            self.random_zoom,
            self.random_crop,
        ]

    @staticmethod
    def flip(img):
        img = cv.flip(img, 1)
        return img

    @staticmethod
    def random_rotate(img, max_angle=90):
        angle = np.random.uniform(low=-max_angle, high=max_angle)
        h, w = img.shape[:2]
        rotation_matrix = cv.getRotationMatrix2D(
            (h / 2.0, w / 2.0), angle=angle, scale=1
        )
        img = cv.warpAffine(img, rotation_matrix, (w, h))
        return img

    @staticmethod
    def random_zoom(img, min_factor=0.8, max_factor=1.2):
        zoom_factor = np.random.uniform(low=min_factor, high=max_factor)
        img = cv.resize(img, None, fx=zoom_factor, fy=zoom_factor)
        return img

    @staticmethod
    def random_crop(img, max_ratio=0.1):
        pad1 = int(np.random.uniform(low=0, high=max_ratio) * img.shape[0])
        pad2 = int(np.random.uniform(low=0, high=max_ratio) * img.shape[0])
        pad3 = int(np.random.uniform(low=0, high=max_ratio) * img.shape[1])
        pad4 = int(np.random.uniform(low=0, high=max_ratio) * img.shape[1])
        img = img[pad1:, :, :]
        img = img[:-pad2, :, :]
        img = img[:, pad3:, :]
        return img[:, :-pad4, :]

    def select_random_images(self, class_type):
        return np.random.choice(
            self.images.img[self.images.defect == class_type], self.N, replace=False
        )

    def apply_random_function(self, img):
        rand_func = np.random.choice(self.augment_functions)
        img = rand_func(img)
        return img

    def save_image(self, img, img_path):
        img_path = img_path.split("/")
        img_path[1] = self.save_folder
        path = os.path.join(*img_path[1:-1])
        filename = img_path[-1].split(".")
        filename = "".join([filename[0], "_augmented.", filename[1]])
        img_path = os.path.join(path, filename)
        cv.imwrite(img_path, img)
        del img
        return img_path

    def update_df(self, img, img_path):
        img_path = img_path.split("/")
        t = img_path[2].lower()
        defect_str = img_path[3]
        path = os.path.join(*img_path[:-1])
        filename = img_path[-1]
        img_path = os.path.join(path, filename)
        data = {
            "type": t,
            "defect": 1 if defect_str == "Defective" else 0,
            "defect_str": defect_str,
            "path": path,
            "filename": filename,
            "img": img_path,
            "height": img.shape[0],
            "width": img.shape[1],
            "components": img.shape[2],
            "R_mean": np.mean(img[:, :, 0]),
            "G_mean": np.mean(img[:, :, 1]),
            "B_mean": np.mean(img[:, :, 2]),
        }
        df = pd.DataFrame(data=data, index=[1])
        self.augmented_images = pd.concat([self.augmented_images, df])
        self.augmented_images.reset_index(inplace=True, drop=True)

    def augment_images(self):
        for cl in set(self.images.defect):
            img_path_list = self.select_random_images(cl)
            for img_path in img_path_list:
                img = cv.imread(img_path)
                img = self.apply_random_function(img)
                img_path_new = self.save_image(img, img_path)
                self.update_df(img, img_path_new)

    def save_dataframe(self):
        return self.augmented_images


class Model(BaseClass):
    def __init__(self, img_height, img_width, load_folder="./preprocessed/"):
        self.height = img_height
        self.width = img_width
        self.load_folder = load_folder
        self.model = None
        self.data = {}
        self.class_names = None
        self.history = None
        self.predictions = None
        self.score = None
        self.y_pred = None
        self.y_true = None
        self.epochs = 10
        self.batch_size = 32

    def setup_neural_net(self):
        datasets = ["Train", "Validation", "Test"]
        for dataset in datasets:
            self.data.update(
                {
                    dataset: image_dataset_from_directory(
                        os.path.join(self.load_folder, dataset),
                        labels="inferred",
                        label_mode="binary",
                        image_size=(self.height, self.width),
                        color_mode="grayscale",
                        batch_size=self.batch_size,
                    )
                }
            )
        self.class_names = self.data["Train"].class_names

        self.model.compile(
            optimizer=Adam(
                lr=ExponentialDecay(
                    initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.9
                )
            ),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
        )

    def train_net(self):
        self.history = self.model.fit(
            self.data["Train"],
            validation_data=self.data["Validation"],
            epochs=self.epochs,
        )

    def predict_test(self):
        self.predictions = self.model.predict(self.data["Test"])
        self.y_pred = (1 * (self.predictions > 0.5)).ravel()
        self.y_true = (
            np.array(list(self.data["Test"].take(-1))[-1][-1]).ravel().astype(int)
        )
        print("Test values:         ", self.y_true)
        print("Predicted values:    ", self.y_pred)

    def show_metrics(self):
        _, ax = plt.subplots(2, 2, figsize=(15, 10), tight_layout=True)
        sns.lineplot(
            x=np.arange(self.epochs),
            y="binary_accuracy",
            data=self.history.history,
            ax=ax[0, 0],
            label="training",
        )
        sns.lineplot(
            x=np.arange(self.epochs),
            y="val_binary_accuracy",
            data=self.history.history,
            ax=ax[0, 0],
            label="validation",
        )
        sns.lineplot(
            x=np.arange(self.epochs),
            y="loss",
            data=self.history.history,
            ax=ax[0, 1],
            label="training",
        )
        sns.lineplot(
            x=np.arange(self.epochs),
            y="val_loss",
            data=self.history.history,
            ax=ax[0, 1],
            label="validation",
        )
        ConfusionMatrixDisplay.from_predictions(
            self.y_true,
            self.y_pred,
            display_labels=self.class_names,
            ax=ax[1, 0],
            cmap="crest",
            colorbar=False,
        )
        RocCurveDisplay.from_predictions(self.y_true, self.y_pred, ax=ax[1, 1])
        plt.show()
