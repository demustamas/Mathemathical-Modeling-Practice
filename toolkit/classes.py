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
from tensorflow.keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

from tqdm import tqdm

plt.rcParams.update({"font.size": 16})


class DataSet:
    def __init__(
        self,
        raw_folder="./raw/",
        data_folder="./data/",
        preprocessed_folder="./preprocessed/",
        augmented_folder="./augmented/",
        plot_folder="./tex_graphs/",
        PATH=None,
    ):
        self.raw = pd.DataFrame()
        self.data = pd.DataFrame()
        self.augmented = pd.DataFrame()
        self.preprocessed = pd.DataFrame()
        self.random_sample = pd.DataFrame()

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

    def update_dataset(self, dataset=None):
        def fill_entries(data_path):
            df = pd.DataFrame(
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
            for each in data_path:
                s = each.split("/")
                y = 1 if s[3] == "Defective" else 0
                if len(os.listdir(each)) > 0:
                    for i, img in enumerate(os.listdir(each)):
                        data = {
                            "type": s[2].lower(),
                            "defect": y,
                            "defect_str": s[3],
                            "path": each,
                            "filename": img,
                            "img": os.path.join(each, img),
                            "height": 0,
                            "width": 0,
                            "components": 0,
                            "R_mean": 0,
                            "G_mean": 0,
                            "B_mean": 0,
                        }
                        new_entry = pd.DataFrame(data=data, index=[len(df.index)])
                        df = pd.concat([df, new_entry])
                    print(f"Found {i+1} images in {each}")
                else:
                    print(f"Found 0 images in {each}")
            return df

        if dataset == "raw":
            self.raw = fill_entries(self.RAW_PATH)
        elif dataset == "data":
            self.data = fill_entries(self.DATA_PATH)
        elif dataset == "augmented":
            self.augmented = fill_entries(self.AUGMENTED_PATH)
        elif dataset == "preprocessed":
            self.preprocessed = fill_entries(self.PREPROCESSED_PATH)
        else:
            print("Dataset not found! No update done!")

    def generate_random_sample(self, size=4):
        random_sample_idx = []
        idx1 = np.random.choice(
            self.data[(self.data.type == "train") & (self.data.defect == 0)].index,
            size=size,
            replace=False,
        )
        idx2 = np.random.choice(
            self.data[(self.data.type == "train") & (self.data.defect == 1)].index,
            size=size,
            replace=False,
        )
        random_sample_idx = np.concatenate([idx1, idx2])
        self.random_sample = self.data.iloc[random_sample_idx].copy()
        self.random_sample.reset_index(drop=True, inplace=True)

    def show_random_sample(self):
        size = len(self.random_sample.index) // 2
        _, ax = plt.subplots(size, 2, figsize=(15, 6 * size), tight_layout=True)
        ax = ax.flatten()

        for i, img in enumerate(self.random_sample.img):
            image = cv.imread(img)
            ax[i].imshow(image)
            ax[i].set_xticks(ticks=[])
            ax[i].set_yticks(ticks=[])
            ax[i].set_title(
                f"{self.random_sample.defect_str[i]} - {self.random_sample.filename[i]}"
            )

        plt.show()

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
    def __init__(
        self, steps=None, kwargs=None, plot_kwargs=None, save_folder="./preprocessed/"
    ):
        if steps is None:
            steps = []
        if kwargs is None:
            kwargs = {}
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
            "rotate": self.rotate,
            "grayscale": self.grayscale,
            "threshold": self.threshold,
            "hist_eq": self.hist_eq,
            "noise_filt": self.noise_filt,
            "stretching": self.stretching,
            "edge_detector": self.edge_detector,
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

    def rotate(self):
        angle = self.kwargs.get("rotate").get("angle")
        h, w = self.imgs[-1].shape[:2]
        rotation_matrix = cv.getRotationMatrix2D(
            (h / 2.0, w / 2.0), angle=angle, scale=1
        )
        self.steps.append("rotation")
        self.imgs.append(cv.warpAffine(self.imgs[-1], rotation_matrix, (w, h)))

    def grayscale(self):
        self.steps.append("grayscale")
        self.imgs.append(cv.cvtColor(self.imgs[-1], cv.COLOR_BGR2GRAY))

    def threshold(self):
        bin_type = self.kwargs.get("threshold").get("type")
        self.steps.append("threshold")
        if bin_type == "global":
            self.imgs.append(cv.threshold(self.imgs[-1], 127, 255, cv.THRESH_BINARY)[1])
        elif bin_type == "otsu":
            self.imgs.append(
                cv.threshold(self.imgs[-1], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[
                    1
                ]
            )
        else:
            self.imgs.append(self.imgs[-1])
            print("Threshold type not yet implemented!")

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

    def edge_detector(self):
        edge_type = self.kwargs.get("edge_detector").get("type")
        size = self.kwargs.get("edge_detector").get("size")
        low = self.kwargs.get("edge_detector").get("low")
        high = self.kwargs.get("edge_detector").get("high")
        self.steps.append("edge_detector")
        if edge_type == "Laplacian":
            self.imgs.append(cv.Laplacian(self.imgs[-1], ddepth=cv.CV_8U, ksize=size))
        elif edge_type == "Canny":
            self.imgs.append(cv.Canny(self.imgs[-1], low=low, high=high))
        else:
            self.imgs.append(self.imgs[-1])
            print("Edge detector type not yet implemented!")

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
        for df_entry in tqdm(self.df.itertuples()):
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

    def augment_images(self):
        for cl in set(self.images.defect):
            img_path_list = self.select_random_images(cl)
            for img_path in tqdm(img_path_list):
                img = cv.imread(img_path)
                img = self.apply_random_function(img)
                self.save_image(img, img_path)
                del img

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
            optimizer=Adam(learning_rate=0.01),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

    def train_net(self):
        self.history = self.model.fit(
            self.data["Train"],
            validation_data=self.data["Validation"],
            epochs=self.epochs,
            callbacks=[
                TensorBoard(log_dir="./logs"),
                ModelCheckpoint(
                    "./model/vgg16_1.h5",
                    monitor="val_accuracy",
                    verbose=1,
                    save_best_only=True,
                    save_weights_only=False,
                    mode="auto",
                    save_freq="epoch",
                ),
                EarlyStopping(
                    monitor="val_accuracy",
                    min_delta=0,
                    patience=30,
                    verbose=1,
                    mode="auto",
                ),
                # ReduceLROnPlateau(
                #     monitor="accuracy", factor=0.1, patience=5, min_lr=0.001
                # ),
            ],
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
            y="accuracy",
            data=self.history.history,
            ax=ax[0, 0],
            label="training",
        )
        sns.lineplot(
            x=np.arange(self.epochs),
            y="val_accuracy",
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
