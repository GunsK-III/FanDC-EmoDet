# common dependencies 公共依赖项
import os
import warnings
import logging
from typing import Any, Dict, List, Union, Optional

# this has to be set before importing tensorflow
# 指示TensorFlow使用旧版的Keras接口进行操作
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# pylint: disable=wrong-import-position

# 3rd party dependencies    第三方依赖
import numpy as np
import pandas as pd
import tensorflow as tf

# package dependencies      包的依赖关系
from deepface.commons import package_utils, folder_utils
from deepface.commons.logger import Logger
from deepface.modules import (
    modeling,
    representation,
    verification,
    recognition,
    demography,
    detection,
    streaming,
    preprocessing,
)
from deepface import __version__

logger = Logger()

# -----------------------------------
# configurations for dependencies   依赖项的配置

# users should install tf_keras package if they are using tf 2.16 or later versions
package_utils.validate_for_keras3()

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf_version = package_utils.get_tf_major_version()
if tf_version == 2:
    tf.get_logger().setLevel(logging.ERROR)
# -----------------------------------

# create required folders if necessary to store model weights
# 如果有必要，创建所需的文件夹来存储模型权重
folder_utils.initialize_folder()


def build_model(model_name: str, task: str = "facial_recognition") -> Any:
    return modeling.build_model(task=task, model_name=model_name)


def verify(
        img1_path: Union[str, np.ndarray, List[float]],
        img2_path: Union[str, np.ndarray, List[float]],
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        distance_metric: str = "cosine",
        enforce_detection: bool = True,
        align: bool = True,
        expand_percentage: int = 0,
        normalization: str = "base",
        silent: bool = False,
        threshold: Optional[float] = None,
        anti_spoofing: bool = False,
) -> Dict[str, Any]:

    return verification.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        normalization=normalization,
        silent=silent,
        threshold=threshold,
        anti_spoofing=anti_spoofing,
    )


def analyze(
        img_path: Union[str, np.ndarray],
        actions: Union[tuple, list] = ("emotion", "age", "gender", "race"),
        enforce_detection: bool = True,
        detector_backend: str = "opencv",
        align: bool = True,
        expand_percentage: int = 0,
        silent: bool = False,
        anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:

    return demography.analyze(
        img_path=img_path,
        actions=actions,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        silent=silent,
        anti_spoofing=anti_spoofing,
    )


def find(
        img_path: Union[str, np.ndarray],
        db_path: str,
        model_name: str = "VGG-Face",
        distance_metric: str = "cosine",
        enforce_detection: bool = True,
        detector_backend: str = "opencv",
        align: bool = True,
        expand_percentage: int = 0,
        threshold: Optional[float] = None,
        normalization: str = "base",
        silent: bool = False,
        refresh_database: bool = True,
        anti_spoofing: bool = False,
        batched: bool = False,
) -> Union[List[pd.DataFrame], List[List[Dict[str, Any]]]]:

    return recognition.find(
        img_path=img_path,
        db_path=db_path,
        model_name=model_name,
        distance_metric=distance_metric,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        threshold=threshold,
        normalization=normalization,
        silent=silent,
        refresh_database=refresh_database,
        anti_spoofing=anti_spoofing,
        batched=batched,
    )


def represent(
        img_path: Union[str, np.ndarray],
        model_name: str = "VGG-Face",
        enforce_detection: bool = True,
        detector_backend: str = "opencv",
        align: bool = True,
        expand_percentage: int = 0,
        normalization: str = "base",
        anti_spoofing: bool = False,
        max_faces: Optional[int] = None,
) -> List[Dict[str, Any]]:

    return representation.represent(
        img_path=img_path,
        model_name=model_name,
        enforce_detection=enforce_detection,
        detector_backend=detector_backend,
        align=align,
        expand_percentage=expand_percentage,
        normalization=normalization,
        anti_spoofing=anti_spoofing,
        max_faces=max_faces,
    )


def stream(
        db_path: str = "",
        model_name: str = "VGG-Face",
        detector_backend: str = "opencv",
        distance_metric: str = "cosine",
        enable_face_analysis: bool = True,
        source: Any = 0,
        time_threshold: int = 5,
        frame_threshold: int = 5,
        anti_spoofing: bool = False,
) -> None:

    time_threshold = max(time_threshold, 1)
    frame_threshold = max(frame_threshold, 1)

    streaming.analysis(
        db_path=db_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        enable_face_analysis=enable_face_analysis,
        source=source,
        time_threshold=time_threshold,
        frame_threshold=frame_threshold,
        anti_spoofing=anti_spoofing,
    )


def extract_faces(
        img_path: Union[str, np.ndarray],
        detector_backend: str = "opencv",
        enforce_detection: bool = True,
        align: bool = True,
        expand_percentage: int = 0,
        grayscale: bool = False,
        color_face: str = "rgb",
        normalize_face: bool = True,
        anti_spoofing: bool = False,
) -> List[Dict[str, Any]]:


    return detection.extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
        expand_percentage=expand_percentage,
        grayscale=grayscale,
        color_face=color_face,
        normalize_face=normalize_face,
        anti_spoofing=anti_spoofing,
    )


def cli() -> None:
    import fire
    fire.Fire()

# deprecated function(s)


def detectFace(
        img_path: Union[str, np.ndarray],
        target_size: tuple = (224, 224),
        detector_backend: str = "opencv",
        enforce_detection: bool = True,
        align: bool = True,
) -> Union[np.ndarray, None]:

    logger.warn("Function detectFace is deprecated. Use extract_faces instead.")
    face_objs = extract_faces(
        img_path=img_path,
        detector_backend=detector_backend,
        grayscale=False,
        enforce_detection=enforce_detection,
        align=align,
    )
    extracted_face = None
    if len(face_objs) > 0:
        extracted_face = face_objs[0]["face"]
        extracted_face = preprocessing.resize_image(img=extracted_face, target_size=target_size)
    return extracted_face
