# Paths to your data
import os

LUNAR_CATALOG_PATH = os.path.abspath(
    "../../data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv"
)
LUNAR_DATA_DIR = os.path.abspath("../../data/lunar/training/data/S12_GradeA/")
LUNAR_SAVE_DIR = os.path.abspath("../../model_output/lunar_preprocessed_images/")
MARTIAN_DATA_DIR = os.path.abspath("../../data/mars/training/data/")
MARTIAN_SAVE_DIR = os.path.abspath("../../model_output/martian_preprocessed_images/")
MODEL_FILENAME = "seismic_cnn_model.pth"
MODEL_DICT_FILENAME = "seismic_cnn_model_dict.pth"
ONNX_MODEL_PATH = os.path.abspath("../../model_output/martian_seismic_cnn_model.onnx")
SAVE_DIR = os.path.abspath("../../model_output")
TEST_DATA_DIRS = [
    os.path.abspath("../../data/mars/test"),
    os.path.abspath("../../data/lunar/test/data/S12_GradeB"),
    os.path.abspath("../../data/lunar/test/data/S15_GradeA"),
    os.path.abspath("../../data/lunar/test/data/S15_GradeB"),
    os.path.abspath("../../data/lunar/test/data/S16_GradeA"),
    os.path.abspath("../../data/lunar/test/data/S16_GradeB"),
]
CATALOG_PNG_DIR = os.path.abspath("../../model_output/catalog_png")
