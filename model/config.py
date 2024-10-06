# Paths to your data
LUNAR_CATALOG_PATH = "data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv"
LUNAR_DATA_DIR = "data/lunar/training/data/S12_GradeA/"
LUNAR_SAVE_DIR = "model_output/lunar_preprocessed_images/"
MARTIAN_DATA_DIR = "data/mars/training/data/"
MARTIAN_SAVE_DIR = "model_output/martian_preprocessed_images/"
MODEL_FILENAME = "seismic_cnn_model.pth"
ONNX_MODEL_PATH = "model_output/martian_seismic_cnn_model.onnx"
SAVE_DIR = "model_output"
TEST_DATA_DIRS = [
    "data/mars/test",
    "data/lunar/test/data/S12_GradeB",
    "data/lunar/test/data/S15_GradeA",
    "data/lunar/test/data/S15_GradeB",
    "data/lunar/test/data/S16_GradeA",
    "data/lunar/test/data/S16_GradeB",
]
