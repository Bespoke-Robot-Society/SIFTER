import os
from config import SAVE_DIR, MODEL_FILENAME, CATALOG_PNG_DIR, TEST_DATA_DIRS

from cnn.model_predictor import Predictor

# Main execution
if __name__ == "__main__":

    # Directory to save .png plots
    if not os.path.exists(CATALOG_PNG_DIR):
        os.makedirs(CATALOG_PNG_DIR)

    # Instantiate the Predictor class with the model path and save directories
    predictor = Predictor(
        cnn_model_path=os.path.join(SAVE_DIR, MODEL_FILENAME),
        save_dir=SAVE_DIR,
        catalog_png_dir=CATALOG_PNG_DIR,
        batch_size=32,
    )

    # Run the prediction process on the defined test data directories
    predictor.process_and_predict(test_data_dirs=TEST_DATA_DIRS)
