import logging

logger = logging.getLogger(__name__)

class MotionEstimator:
    def __init__(self, input_csv, output_dir):
        self.input_csv = input_csv
        self.output_dir = output_dir
        logger.warning("Placeholder MotionEstimator initialized. Motion analysis is disabled.")

    def estimate_movement(self):
        logger.warning("Placeholder: Motion estimation not implemented.")
        return False