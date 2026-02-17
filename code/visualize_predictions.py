#!/usr/bin/env python3
"""
Visualize per-window predictions over time for a test CSV file.

Usage:
    python3 code/visualize_predictions.py --models models_out --model stacked_model --file dataset/test/10sec.csv

This script:
- loads a saved model (random_forest or xgboost or stacked_model) from the models directory
- generates feature windows from the specified CSV using the existing extractor
- predicts labels for each window
- opens a small GUI that shows:
  - a timeline (bar) of predicted labels across windows
  - a large label + colored background indicating the current window's class
  - play/pause and step controls

Notes:
- The visualization simulates "real time" using step_seconds=0.5s per window.
- Requires PyQt / pyqtgraph (the repo already references these in `live_data/`).
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import joblib
import traceback
from typing import Tuple, Dict, Any, Optional

# Use pyqtgraph's Qt wrapper (consistent with other project files)
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph as pg

from EEG_feature_extraction_adv import generate_feature_vectors_from_samples

# Constants
LABEL_MAP = {
    0: 'relaxed',
    1: 'neutral',
    2: 'concentrating'
}

LABEL_COLORS = {
    'relaxed': (100, 180, 255),       # light blue
    'neutral': (200, 200, 200),       # gray
    'concentrating': (120, 255, 120)  # light green
}


def setup_arg_parser() -> argparse.ArgumentParser:
    """Set up and return the argument parser with required arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default='models_out')
    parser.add_argument('--model', default='stacked_model', 
                       choices=['random_forest', 'xgboost', 'stacked_model'])
    parser.add_argument('--file', required=True, 
                       help='test CSV file to visualize')
    parser.add_argument('--nsamples', type=int, default=150)
    parser.add_argument('--period', type=float, default=1.0)
    parser.add_argument('--cols_to_ignore', default=-1)
    return parser


def load_model(models_dir: Path, model_name: str) -> Any:
    """Load a trained model from the specified directory."""
    mapping = {
        'random_forest': 'random_forest.joblib',
        'xgboost': 'xgboost.joblib',
        'stacked_model': 'stacked_model.joblib'
    }
    model_file = models_dir / mapping[model_name]
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    return joblib.load(model_file)


def validate_file_path(file_path: Path) -> None:
    """Check if the specified file exists, show error and exit if not."""
    if not file_path.exists():
        QtWidgets.QMessageBox.critical(
            None, 'File Not Found', 
            f'The specified file was not found: {file_path}'
        )
        sys.exit(1)


def extract_features(file_path: Path, nsamples: int, 
                   period: float, cols_to_ignore: int) -> Tuple[np.ndarray, list]:
    """Extract feature vectors from the input file."""
    vectors, headers = generate_feature_vectors_from_samples(
        str(file_path), 
        nsamples=nsamples,
        period=period, 
        state=None,
        cols_to_ignore=cols_to_ignore
    )
    
    if vectors is None or len(vectors) == 0:
        QtWidgets.QMessageBox.critical(
            None, 'Feature Extraction Failed',
            'No feature vectors could be generated from the input file.\n\n'
            'Possible reasons:\n'
            '1. The file format may be incorrect\n'
            '2. The recording may be too short\n'
            '3. The headset may not be properly connected'
        )
        sys.exit(1)
    
    return np.asarray(vectors, dtype=float), headers


def show_error_dialog(title: str, message: str) -> None:
    """Display an error dialog with the given title and message."""
    QtWidgets.QMessageBox.critical(None, title, message)


class Visualizer(QtWidgets.QWidget):
    """Main visualization widget for displaying prediction results."""
    
    def __init__(self, preds: np.ndarray, step_seconds: float = 0.5):
        """Initialize the visualizer with predictions and step size."""
        super().__init__()
        self.preds = np.asarray(preds, dtype=int)
        self.n = len(self.preds)
        self.step = step_seconds
        self.idx = 0
        self.setup_timer()
        self.setup_ui()
        self.update_display()

    def setup_timer(self) -> None:
        """Set up the timer for automatic playback."""
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)

    def setup_ui(self) -> None:
        """Initialize the user interface components."""
        self.setWindowTitle('Prediction Visualizer')
        layout = self.create_main_layout()
        self.setLayout(layout)
        self.plot_widget.setXRange(-1, max(10, self.n))

    def create_main_layout(self) -> QtWidgets.QVBoxLayout:
        """Create and return the main layout with all UI components."""
        layout = QtWidgets.QVBoxLayout()
        self.setup_status_label(layout)
        self.setup_error_label(layout)
        self.setup_plot_widget(layout)
        self.setup_control_buttons(layout)
        return layout

    def setup_status_label(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up the main status label."""
        self.label = QtWidgets.QLabel('', alignment=QtCore.Qt.AlignCenter)
        font = self.label.font()
        font.setPointSize(28)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setFixedHeight(120)
        layout.addWidget(self.label)

    def setup_error_label(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up the error message label."""
        self.error_label = QtWidgets.QLabel('', alignment=QtCore.Qt.AlignCenter)
        self.error_label.setStyleSheet('color: red; font-size: 16px;')
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)

    def setup_plot_widget(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up the plot widget for timeline visualization."""
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setFixedHeight(120)
        self.plot_widget.hideAxis('left')
        self.plot_widget.showGrid(x=True, y=False)
        self.setup_timeline()
        layout.addWidget(self.plot_widget)

    def setup_timeline(self) -> None:
        """Set up the timeline visualization with colored segments."""
        img = self.create_timeline_image()
        self.image = pg.ImageItem(img)
        self.plot_widget.addItem(self.image)
        self.setup_vline()
        self.plot_widget.setYRange(0, 1)

    def create_timeline_image(self) -> np.ndarray:
        """Create the colored timeline image from predictions."""
        img = np.zeros((1, self.n, 3), dtype=np.uint8)
        for i, p in enumerate(self.preds):
            name = LABEL_MAP.get(int(p), str(p))
            color = LABEL_COLORS.get(name, (220, 220, 220))
            img[0, i, :] = np.array(color, dtype=np.uint8)
        return img

    def setup_vline(self) -> None:
        """Set up the vertical line indicator."""
        self.vline = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.plot_widget.addItem(self.vline)

    def setup_control_buttons(self, layout: QtWidgets.QVBoxLayout) -> None:
        """Set up the control buttons and speed control."""
        btn_layout = QtWidgets.QHBoxLayout()
        self.setup_playback_buttons(btn_layout)
        self.setup_speed_control(btn_layout)
        layout.addLayout(btn_layout)

    def setup_playback_buttons(self, layout: QtWidgets.QHBoxLayout) -> None:
        """Set up the playback control buttons."""
        self.play_btn = QtWidgets.QPushButton('Play')
        self.play_btn.clicked.connect(self.toggle_play)
        layout.addWidget(self.play_btn)

        self.step_back_btn = QtWidgets.QPushButton('Step <-')
        self.step_back_btn.clicked.connect(self.step_back)
        layout.addWidget(self.step_back_btn)

        self.step_fwd_btn = QtWidgets.QPushButton('-> Step')
        self.step_fwd_btn.clicked.connect(self.next_frame)
        layout.addWidget(self.step_fwd_btn)

    def setup_speed_control(self, layout: QtWidgets.QHBoxLayout) -> None:
        """Set up the playback speed control."""
        layout.addStretch()
        self.speed_label = QtWidgets.QLabel('step (s):')
        layout.addWidget(self.speed_label)
        
        self.step_spin = QtWidgets.QDoubleSpinBox()
        self.step_spin.setSingleStep(0.1)
        self.step_spin.setDecimals(2)
        self.step_spin.setValue(self.step)
        self.step_spin.valueChanged.connect(self.change_step)
        layout.addWidget(self.step_spin)

    def change_step(self, val: float) -> None:
        """Change the playback step size."""
        self.step = float(val)
        if self.timer.isActive():
            self.timer.setInterval(int(self.step * 1000))

    def toggle_play(self) -> None:
        """Toggle between play and pause states."""
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText('Play')
        else:
            self.timer.start(int(self.step * 1000))
            self.play_btn.setText('Pause')

    def step_back(self) -> None:
        """Move one step back in the timeline."""
        self.idx = max(0, self.idx - 1)
        self.update_display()

    def next_frame(self) -> None:
        """Advance to the next frame in the timeline."""
        if self.idx < self.n - 1:
            self.idx += 1
            self.update_display()
        else:
            self.timer.stop()
            self.play_btn.setText('Play')

    def update_display(self) -> None:
        """Update the display with the current prediction."""
        if self.n == 0:
            self.label.setText('No windows')
            self.error_label.clear()
            return
            
        try:
            self.update_prediction_display()
        except Exception as e:
            self.handle_display_error(e)
    
    def update_prediction_display(self) -> None:
        """Update the display with the current prediction."""
        p = int(self.preds[self.idx])
        name = LABEL_MAP.get(p, str(p))
        color = LABEL_COLORS.get(name, (220, 220, 220))
        r, g, b = color
        self.label.setText(f"{self.idx+1}/{self.n}: {name}")
        self.label.setStyleSheet(f'background-color: rgb({r},{g},{b});')
        self.error_label.clear()
        self.vline.setPos(self.idx)
        self.plot_widget.setXRange(max(0, self.idx - 10), self.idx + 5)
    
    def handle_display_error(self, error: Exception) -> None:
        """Handle errors that occur during display updates."""
        error_msg = str(error)
        if 'HeadsetError' in error_msg:
            self.show_error(error_msg.replace('HeadsetError: ', ''))
        else:
            self.show_error(f"Error: {error_msg}")
    
    def show_error(self, message: str) -> None:
        """Display an error message in the GUI."""
        self.error_label.setText(message)
        self.label.setText("ERROR")
        self.label.setStyleSheet('background-color: #FFDDDD;')
        self.plot_widget.hide()
        
    def clear_error(self) -> None:
        """Clear any displayed error messages."""
        self.error_label.clear()
        self.plot_widget.show()


def run_visualization(args: argparse.Namespace) -> None:
    """Run the visualization with the given command line arguments."""
    try:
        models_dir = Path(args.models)
        model = load_model(models_dir, args.model)
        validate_file_path(Path(args.file))
        
        X, _ = extract_features(
            Path(args.file), 
            args.nsamples, 
            args.period, 
            args.cols_to_ignore
        )
        
        preds = model.predict(X)
        show_visualizer(preds)
        
    except Exception as e:
        handle_runtime_error(e)


def show_visualizer(predictions: np.ndarray) -> None:
    """Create and show the visualization window."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    viz = Visualizer(predictions, step_seconds=0.5)
    viz.resize(900, 300)
    viz.show()
    sys.exit(app.exec_())


def handle_runtime_error(error: Exception) -> None:
    """Handle runtime errors by showing an error dialog and exiting."""
    error_msg = f"An unexpected error occurred:\n{str(error)}\n\n{traceback.format_exc()}"
    show_error_dialog('Error', error_msg)
    sys.exit(1)


def main() -> None:
    """Main entry point for the visualization script."""
    parser = setup_arg_parser()
    args = parser.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    run_visualization(args)


if __name__ == '__main__':
    main()
