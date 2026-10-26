"""Reusable widgets and small Qt helpers."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt


class ImageCanvas(QtWidgets.QLabel):
    def __init__(self, placeholder: str = "") -> None:
        super().__init__()
        self._placeholder = placeholder or (
            "将截图拖到这里\n或使用区域截图 / 自动监控"
        )
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(620, 420)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Expanding,
        )
        self._image: QtGui.QImage | None = None
        self.setText(self._placeholder)
        self.setObjectName("ImageCanvas")

    def set_image(self, image: QtGui.QImage | None) -> None:
        self._image = image
        if image is None or image.isNull():
            self.clear()
            self.setText(self._placeholder)
            return
        self._refresh()

    def image(self) -> QtGui.QImage | None:
        return self._image

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._image is not None:
            self._refresh()

    def _refresh(self) -> None:
        if self._image is None:
            return
        pixmap = QtGui.QPixmap.fromImage(self._image)
        scaled = pixmap.scaled(
            self.size() - QtCore.QSize(32, 32),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)


class QMessageBoxCompat:
    @staticmethod
    def warning(parent: QtWidgets.QWidget, title: str, text: str) -> None:
        box = QtWidgets.QMessageBox(parent)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        box.open()

    @staticmethod
    def info(parent: QtWidgets.QWidget, title: str, text: str) -> None:
        box = QtWidgets.QMessageBox(parent)
        box.setIcon(QtWidgets.QMessageBox.Icon.Information)
        box.setWindowTitle(title)
        box.setText(text)
        box.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        box.open()


def labeled(text: str, widget: QtWidgets.QWidget) -> QtWidgets.QVBoxLayout:
    layout = QtWidgets.QVBoxLayout()
    layout.setSpacing(5)
    label = QtWidgets.QLabel(text)
    label.setObjectName("FieldLabel")
    layout.addWidget(label)
    layout.addWidget(widget)
    return layout


def section_label(index: str, title: str, subtitle: str) -> QtWidgets.QWidget:
    widget = QtWidgets.QWidget()
    layout = QtWidgets.QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(9)
    index_label = QtWidgets.QLabel(index)
    index_label.setObjectName("SectionIndex")
    text_box = QtWidgets.QVBoxLayout()
    text_box.setSpacing(0)
    title_label = QtWidgets.QLabel(title)
    title_label.setObjectName("SectionTitle")
    subtitle_label = QtWidgets.QLabel(subtitle)
    subtitle_label.setObjectName("SectionSub")
    text_box.addWidget(title_label)
    text_box.addWidget(subtitle_label)
    layout.addWidget(index_label)
    layout.addLayout(text_box)
    layout.addStretch(1)
    return widget


def slider_row(
    *,
    title: str,
    minimum: int,
    maximum: int,
    value: int,
    suffix: str = "",
    step: int = 1,
) -> tuple[QtWidgets.QWidget, QtWidgets.QSlider, QtWidgets.QLabel]:
    container = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(5)
    header = QtWidgets.QHBoxLayout()
    header.setContentsMargins(0, 0, 0, 0)
    label = QtWidgets.QLabel(title)
    label.setObjectName("FieldLabel")
    output = QtWidgets.QLabel()
    output.setObjectName("RangeOutput")
    output.setAlignment(Qt.AlignmentFlag.AlignRight)
    header.addWidget(label)
    header.addStretch(1)
    header.addWidget(output)
    slider = QtWidgets.QSlider(Qt.Orientation.Horizontal)
    slider.setRange(minimum, maximum)
    slider.setSingleStep(step)
    slider.setPageStep(step)
    slider.setValue(value)
    slider.setProperty("suffix", suffix)
    layout.addLayout(header)
    layout.addWidget(slider)
    return container, slider, output