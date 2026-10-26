"""O-Translator desktop application.

Three capture modes:
  * Manual upload / clipboard paste
  * Area hotkey capture (DPI-aware snipper)
  * Fixed-region auto monitoring with debounce and cooldown
"""

from __future__ import annotations

import ctypes
import html
import json
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QThreadPool, QTimer

from client.config import (
    API_URL,
    APP_NAME,
    DEFAULT_HOTKEY,
    HOTKEY_SETTINGS_VERSION,
    ICON_PATH,
    LEGACY_DEFAULT_HOTKEY,
    ensure_capture_dir,
    is_image_path,
)
from client.services.api_client import ApiClient, BackendManager, Worker
from client.services.hotkey import GlobalHotkey
from client.services.region_monitor import MonitorSettings, RegionMonitor
from client.services.screen_capture import (
    CaptureError,
    CaptureRegion,
    CaptureResult,
    capture_region_image,
    refresh_region,
)
from client.services.single_instance import SingleInstanceGuard
from client.ui.floating_window import FloatingTranslationWindow
from client.ui.overlay_snipper import OverlaySnipper
from client.ui.styles import APP_STYLE
from client.ui.widgets import (
    ImageCanvas,
    QMessageBoxCompat,
    labeled,
    section_label,
    slider_row,
)

STAGE_TEXT = {
    "queued": "任务已排队",
    "decoding": "正在读取截图",
    "ocr": "正在 OCR 识别文字",
    "masking": "正在生成文字蒙版",
    "processing": "正在修复背景并翻译",
    "inpainting": "背景修复完成，等待翻译",
    "translation": "翻译完成，等待背景修复",
    "preview": "译文预览已显示，正在精修背景",
    "rendering": "正在自适应排版",
    "saving": "正在保存结果",
    "complete": "处理完成",
    "failed": "处理失败",
}

MODE_LABELS = {
    "manual": "手动上传",
    "area": "区域截图",
    "monitor": "区域监控",
}


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        *,
        register_hotkey: bool = True,
        bootstrap_backend: bool = True,
    ) -> None:
        super().__init__()
        self.settings = QtCore.QSettings("O-Translator", "Desktop")
        self.api = ApiClient()
        self.backend = BackendManager(self.api)
        self.pool = QThreadPool.globalInstance()
        self.tray: QtWidgets.QSystemTrayIcon | None = None

        self.current_job_id: str | None = None
        self.current_result: dict[str, Any] | None = None
        self.result_image: QtGui.QImage | None = None
        self.preview_image: QtGui.QImage | None = None
        self._preview_job_id: str | None = None
        self._preview_inflight = False
        self.last_file: Path | None = None
        self._last_result_image: QtGui.QImage | None = None
        self._last_result_region: CaptureRegion | None = None
        self._last_result_source = "upload"

        self._poll_inflight = False
        self._poll_failures = 0
        self._busy = False
        self._hotkey_active = False
        self._force_quit = False
        self._restore_after_capture = False
        self._selection_context = "area"
        self._selection_cursor: QtCore.QPoint | None = None
        self._selection_pending = False
        self._selector: OverlaySnipper | None = None
        self._floating: FloatingTranslationWindow | None = None
        self._floating_was_visible = False

        self._job_region: CaptureRegion | None = None
        self._job_source = "upload"
        self._job_background = False
        self._job_auto_overlay = False

        self._mode = "manual"
        self._monitor_region: CaptureRegion | None = None
        self._monitor_running = False
        self._monitor_paused = False
        self._pending_monitor_image: QtGui.QImage | None = None

        saved_hotkey = str(self.settings.value("hotkey", "")).strip().lower()
        settings_version = str(self.settings.value("hotkey_version", "1"))
        if (
            not saved_hotkey
            or (
                settings_version != HOTKEY_SETTINGS_VERSION
                and saved_hotkey == LEGACY_DEFAULT_HOTKEY
            )
        ):
            saved_hotkey = DEFAULT_HOTKEY
            self.settings.setValue("hotkey", saved_hotkey)
            self.settings.setValue("hotkey_version", HOTKEY_SETTINGS_VERSION)
        self.hotkey = GlobalHotkey(saved_hotkey, self)
        self.monitor = RegionMonitor(self)
        self._bind_monitor()

        self.setWindowTitle("O-Translator · 截图翻译助手")
        self.setMinimumSize(1080, 740)
        self.resize(1200, 840)
        self.setAcceptDrops(True)
        self._build_ui()
        self._load_preferences()
        self._load_monitor_preferences()
        self._build_tray()

        self.paste_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence.StandardKey.Paste, self
        )
        self.paste_shortcut.activated.connect(self._paste_clipboard)
        self.retranslate_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Return"), self
        )
        self.retranslate_shortcut.activated.connect(self._retranslate)

        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(300)
        self.poll_timer.timeout.connect(self._poll_job)

        self.hotkey.activated.connect(self._on_hotkey_activated)
        if register_hotkey:
            QTimer.singleShot(80, self._register_hotkey)
        if bootstrap_backend:
            QTimer.singleShot(200, self._bootstrap_backend)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._shutdown)

        self._sync_mode_ui()
        self._sync_monitor_ui()
        self._set_mode(str(self.settings.value("app_mode", "manual")))

    # ------------------------------------------------------------------ build
    def _build_ui(self) -> None:
        if ICON_PATH.exists():
            self.setWindowIcon(QtGui.QIcon(str(ICON_PATH)))
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(26, 18, 26, 18)
        root.setSpacing(15)

        root.addLayout(self._build_header())
        split = QtWidgets.QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(14)
        split.setChildrenCollapsible(False)
        root.addWidget(split, 1)
        split.addWidget(self._build_stage())
        split.addWidget(self._build_sidebar())
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        split.setSizes([760, 390])

        footer = QtWidgets.QHBoxLayout()
        self.queue_label = QtWidgets.QLabel("LOCAL PIPELINE")
        self.queue_label.setObjectName("FooterLabel")
        footer.addWidget(self.queue_label)
        footer.addStretch(1)
        self.mode_hint = QtWidgets.QLabel(
            "拖入图片 · Ctrl+V 粘贴 · Ctrl+Enter 重译 · 全局热键框选"
        )
        self.mode_hint.setObjectName("FooterLabel")
        footer.addWidget(self.mode_hint)
        root.addLayout(footer)
        self.setStyleSheet(APP_STYLE)

    def _build_header(self) -> QtWidgets.QHBoxLayout:
        header = QtWidgets.QHBoxLayout()
        brand_box = QtWidgets.QVBoxLayout()
        brand_box.setSpacing(1)
        brand = QtWidgets.QLabel("O-Translator")
        brand.setObjectName("BrandTitle")
        subtitle = QtWidgets.QLabel(
            "SCREENSHOT · OCR · INPAINT · TRANSLATE · AUTO MONITOR"
        )
        subtitle.setObjectName("BrandSub")
        brand_box.addWidget(brand)
        brand_box.addWidget(subtitle)
        header.addLayout(brand_box)
        header.addStretch(1)
        self.health_label = QtWidgets.QLabel("连接中…")
        self.health_label.setObjectName("HealthBadge")
        self.health_label.setStyleSheet("color:#8b8e98;")
        self.hotkey_label = QtWidgets.QLabel(self.hotkey.display)
        self.hotkey_label.setObjectName("HotkeyPill")
        self.api_label = QtWidgets.QLabel(API_URL.replace("http://", ""))
        self.api_label.setObjectName("ApiPill")
        header.addWidget(self.health_label)
        header.addSpacing(8)
        header.addWidget(self.hotkey_label)
        header.addSpacing(8)
        header.addWidget(self.api_label)
        return header

    def _build_stage(self) -> QtWidgets.QWidget:
        stage = QtWidgets.QFrame()
        stage.setObjectName("Stage")
        layout = QtWidgets.QVBoxLayout(stage)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(9)
        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, 1)
        footer = QtWidgets.QHBoxLayout()
        self.image_meta = QtWidgets.QLabel("NO IMAGE")
        self.image_meta.setObjectName("MetaLabel")
        self.open_overlay_btn = QtWidgets.QPushButton("浮窗预览")
        self.open_overlay_btn.setObjectName("SubtleButton")
        self.open_overlay_btn.setEnabled(False)
        self.open_overlay_btn.clicked.connect(self.show_overlay)
        self.save_btn = QtWidgets.QPushButton("导出 PNG")
        self.save_btn.setObjectName("SubtleButton")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_result)
        footer.addWidget(self.image_meta)
        footer.addStretch(1)
        footer.addWidget(self.open_overlay_btn)
        footer.addWidget(self.save_btn)
        layout.addLayout(footer)
        return stage

    def _build_sidebar(self) -> QtWidgets.QScrollArea:
        side = QtWidgets.QScrollArea()
        side.setWidgetResizable(True)
        side.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side.setObjectName("SideScroll")
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(2, 0, 2, 4)
        layout.setSpacing(12)

        settings_card = QtWidgets.QFrame()
        settings_card.setObjectName("Card")
        card = QtWidgets.QVBoxLayout(settings_card)
        card.setContentsMargins(17, 15, 17, 17)
        card.setSpacing(10)
        card.addWidget(section_label("01", "翻译设置", "捕获方式、上下文与版面控制"))
        card.addWidget(self._build_mode_tabs())

        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.source_combo = QtWidgets.QComboBox()
        self.source_combo.addItems(["自动检测", "日本語", "English", "简体中文", "한국어"])
        self.source_combo.setProperty("values", ["auto", "ja", "en", "zh", "ko"])
        self.target_combo = QtWidgets.QComboBox()
        self.target_combo.addItems(["简体中文", "English", "日本語", "한국어"])
        self.target_combo.setProperty("values", ["zh", "en", "ja", "ko"])
        row.addLayout(labeled("源语言", self.source_combo))
        row.addLayout(labeled("目标语言", self.target_combo))
        card.addLayout(row)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(
            ["自动场景", "漫画 / 条漫", "游戏对话", "轻小说插画", "软件界面"]
        )
        self.mode_combo.setProperty("values", ["auto", "manga", "game", "novel", "ui"])
        card.addLayout(labeled("场景模式", self.mode_combo))

        self.order_combo = QtWidgets.QComboBox()
        self.order_combo.addItems(["自动判断", "从左到右 · 横向", "从右到左 · 漫画"])
        self.order_combo.setProperty("values", ["auto", "ltr", "rtl"])
        card.addLayout(labeled("阅读顺序", self.order_combo))

        advanced_row = QtWidgets.QHBoxLayout()
        self.dilate_spin = QtWidgets.QSpinBox()
        self.dilate_spin.setRange(3, 7)
        self.dilate_spin.setSingleStep(2)
        self.dilate_spin.setValue(7)
        self.dilate_spin.setSuffix(" px")
        self.font_scale_spin = QtWidgets.QDoubleSpinBox()
        self.font_scale_spin.setRange(0.6, 1.4)
        self.font_scale_spin.setSingleStep(0.05)
        self.font_scale_spin.setValue(1.0)
        self.font_scale_spin.setSuffix("×")
        advanced_row.addLayout(labeled("蒙版膨胀", self.dilate_spin))
        advanced_row.addLayout(labeled("字号缩放", self.font_scale_spin))
        card.addLayout(advanced_row)

        self.translate_check = QtWidgets.QCheckBox("调用 DeepSeek 翻译")
        self.translate_check.setChecked(True)
        card.addWidget(self.translate_check)

        advanced = QtWidgets.QFrame()
        advanced.setObjectName("AdvancedBox")
        advanced_layout = QtWidgets.QVBoxLayout(advanced)
        advanced_layout.setContentsMargins(0, 6, 0, 0)
        advanced_layout.setSpacing(7)
        key_label = QtWidgets.QLabel("全局框选热键")
        key_label.setObjectName("FieldLabel")
        advanced_layout.addWidget(key_label)
        key_row = QtWidgets.QHBoxLayout()
        key_row.setSpacing(6)
        self.hotkey_edit = QtWidgets.QKeySequenceEdit(
            QtGui.QKeySequence(self.hotkey.sequence)
        )
        self.hotkey_edit.setMaximumSequenceLength(1)
        self.hotkey_apply_btn = QtWidgets.QPushButton("应用")
        self.hotkey_apply_btn.setObjectName("MonitorButton")
        self.hotkey_apply_btn.clicked.connect(self._apply_hotkey)
        key_row.addWidget(self.hotkey_edit, 1)
        key_row.addWidget(self.hotkey_apply_btn)
        advanced_layout.addLayout(key_row)
        card.addWidget(advanced)

        self.capture_btn = QtWidgets.QPushButton("")
        self.capture_btn.setObjectName("PrimaryButton")
        self.capture_btn.setFixedHeight(42)
        self.capture_btn.clicked.connect(self._on_primary_action)
        card.addWidget(self.capture_btn)

        self.upload_btn = QtWidgets.QPushButton("上传一张截图")
        self.upload_btn.setObjectName("SecondaryButton")
        self.upload_btn.setFixedHeight(34)
        self.upload_btn.clicked.connect(self.choose_file)
        card.addWidget(self.upload_btn)

        self.provider_label = QtWidgets.QLabel("正在检查翻译配置…")
        self.provider_label.setObjectName("ProviderNotice")
        self.provider_label.setWordWrap(True)
        card.addWidget(self.provider_label)
        layout.addWidget(settings_card)

        layout.addWidget(self._build_monitor_card())
        layout.addWidget(self._build_result_card(), 1)
        side.setWidget(content)
        return side

    def _build_mode_tabs(self) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("ModeTabs")
        layout = QtWidgets.QHBoxLayout(frame)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)
        self.mode_group = QtWidgets.QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self.mode_buttons: dict[str, QtWidgets.QPushButton] = {}
        for key, label in MODE_LABELS.items():
            button = QtWidgets.QPushButton(label)
            button.setObjectName("ModeTab")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, value=key: self._set_mode(value))
            self.mode_group.addButton(button)
            layout.addWidget(button, 1)
            self.mode_buttons[key] = button
        self.mode_buttons["manual"].setChecked(True)
        return frame

    def _build_monitor_card(self) -> QtWidgets.QFrame:
        self.monitor_card = QtWidgets.QFrame()
        self.monitor_card.setObjectName("MonitorCard")
        layout = QtWidgets.QVBoxLayout(self.monitor_card)
        layout.setContentsMargins(17, 15, 17, 16)
        layout.setSpacing(9)
        layout.addWidget(
            section_label("02", "区域自动监控", "锁定区域 · 防抖 · 冷却 · 自动刷新")
        )
        self.monitor_status = QtWidgets.QLabel("选择监控区域后开始")
        self.monitor_status.setObjectName("MonitorStatus")
        self.monitor_status.setWordWrap(True)
        layout.addWidget(self.monitor_status)
        self.monitor_coords = QtWidgets.QLabel("REGION —")
        self.monitor_coords.setObjectName("MonitorCoords")
        layout.addWidget(self.monitor_coords)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        self.sensitivity_slider, self.sensitivity_output = self._monitor_slider(
            grid, 0, 0, "变化灵敏度", 1, 30, 5, "%"
        )
        self.poll_slider, self.poll_output = self._monitor_slider(
            grid, 0, 1, "采样频率", 100, 1000, 200, " ms", step=50
        )
        self.debounce_slider, self.debounce_output = self._monitor_slider(
            grid, 1, 0, "稳定防抖", 150, 1500, 400, " ms", step=50
        )
        self.cooldown_slider, self.cooldown_output = self._monitor_slider(
            grid, 1, 1, "请求冷却", 500, 10000, 2000, " ms", step=250
        )
        layout.addLayout(grid)

        self.monitor_auto_overlay = QtWidgets.QCheckBox("检测到变化时自动弹出译文浮窗")
        self.monitor_auto_overlay.setChecked(True)
        layout.addWidget(self.monitor_auto_overlay)
        self.monitor_minimize = QtWidgets.QCheckBox("开始监控时隐藏主窗口到托盘")
        self.monitor_minimize.setChecked(True)
        layout.addWidget(self.monitor_minimize)

        actions = QtWidgets.QHBoxLayout()
        actions.setSpacing(5)
        self.monitor_start_btn = QtWidgets.QPushButton("开始监控")
        self.monitor_start_btn.setObjectName("MonitorButtonPrimary")
        self.monitor_start_btn.clicked.connect(lambda: self._start_monitor(translate_now=True))
        self.monitor_pause_btn = QtWidgets.QPushButton("暂停")
        self.monitor_pause_btn.setObjectName("MonitorButton")
        self.monitor_pause_btn.clicked.connect(self._toggle_monitor_pause)
        self.monitor_stop_btn = QtWidgets.QPushButton("停止")
        self.monitor_stop_btn.setObjectName("MonitorButton")
        self.monitor_stop_btn.clicked.connect(self._stop_monitor)
        self.monitor_now_btn = QtWidgets.QPushButton("立即翻译")
        self.monitor_now_btn.setObjectName("MonitorButton")
        self.monitor_now_btn.clicked.connect(self._translate_monitor_now)
        actions.addWidget(self.monitor_start_btn, 1)
        actions.addWidget(self.monitor_pause_btn)
        actions.addWidget(self.monitor_stop_btn)
        actions.addWidget(self.monitor_now_btn)
        layout.addLayout(actions)
        note = QtWidgets.QLabel(
            "仅比对灰度降采样帧；画面稳定并越过冷却后才提交翻译。"
        )
        note.setObjectName("FieldLabel")
        note.setWordWrap(True)
        layout.addWidget(note)
        for slider in (
            self.sensitivity_slider,
            self.poll_slider,
            self.debounce_slider,
            self.cooldown_slider,
        ):
            slider.valueChanged.connect(self._on_monitor_setting_changed)
        self._update_monitor_outputs()
        return self.monitor_card

    def _monitor_slider(
        self,
        layout: QtWidgets.QGridLayout,
        row: int,
        column: int,
        title: str,
        minimum: int,
        maximum: int,
        value: int,
        suffix: str,
        *,
        step: int = 1,
    ) -> tuple[QtWidgets.QSlider, QtWidgets.QLabel]:
        widget, slider, output = slider_row(
            title=title,
            minimum=minimum,
            maximum=maximum,
            value=value,
            suffix=suffix,
            step=step,
        )
        layout.addWidget(widget, row, column)
        output.setProperty("suffix", suffix)
        return slider, output

    def _build_result_card(self) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setObjectName("Card")
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(17, 15, 17, 15)
        layout.setSpacing(8)
        layout.addWidget(section_label("03", "任务状态", "实时队列与翻译明细"))
        self.progress = QtWidgets.QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        self.status_label = QtWidgets.QLabel("等待截图")
        self.status_label.setObjectName("StatusLabel")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.warning_label = QtWidgets.QLabel()
        self.warning_label.setObjectName("WarningLabel")
        self.warning_label.setWordWrap(True)
        self.warning_label.hide()
        layout.addWidget(self.warning_label)
        self.translation_view = QtWidgets.QTextBrowser()
        self.translation_view.setObjectName("TranslationView")
        self.translation_view.setMinimumHeight(210)
        self.translation_view.setHtml(
            "<div style='color:#747782;font-size:11px;line-height:1.6'>"
            "完成一次翻译后，这里会显示原文与译文的逐行对照。</div>"
        )
        layout.addWidget(self.translation_view, 1)
        return card

    # ------------------------------------------------------------------ tray
    def _build_tray(self) -> None:
        if not QtWidgets.QSystemTrayIcon.isSystemTrayAvailable():
            self.tray = None
            return
        self.tray = QtWidgets.QSystemTrayIcon(self.windowIcon(), self)
        self.tray.setToolTip(f"O-Translator · {self.hotkey.display} 区域翻译")
        menu = QtWidgets.QMenu()
        self.select_area_action = menu.addAction("框选区域并翻译")
        self.select_area_action.triggered.connect(self.start_area_capture)
        self.monitor_action = menu.addAction("开始 / 重选自动监控")
        self.monitor_action.triggered.connect(self._tray_monitor_action)
        menu.addSeparator()
        self.pause_action = menu.addAction("暂停自动监控")
        self.pause_action.setCheckable(True)
        self.pause_action.toggled.connect(self._on_pause_action)
        self.stop_action = menu.addAction("停止自动监控")
        self.stop_action.triggered.connect(self._stop_monitor)
        menu.addSeparator()
        self.click_through_action = menu.addAction("浮窗鼠标穿透")
        self.click_through_action.setToolTip(
            "区域监控默认开启；区域截图默认关闭，可直接拖动/保存浮窗。"
        )
        self.click_through_action.setCheckable(True)
        self.click_through_action.setEnabled(False)
        self.click_through_action.toggled.connect(self._on_click_through_action)
        menu.addSeparator()
        show_action = menu.addAction("打开主窗口")
        show_action.triggered.connect(self._restore_window)
        quit_action = menu.addAction("退出")
        quit_action.triggered.connect(self._quit)
        self.tray.setContextMenu(menu)
        self.tray_menu = menu
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()
        self._sync_tray_actions()

    def _tray_activated(
        self, reason: QtWidgets.QSystemTrayIcon.ActivationReason
    ) -> None:
        if reason == QtWidgets.QSystemTrayIcon.ActivationReason.DoubleClick:
            self._restore_window()

    def _notify_tray(
        self,
        message: str,
        *,
        warning: bool = False,
        timeout_ms: int = 2200,
    ) -> None:
        """Surface a short status without opening the workbench."""
        if self.tray is None:
            return
        icon = (
            QtWidgets.QSystemTrayIcon.MessageIcon.Warning
            if warning
            else QtWidgets.QSystemTrayIcon.MessageIcon.Information
        )
        self.tray.showMessage(APP_NAME, message, icon, timeout_ms)

    def notify_started(self) -> None:
        """Announce a tray-only launch.

        The app deliberately starts without showing the main workbench so a
        login-time launch or a double-click does not cover the desktop. This
        balloon is the only feedback; F2 and translation results appear in the
        floating HUD.
        """
        self._notify_tray(
            f"已在托盘运行。按 {self.hotkey.display} 框选翻译，"
            "需要工作台时打开托盘菜单。",
            timeout_ms=3200,
        )

    def _tray_monitor_action(self) -> None:
        if self._monitor_running:
            self._restore_window()
            self._set_mode("monitor")
            return
        if self._monitor_region is None:
            self.select_monitor_region()
        else:
            self._start_monitor(translate_now=True)

    def _on_pause_action(self, checked: bool) -> None:
        if not self._monitor_running:
            self._sync_tray_actions()
            return
        self._set_monitor_paused(checked)

    # ------------------------------------------------------------------ hotkey
    def _register_hotkey(self) -> None:
        if sys.platform != "win32":
            self._hotkey_active = False
            self.hotkey_label.setText("热键仅支持 Windows")
            return
        if self.hotkey.register():
            self._hotkey_active = True
            self.hotkey_label.setText(self.hotkey.display)
        else:
            self._hotkey_active = False
            self.hotkey_label.setText("热键被占用")
            self.show_status(
                "全局热键注册失败，可能已被其他程序占用。可点击按钮框选。",
                warning=True,
            )

    def _apply_hotkey(self) -> None:
        sequence = self.hotkey_edit.keySequence().toString().strip().lower()
        if not sequence:
            QMessageBoxCompat.warning(self, APP_NAME, "请先按下一组快捷键。")
            return
        if sequence == self.hotkey.sequence and self._hotkey_active:
            self.show_status("热键未变化。")
            return
        old = self.hotkey
        old.unregister()
        candidate = GlobalHotkey(sequence, self)
        candidate.activated.connect(self._on_hotkey_activated)
        if candidate.register():
            self.hotkey = candidate
            self._hotkey_active = True
            self.hotkey_label.setText(candidate.display)
            self.settings.setValue("hotkey", candidate.sequence)
            self.settings.setValue("hotkey_version", HOTKEY_SETTINGS_VERSION)
            if self.tray is not None:
                self.tray.setToolTip(f"O-Translator · {candidate.display} 区域翻译")
            self._sync_mode_ui()
            self.show_status(f"全局热键已更新为 {candidate.display}。")
        else:
            old.register()
            self.hotkey = old
            self.hotkey_edit.setKeySequence(QtGui.QKeySequence(old.sequence))
            QMessageBoxCompat.warning(
                self,
                APP_NAME,
                "该快捷键注册失败，可能已被其他程序占用。",
            )

    # ------------------------------------------------------------------ modes
    def _set_mode(self, mode: str) -> None:
        if mode not in MODE_LABELS:
            mode = "manual"
        self._mode = mode
        button = self.mode_buttons.get(mode)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        self._sync_mode_ui()
        self._sync_monitor_ui()

    def _sync_mode_ui(self) -> None:
        self.upload_btn.setVisible(self._mode == "manual")
        if self._mode == "manual":
            self.capture_btn.setText("选择截图文件")
        elif self._mode == "area":
            self.capture_btn.setText(f"框选屏幕并翻译    {self.hotkey.display}")
        else:
            self.capture_btn.setText(
                "重新选择监控区域" if self._monitor_region else "选择监控区域"
            )
            if self._monitor_region is None and not self._monitor_running:
                self._set_monitor_status("选择监控区域后开始", "info")

    def _on_primary_action(self) -> None:
        if self._mode == "manual":
            self.choose_file()
        elif self._mode == "area":
            self.start_area_capture()
        else:
            self.select_monitor_region()

    # ------------------------------------------------------------------ backend
    def _bootstrap_backend(self) -> None:
        self._set_health("连接中…", "muted")
        worker = Worker(self.backend.ensure)
        worker.signals.result.connect(self._on_backend_ready)
        worker.signals.error.connect(self._on_backend_error)
        self.pool.start(worker)

    def _on_backend_ready(self, health: dict[str, Any]) -> None:
        translation = health.get("engines", {}).get("translation", {})
        if translation.get("deepseek_configured"):
            self._set_health("DeepSeek 已连接", "ok")
            self.provider_label.setText(
                f"DeepSeek {translation.get('model', 'V3')} 已连接 · 整页上下文批量翻译"
            )
            self.provider_label.setProperty("tone", "ok")
        else:
            self._set_health("离线模式", "warn")
            self.provider_label.setText(
                "未配置 DEEPSEEK_API_KEY：可完成 OCR、修复与排版，但会保留原文。"
            )
            self.provider_label.setProperty("tone", "warn")
        self._repolish(self.provider_label)
        queue = health.get("queue", {})
        self.queue_label.setText(
            f"QUEUE {queue.get('queued', 0)} · ACTIVE {queue.get('active', 0)} "
            f"· WORKERS {queue.get('workers', 0)}"
        )
        self.show_status("后端就绪。可手动上传、按热键框选，或开启区域监控。")

    def _on_backend_error(self, message: str) -> None:
        self._set_health("服务离线", "error")
        self.provider_label.setText(message)
        self.provider_label.setProperty("tone", "error")
        self._repolish(self.provider_label)
        self.show_status(message, warning=True)

    def _set_health(self, text: str, tone: str) -> None:
        colors = {
            "ok": ("#75e6d3", "rgba(117,230,211,.10)"),
            "warn": ("#f4b95f", "rgba(244,185,95,.10)"),
            "error": ("#ff7d86", "rgba(255,125,134,.10)"),
            "muted": ("#8b8e98", "rgba(255,255,255,.03)"),
        }
        color, background = colors.get(tone, colors["muted"])
        self.health_label.setText(f"●  {text}")
        self.health_label.setStyleSheet(
            f"color:{color}; background:{background};"
            "border:1px solid rgba(255,255,255,.07); border-radius:11px;"
            "padding:5px 9px; font-size:10px;"
        )

    # ------------------------------------------------------------------ capture
    def _on_hotkey_activated(self) -> None:
        self.start_area_capture(from_hotkey=True)

    def start_area_capture(
        self, _checked: bool = False, *, from_hotkey: bool = False
    ) -> None:
        if self.hotkey_edit.hasFocus():
            return
        # The overlay opens after a short compositor-settling delay. Freeze the
        # pointer position now so the first highlight is based on where the user
        # actually pressed the hotkey, not on where the mouse drifted meanwhile.
        self._selection_cursor = (
            QtGui.QCursor.pos() if from_hotkey else None
        )
        self._begin_selection("area")

    def select_monitor_region(self) -> None:
        self._selection_cursor = None
        self._begin_selection("monitor")

    def _begin_selection(self, context: str) -> None:
        if self._busy:
            # A tray/hotkey capture must never surface the workbench just
            # because the previous job is still running.
            message = "当前任务仍在处理中，请稍候。"
            self.show_status(message, warning=True)
            self._notify_tray(message)
            return
        if self._selection_pending:
            # A second F2/tray click during the 170 ms compositor-settling
            # delay must not schedule a second overlay.
            return
        if self._selector is not None:
            try:
                if self._selector.is_open:
                    return
            except RuntimeError:
                self._selector = None
        self._selection_context = context
        self._restore_after_capture = self.isVisible()
        if self._monitor_running:
            self.monitor.suspend_capture(True, "区域选择中，自动监控暂缓")
        if self._floating is not None:
            self._floating_was_visible = self._floating.isVisible()
            self._floating.hide()
            self.monitor.set_exclusions([])
        else:
            self._floating_was_visible = False
        self._sync_tray_actions()
        self.hide()
        self._selection_pending = True
        QTimer.singleShot(170, self._show_selector)

    def _show_selector(self) -> None:
        self._selection_pending = False
        if self._force_quit:
            return
        if self._selector is not None:
            try:
                if self._selector.is_open:
                    return
            except RuntimeError:
                self._selector = None
        selector = OverlaySnipper(mode=self._selection_context)
        self._selector = selector
        selector.captured.connect(self._on_region_captured)
        selector.cancelled.connect(self._on_selection_cancelled)
        selector.failed.connect(self._on_selection_failed)
        selector.saved.connect(self._on_selection_saved)
        cursor = self._selection_cursor
        self._selection_cursor = None
        selector.open(cursor)

    def _finish_selection_restore(self, *, restore: bool = True) -> None:
        if restore and self._restore_after_capture:
            self._restore_window()
        self._restore_after_capture = False

    def _on_selection_cancelled(self) -> None:
        self._selection_pending = False
        self._selector = None
        self._finish_selection_restore()
        self._resume_monitor()
        if self._floating_was_visible and self._floating is not None:
            self._floating.show()
            self._update_monitor_exclusion()
            self._sync_tray_actions()
        self._floating_was_visible = False

    def _on_selection_saved(self, path: str) -> None:
        self._selection_pending = False
        self._selector = None
        self._finish_selection_restore()
        self._resume_monitor()
        if self._floating_was_visible and self._floating is not None:
            self._floating.show()
            self._update_monitor_exclusion()
            self._sync_tray_actions()
        self._floating_was_visible = False
        self.show_status(f"截图已保存：{path}")
        if self.tray is not None:
            self.tray.showMessage(
                APP_NAME,
                f"截图已保存：{Path(path).name}",
                QtWidgets.QSystemTrayIcon.MessageIcon.Information,
                2200,
            )

    def _on_selection_failed(self, message: str) -> None:
        self._selection_pending = False
        self._selector = None
        self._finish_selection_restore()
        self._resume_monitor()
        if self._floating_was_visible and self._floating is not None:
            self._floating.show()
            self._update_monitor_exclusion()
            self._sync_tray_actions()
        self._floating_was_visible = False
        self.show_status(f"截图失败：{message}", warning=True)
        QMessageBoxCompat.warning(self, APP_NAME, f"截图失败：{message}")

    def _on_region_captured(self, result: CaptureResult) -> None:
        self._selection_pending = False
        self._selector = None
        if (
            self._selection_context == "monitor"
            and self.monitor_minimize.isChecked()
        ):
            self._restore_after_capture = False
        # A screenshot must never bring the workbench back. The result appears
        # in the floating HUD; the main window stays hidden until the user opens
        # it from the tray.
        self._finish_selection_restore(restore=False)
        self._floating_was_visible = False
        self.canvas.set_image(result.image)
        if self._selection_context == "monitor":
            meta_label = "MONITOR"
        else:
            meta_label = {
                "window": "WINDOW",
                "fullscreen": "FULLSCREEN",
            }.get(result.source, "CAPTURE")
        self.image_meta.setText(
            f"{result.region.logical_width} × {result.region.logical_height}  ·  "
            f"{meta_label}"
        )
        if self._selection_context == "monitor":
            self._set_monitor_region(result.region)
            self._set_mode("monitor")
            self._start_monitor(initial_image=result.image)
        else:
            self._set_mode("area")
            self._dispatch_image(
                result.image,
                result.region,
                source=result.source,
                background=True,
                auto_overlay=True,
            )
            if not self._busy:
                self._resume_monitor()

    def choose_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "选择截图",
            str(Path.home()),
            "图片 (*.png *.jpg *.jpeg *.webp *.bmp)",
        )
        if path:
            self.load_file(Path(path), submit=True)

    def _paste_clipboard(self) -> None:
        if self._busy:
            self.show_status("当前任务仍在处理中，请稍候。", warning=True)
            return
        image = QtGui.QGuiApplication.clipboard().image()
        if image.isNull():
            self.show_status("剪贴板中没有图片。", warning=True)
            return
        directory = ensure_capture_dir()
        path = directory / f"clipboard-{int(time.time() * 1000)}.png"
        if not image.save(str(path), "PNG"):
            self.show_status("无法保存剪贴板图片。", warning=True)
            return
        self.load_file(path, submit=True)

    def _retranslate(self) -> None:
        if self.last_file is None or self._busy:
            return
        self.load_file(self.last_file, submit=True)

    def load_file(self, path: Path, *, submit: bool = True) -> None:
        if self._busy:
            self.show_status("当前任务仍在处理中，请稍候。", warning=True)
            return
        image = QtGui.QImage(str(path))
        if image.isNull():
            QMessageBoxCompat.warning(self, APP_NAME, "无法读取该图片。")
            return
        self.last_file = path
        self.canvas.set_image(image)
        self.image_meta.setText(
            f"{image.width()} × {image.height()}  ·  {path.name}"
        )
        self._restore_window()
        if submit:
            self._job_region = None
            self._job_source = "upload"
            self._job_background = False
            self._job_auto_overlay = False
            self._submit(path)

    # ------------------------------------------------------------------ jobs
    def _dispatch_image(
        self,
        image: QtGui.QImage,
        region: CaptureRegion | None,
        *,
        source: str,
        background: bool,
        auto_overlay: bool,
    ) -> None:
        if image.isNull():
            return
        if source == "monitor":
            self._suspend_monitor()
        if self._busy:
            if source == "monitor":
                self._pending_monitor_image = image.copy()
                self._set_monitor_status(
                    "检测到新画面，等待当前翻译任务完成…", "info"
                )
            else:
                self.show_status("当前任务仍在处理中，请稍候。", warning=True)
                self._resume_monitor()
            return
        directory = ensure_capture_dir()
        safe_source = (
            source
            if source in {"selection", "fullscreen", "monitor", "window"}
            else "capture"
        )
        path = directory / f"{safe_source}-{int(time.time() * 1000)}.png"
        if not image.save(str(path), "PNG"):
            self.show_status("无法保存临时截图。", warning=True)
            self._resume_monitor()
            return
        self.last_file = path
        self._job_region = region
        self._job_source = source
        self._job_background = background
        self._job_auto_overlay = auto_overlay
        self.canvas.set_image(image)
        self.image_meta.setText(
            f"{image.width()} × {image.height()}  ·  {source.upper()}"
        )
        self._submit(path)

    def _submit(self, path: Path) -> None:
        if self._busy:
            return
        self._busy = True
        self.current_result = None
        self.result_image = None
        self.preview_image = None
        self._preview_job_id = None
        self._preview_inflight = False
        self.current_job_id = None
        self._poll_failures = 0
        self.open_overlay_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.progress.setValue(2)
        self.warning_label.hide()
        self.translation_view.setHtml(
            "<div style='color:#747782;font-size:11px;line-height:1.6'>"
            "正在处理截图…</div>"
        )
        self._set_busy_controls(True)
        self.show_status("正在上传截图并创建任务…")
        worker = Worker(
            self.api.create_job,
            path,
            source_lang=self._combo_value(self.source_combo, "auto"),
            target_lang=self._combo_value(self.target_combo, "zh"),
            mode=self._combo_value(self.mode_combo, "auto"),
            reading_order=self._combo_value(self.order_combo, "auto"),
            dilate_px=self.dilate_spin.value(),
            font_scale=self.font_scale_spin.value(),
            translate=self.translate_check.isChecked(),
        )
        worker.signals.result.connect(self._on_job_created)
        worker.signals.error.connect(self._on_job_error)
        self.pool.start(worker)

    def _on_job_created(self, job: dict[str, Any]) -> None:
        self.current_job_id = job.get("id")
        self.progress.setValue(max(3, int(job.get("progress", 0))))
        self.show_status(STAGE_TEXT.get(job.get("stage", "queued"), "任务已排队"))
        self.poll_timer.start()

    def _poll_job(self) -> None:
        if self._poll_inflight or not self.current_job_id:
            return
        self._poll_inflight = True
        job_id = self.current_job_id
        worker = Worker(self.api.get_job, job_id)
        worker.signals.result.connect(self._on_job_state)
        worker.signals.error.connect(self._on_poll_error)
        worker.signals.done.connect(self._poll_done)
        self.pool.start(worker)

    def _poll_done(self) -> None:
        self._poll_inflight = False

    def _on_poll_error(self, message: str) -> None:
        self._poll_failures += 1
        if self._poll_failures >= 4:
            self.poll_timer.stop()
            self._on_job_error(message)
        else:
            self.show_status(f"暂时无法获取状态，正在重试… ({message})", warning=True)

    def _on_job_state(self, job: dict[str, Any]) -> None:
        if job.get("id") != self.current_job_id:
            return
        self._poll_failures = 0
        stage = job.get("stage", "queued")
        progress = int(job.get("progress", 0))
        self.progress.setValue(max(self.progress.value(), progress))
        self.show_status(
            f"{STAGE_TEXT.get(stage, '处理中')} · {job.get('message', '')}".strip(" ·")
        )
        if job.get("status") in {"queued", "running"}:
            self._request_preview(job)
        if job.get("status") == "completed":
            self.poll_timer.stop()
            self.current_result = job.get("result")
            self._on_complete(job)
        elif job.get("status") in {"failed", "cancelled"}:
            self.poll_timer.stop()
            error = job.get("error") or {}
            self._on_job_error(
                error.get("message") or job.get("message") or "任务失败"
            )

    def _request_preview(self, job: dict[str, Any]) -> None:
        """Fetch pass-1 as soon as the backend publishes it."""
        if self._preview_inflight or self.preview_image is not None:
            return
        preview = job.get("preview") or {}
        url = preview.get("image_url")
        job_id = job.get("id")
        if not url or not job_id or job_id != self.current_job_id:
            return
        self._preview_inflight = True
        self._preview_job_id = job_id
        worker = Worker(self.api.fetch_image, url)
        worker.signals.result.connect(
            lambda data, jid=job_id: self._on_preview_downloaded(jid, data)
        )
        worker.signals.error.connect(self._on_preview_error)
        worker.signals.done.connect(self._on_preview_done)
        self.pool.start(worker)

    def _on_preview_done(self) -> None:
        self._preview_inflight = False

    def _on_preview_error(self, message: str) -> None:
        # Preview is an optimization; the final result is still on its way.
        self._preview_inflight = False
        self.show_status(f"预览图暂时不可用，继续等待最终结果… ({message})", warning=True)

    def _on_preview_downloaded(self, job_id: str, data: bytes) -> None:
        if job_id != self.current_job_id:
            return
        # A slow preview fetch can finish after the final result; never replace
        # the finished overlay with the lower-quality pass-1 image.
        if self.result_image is not None:
            return
        image = QtGui.QImage.fromData(data)
        if image.isNull():
            return
        self.preview_image = image
        self.canvas.set_image(image)
        self.image_meta.setText(
            f"{image.width()} × {image.height()}  ·  PREVIEW"
        )
        if self._job_auto_overlay:
            self._show_floating_result(
                image,
                self._job_region,
                self._job_source,
                activate=False,
            )
        self.progress.setValue(88)
        self.show_status("译文预览已显示，正在精修背景…")

    def _on_complete(self, job: dict[str, Any]) -> None:
        result = job.get("result") or {}
        url = result.get("image_url")
        if not url:
            self._on_job_error("任务完成但未返回图片。")
            return
        self.show_status("正在下载译文成图…")
        worker = Worker(self.api.fetch_image, url)
        worker.signals.result.connect(self._on_image_downloaded)
        worker.signals.error.connect(self._on_job_error)
        self.pool.start(worker)

    def _on_image_downloaded(self, data: bytes) -> None:
        image = QtGui.QImage.fromData(data)
        if image.isNull():
            self._on_job_error("结果图片损坏。")
            return
        self.result_image = image
        self._last_result_image = image
        self._last_result_region = self._job_region
        self._last_result_source = self._job_source
        self.canvas.set_image(image)
        self.open_overlay_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self._busy = False
        self._set_busy_controls(False)
        self._render_translation()
        warnings = (self.current_result or {}).get("warnings") or []
        if warnings:
            self.warning_label.setText("  ·  ".join(warnings[:2]))
            self.warning_label.show()
        self.progress.setValue(100)
        self.show_status("翻译完成。")
        if self.current_result and self.current_result.get("width"):
            self.image_meta.setText(
                f"{self.current_result['width']} × {self.current_result['height']}"
                "  ·  TRANSLATED"
            )
        if self._job_auto_overlay:
            self._show_floating_result(
                image,
                self._job_region,
                self._job_source,
                activate=not self._job_background,
                fade=self.preview_image is not None,
            )
        self.preview_image = None
        self._preview_job_id = None
        if not self._job_background:
            self._restore_window()
        had_pending = self._pending_monitor_image is not None
        self._flush_pending_monitor()
        if not had_pending:
            self._resume_monitor()

    def _on_job_error(self, message: str) -> None:
        self._busy = False
        self.poll_timer.stop()
        self._poll_inflight = False
        self.preview_image = None
        self._preview_job_id = None
        self._preview_inflight = False
        self.current_job_id = None
        self._set_busy_controls(False)
        self.progress.setValue(0)
        self.warning_label.setText(message)
        self.warning_label.show()
        self.show_status("任务失败。", warning=True)
        if not self._job_background:
            self._restore_window()
            QMessageBoxCompat.warning(self, "翻译失败", message)
        elif self.tray is not None:
            self.tray.showMessage(
                APP_NAME,
                f"翻译失败：{message}",
                QtWidgets.QSystemTrayIcon.MessageIcon.Warning,
                3500,
            )
        had_pending = self._pending_monitor_image is not None
        self._flush_pending_monitor(delay_ms=1200)
        if not had_pending:
            self._resume_monitor()

    def _set_busy_controls(self, busy: bool) -> None:
        self.capture_btn.setEnabled(not busy)
        self.upload_btn.setEnabled(not busy)
        self.source_combo.setEnabled(not busy)
        self.target_combo.setEnabled(not busy)
        self.mode_combo.setEnabled(not busy)
        self.order_combo.setEnabled(not busy)
        self.dilate_spin.setEnabled(not busy)
        self.font_scale_spin.setEnabled(not busy)
        self.translate_check.setEnabled(not busy)
        self.monitor_now_btn.setEnabled(not busy and self._monitor_region is not None)
        self.monitor_start_btn.setEnabled(not busy)

    def _flush_pending_monitor(self, delay_ms: int = 80) -> None:
        if self._pending_monitor_image is None:
            return
        if not self._monitor_running or self._monitor_paused:
            self._pending_monitor_image = None
            return
        image = self._pending_monitor_image
        self._pending_monitor_image = None

        def send() -> None:
            if not self._monitor_running or self._monitor_paused:
                return
            self._dispatch_image(
                image,
                self._monitor_region,
                source="monitor",
                background=True,
                auto_overlay=self.monitor_auto_overlay.isChecked(),
            )

        QTimer.singleShot(delay_ms, send)

    # ------------------------------------------------------------------ monitor
    def _bind_monitor(self) -> None:
        self.monitor.triggered.connect(self._on_monitor_triggered)
        self.monitor.stats.connect(self._on_monitor_stats)
        self.monitor.status.connect(self._on_monitor_status_message)
        self.monitor.error.connect(self._on_monitor_error)
        self.monitor.finished_monitoring.connect(self._on_monitor_finished)

    def _monitor_settings(self) -> MonitorSettings:
        return MonitorSettings(
            threshold_percent=self.sensitivity_slider.value(),
            poll_interval_ms=self.poll_slider.value(),
            debounce_ms=self.debounce_slider.value(),
            cooldown_ms=self.cooldown_slider.value(),
        ).normalized()

    def _set_monitor_region(self, region: CaptureRegion) -> None:
        self._monitor_region = region
        self.settings.setValue("monitor_region", json.dumps(region.to_dict()))
        self.monitor_coords.setText(
            f"X {region.logical_left} · Y {region.logical_top} · "
            f"{region.logical_width}×{region.logical_height} · "
            f"{region.screen_name or 'screen'} · {region.scale_x:.2f}×"
        )
        if not self._monitor_running:
            self._set_monitor_status("监控区域已锁定，点击「开始监控」", "info")
        self._sync_monitor_ui()
        self._sync_mode_ui()

    def _start_monitor(
        self,
        *,
        initial_image: QtGui.QImage | None = None,
        translate_now: bool = False,
    ) -> None:
        if self._monitor_region is None:
            self.select_monitor_region()
            return
        try:
            region = refresh_region(self._monitor_region)
        except CaptureError as exc:
            self._set_monitor_status(str(exc), "error")
            return
        self._monitor_region = region
        self._set_monitor_region(region)
        self.monitor.start_monitoring(region, self._monitor_settings())
        self._monitor_running = True
        self._monitor_paused = False
        self._pending_monitor_image = None
        self._set_mode("monitor")
        self._set_monitor_status("监控中 · 等待画面变化", "info")
        self._sync_monitor_ui()
        self._sync_tray_actions()
        self._update_monitor_exclusion()
        if self.monitor_minimize.isChecked() and self.isVisible():
            self.hide()
            if self.tray is not None:
                self.tray.showMessage(
                    APP_NAME,
                    f"自动监控已启动。{self.hotkey.display} 仍可随时框选翻译。",
                    QtWidgets.QSystemTrayIcon.MessageIcon.Information,
                    2200,
                )
        if initial_image is not None:
            self._suspend_monitor()
            self._dispatch_image(
                initial_image,
                region,
                source="monitor",
                background=True,
                auto_overlay=self.monitor_auto_overlay.isChecked(),
            )
        elif translate_now:
            self._translate_monitor_now()

    def _stop_monitor(self) -> None:
        self.monitor.suspend_capture(False)
        self.monitor.stop_monitoring()
        self._monitor_running = False
        self._monitor_paused = False
        self._pending_monitor_image = None
        self.monitor.set_exclusions([])
        self._set_monitor_status("自动监控已停止", "info")
        self._sync_monitor_ui()
        self._sync_tray_actions()

    def _toggle_monitor_pause(self) -> None:
        if not self._monitor_running:
            self._start_monitor(translate_now=True)
            return
        self._set_monitor_paused(not self._monitor_paused)

    def _set_monitor_paused(self, paused: bool) -> None:
        if not self._monitor_running:
            self._sync_tray_actions()
            return
        self._monitor_paused = paused
        self.monitor.set_paused(paused)
        if paused:
            self.monitor.suspend_capture(False)
            self._pending_monitor_image = None
            self._set_monitor_status("已暂停（低功耗）；点击继续恢复检测", "paused")
        else:
            self.monitor.suspend_capture(False)
            self._set_monitor_status("监控中 · 等待画面变化", "info")
        self._sync_monitor_ui()
        self._sync_tray_actions()

    def _translate_monitor_now(self) -> None:
        if self._busy:
            self.show_status("当前任务仍在处理中，请稍候。", warning=True)
            return
        if self._monitor_region is None:
            self.select_monitor_region()
            return
        try:
            region = refresh_region(self._monitor_region)
        except CaptureError as exc:
            self._set_monitor_status(str(exc), "error")
            return
        self._monitor_region = region
        self._set_monitor_region(region)
        self._suspend_monitor()
        worker = Worker(capture_region_image, region)
        worker.signals.result.connect(
            lambda image: self._dispatch_image(
                image,
                region,
                source="monitor",
                background=True,
                auto_overlay=self.monitor_auto_overlay.isChecked(),
            )
        )
        worker.signals.error.connect(
            lambda message: (
                self._set_monitor_status(message, "error"),
                self._resume_monitor(),
            )
        )
        self.pool.start(worker)

    def _on_monitor_triggered(self, image: QtGui.QImage, ratio: float) -> None:
        if not self._monitor_running:
            return
        self._suspend_monitor()
        self._set_monitor_status(f"检测到变化 {ratio:.1f}%，准备翻译", "info")
        if self._busy:
            self._pending_monitor_image = image.copy()
            self._set_monitor_status(
                f"变化 {ratio:.1f}% · 等待当前翻译任务完成", "info"
            )
            return
        self._dispatch_image(
            image,
            self._monitor_region,
            source="monitor",
            background=True,
            auto_overlay=self.monitor_auto_overlay.isChecked(),
        )

    def _on_monitor_stats(
        self, ratio: float, pending: bool, message: str
    ) -> None:
        if not self._monitor_running or self._monitor_paused:
            return
        if pending:
            self._set_monitor_status(message, "info")
        elif ratio > 0.05:
            self._set_monitor_status(f"监控中 · 变化 {ratio:.1f}%", "info")

    def _on_monitor_status_message(self, message: str) -> None:
        tone = "paused" if self._monitor_paused else "info"
        if "监控已停止" in message or "失败" in message:
            tone = "error"
        self._set_monitor_status(message, tone)

    def _on_monitor_error(self, message: str) -> None:
        self._set_monitor_status(message, "error")
        self.show_status(message, warning=True)

    def _on_monitor_finished(self) -> None:
        QTimer.singleShot(0, self._mark_monitor_stopped)

    def _mark_monitor_stopped(self) -> None:
        if self.monitor.isRunning():
            return
        if self._monitor_running:
            self._monitor_running = False
            self._monitor_paused = False
            self._set_monitor_status("自动监控已停止", "info")
            self._sync_monitor_ui()
            self._sync_tray_actions()

    def _on_monitor_setting_changed(self) -> None:
        self._update_monitor_outputs()
        if self._monitor_running:
            self.monitor.update_settings(self._monitor_settings())
            if not self._monitor_paused:
                self._set_monitor_status("监控参数已更新", "info")

    def _update_monitor_outputs(self) -> None:
        for slider, output in (
            (self.sensitivity_slider, self.sensitivity_output),
            (self.poll_slider, self.poll_output),
            (self.debounce_slider, self.debounce_output),
            (self.cooldown_slider, self.cooldown_output),
        ):
            suffix = str(output.property("suffix") or "")
            output.setText(f"{slider.value()}{suffix}")

    def _set_monitor_status(self, message: str, tone: str) -> None:
        self.monitor_status.setText(message)
        self.monitor_status.setProperty("tone", tone)
        self._repolish(self.monitor_status)

    def _sync_monitor_ui(self) -> None:
        show_card = self._mode == "monitor" or self._monitor_running
        self.monitor_card.setVisible(show_card)
        running = self._monitor_running
        paused = self._monitor_paused and running
        self.monitor_start_btn.setText("重新开始" if running else "开始监控")
        self.monitor_pause_btn.setText("继续" if paused else "暂停")
        self.monitor_pause_btn.setEnabled(running)
        self.monitor_stop_btn.setEnabled(running)
        self.monitor_now_btn.setEnabled(
            self._monitor_region is not None and not self._busy
        )
        self._sync_mode_ui()
        self._sync_tray_actions()

    def _sync_tray_actions(self) -> None:
        if getattr(self, "tray", None) is None:
            return
        self.pause_action.blockSignals(True)
        self.pause_action.setChecked(self._monitor_paused and self._monitor_running)
        self.pause_action.setEnabled(self._monitor_running)
        self.pause_action.setText("继续自动监控" if self._monitor_paused else "暂停自动监控")
        self.pause_action.blockSignals(False)
        self.stop_action.setEnabled(self._monitor_running)
        if self._monitor_running:
            self.monitor_action.setText("打开自动监控控制")
        elif self._monitor_region is not None:
            self.monitor_action.setText("开始区域自动监控")
        else:
            self.monitor_action.setText("选择自动监控区域")
        enabled = self._floating is not None and self._floating.isVisible()
        self.click_through_action.setEnabled(enabled)
        if enabled:
            self.click_through_action.blockSignals(True)
            self.click_through_action.setChecked(self._floating.click_through)
            self.click_through_action.blockSignals(False)

    def _load_monitor_preferences(self) -> None:
        raw = str(self.settings.value("monitor_settings", ""))
        if raw:
            try:
                settings = MonitorSettings.from_dict(json.loads(raw))
                self.sensitivity_slider.setValue(int(settings.threshold_percent))
                self.poll_slider.setValue(settings.poll_interval_ms)
                self.debounce_slider.setValue(settings.debounce_ms)
                self.cooldown_slider.setValue(settings.cooldown_ms)
            except (ValueError, TypeError, json.JSONDecodeError):
                pass
        self.monitor_auto_overlay.setChecked(
            str(self.settings.value("monitor_auto_overlay", "true")).lower()
            != "false"
        )
        self.monitor_minimize.setChecked(
            str(self.settings.value("monitor_minimize", "true")).lower() != "false"
        )
        raw_region = str(self.settings.value("monitor_region", ""))
        if raw_region:
            try:
                region = CaptureRegion.from_dict(json.loads(raw_region))
                region = refresh_region(region)
                self._set_monitor_region(region)
                self._set_monitor_status("保存的监控区域已恢复", "info")
            except (ValueError, TypeError, json.JSONDecodeError, CaptureError):
                self._monitor_region = None
        self._update_monitor_outputs()
        self._sync_monitor_ui()

    # ------------------------------------------------------------------ result HUD
    def _ensure_floating(self) -> FloatingTranslationWindow:
        if self._floating is None:
            self._floating = FloatingTranslationWindow()
            self._floating.closed.connect(self._on_floating_closed)
            self._floating.click_through_changed.connect(
                self._on_click_through_changed
            )
            self._floating.geometry_changed.connect(
                self._on_floating_geometry_changed
            )
        return self._floating

    def _show_floating_result(
        self,
        image: QtGui.QImage,
        region: CaptureRegion | None,
        source: str,
        *,
        activate: bool = False,
        fade: bool = False,
    ) -> None:
        floating = self._ensure_floating()
        is_monitor = source == "monitor"
        if (
            fade
            and floating.isVisible()
            and floating._region == region
            and floating._source == source
        ):
            floating.update_image(image, fade=True)
            self._update_monitor_exclusion()
            self._sync_tray_actions()
            return
        floating.show_result(
            image,
            region,
            source=source,
            # Both manual captures and region monitoring snap exactly over the
            # source pixels. The monitor HUD is click-through, and its rectangle
            # is excluded from frame diffing, so covering the region is safe and
            # actually keeps the translation aligned with the game/chat UI.
            align=True,
            avoid_region=False,
            activate=activate,
            # Monitoring and whole-screen overlays pass the pointer through by
            # default; a manual area capture is a finished overlay and stays
            # interactive unless the tray toggle changes it.
            click_through=is_monitor or source == "fullscreen",
            fade=fade,
        )
        self._update_monitor_exclusion()
        self._sync_tray_actions()

    def _update_monitor_exclusion(self) -> None:
        if not self._monitor_running:
            self.monitor.set_exclusions([])
            return
        rects: list[QtCore.QRect] = []
        if self._floating is not None and self._floating.isVisible():
            rects.append(self._floating.frameGeometry())
        if self.isVisible():
            rects.append(self.frameGeometry())
        self.monitor.set_exclusions(rects)

    def _on_floating_geometry_changed(self, rect: QtCore.QRect) -> None:
        self._update_monitor_exclusion()

    def _suspend_monitor(self) -> None:
        if self._monitor_running and not self._monitor_paused:
            self.monitor.suspend_capture(True, "翻译任务处理中，自动监控暂缓")

    def _resume_monitor(self) -> None:
        if self._monitor_running and not self._monitor_paused:
            self.monitor.suspend_capture(False)

    def show_overlay(self) -> None:
        if self._last_result_image is None:
            return
        self._show_floating_result(
            self._last_result_image,
            self._last_result_region,
            self._last_result_source,
            activate=True,
        )

    def _on_floating_closed(self) -> None:
        self._update_monitor_exclusion()
        self._sync_tray_actions()

    def _on_click_through_changed(self, enabled: bool) -> None:
        self._sync_tray_actions()
        if enabled and self.tray is not None:
            self.tray.showMessage(
                APP_NAME,
                "译文浮窗已开启鼠标穿透：鼠标可直接操作下方内容。"
                "需要拖动或关闭浮窗时，请在托盘菜单关闭「浮窗鼠标穿透」。",
                QtWidgets.QSystemTrayIcon.MessageIcon.Information,
                2200,
            )

    def _on_click_through_action(self, enabled: bool) -> None:
        if self._floating is None or not self._floating.isVisible():
            self._sync_tray_actions()
            return
        self._floating.set_click_through(enabled)
        if enabled and self.tray is not None:
            self.tray.showMessage(
                APP_NAME,
                "译文浮窗已恢复鼠标穿透：鼠标可直接操作下方内容。"
                "需要拖动或关闭浮窗时，请在托盘菜单关闭「浮窗鼠标穿透」。",
                QtWidgets.QSystemTrayIcon.MessageIcon.Information,
                2200,
            )

    def save_result(self) -> None:
        if self.result_image is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "导出翻译图片",
            str(Path.home() / f"o-translator-{int(time.time())}.png"),
            "PNG 图片 (*.png)",
        )
        if path and not self.result_image.save(path, "PNG"):
            QMessageBoxCompat.warning(self, APP_NAME, "保存失败。")

    # ------------------------------------------------------------------ text
    def _render_translation(self) -> None:
        result = self.current_result or {}
        regions = result.get("regions") or []
        blocks = result.get("blocks") or []
        block_map = {block.get("id"): block for block in blocks}
        if not regions:
            self.translation_view.setHtml(
                "<div style='color:#8b8e98;font-size:11px;padding:8px'>"
                "未检测到文字。可以更换截图或调整源语言后重试。</div>"
            )
            return
        rows: list[str] = []
        for index, region in enumerate(regions):
            source = html.escape(str(region.get("text") or "—"))
            translated = html.escape(
                str(region.get("translation") or region.get("text") or "—")
            )
            block = block_map.get(region.get("id"), {})
            color = block.get("text_color") or [255, 255, 255]
            font_size = block.get("font_size", "")
            overflow = (
                "<span style='color:#f4b95f;font-size:8px'> · OVERFLOW</span>"
                if block.get("overflow")
                else ""
            )
            rows.append(
                "<tr>"
                f"<td style='color:#555967;font-size:8px;padding:7px 6px 7px 0;"
                f"vertical-align:top'>{index + 1:02d}</td>"
                "<td style='padding:7px 8px 7px 0;vertical-align:top;"
                "border-bottom:1px solid rgba(255,255,255,.055)'>"
                f"<div style='color:#8b8e98;font-size:11px;line-height:1.55'>{source}</div>"
                "<div style='color:#f4f1e9;font-size:11.5px;line-height:1.55;"
                f"margin-top:3px'>{translated}</div>"
                "<div style='color:#555967;font-size:8px;margin-top:3px'>"
                f"<span style='color:rgb({color[0]},{color[1]},{color[2]})'>■</span> "
                f"{font_size}px{overflow}</div></td></tr>"
            )
        self.translation_view.setHtml(
            "<table width='100%' cellspacing='0' cellpadding='0' "
            "style='border-collapse:collapse'>" + "".join(rows) + "</table>"
        )

    # ------------------------------------------------------------------ helpers
    def show_status(self, message: str, *, warning: bool = False) -> None:
        self.status_label.setText(message)
        self.status_label.setProperty("warning", warning)
        self._repolish(self.status_label)

    @staticmethod
    def _repolish(widget: QtWidgets.QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _combo_value(self, combo: QtWidgets.QComboBox, default: str) -> str:
        values = combo.property("values") or []
        index = combo.currentIndex()
        if 0 <= index < len(values):
            return values[index]
        return default

    def _set_combo_value(self, combo: QtWidgets.QComboBox, value: str) -> None:
        values = combo.property("values") or []
        if value in values:
            combo.setCurrentIndex(values.index(value))

    def _load_preferences(self) -> None:
        self._set_combo_value(
            self.source_combo, str(self.settings.value("source_lang", "auto"))
        )
        self._set_combo_value(
            self.target_combo, str(self.settings.value("target_lang", "zh"))
        )
        self._set_combo_value(
            self.mode_combo, str(self.settings.value("mode", "auto"))
        )
        self._set_combo_value(
            self.order_combo, str(self.settings.value("order", "auto"))
        )
        self.dilate_spin.setValue(int(self.settings.value("dilate", 7)))
        self.font_scale_spin.setValue(float(self.settings.value("font_scale", 1.0)))
        self.translate_check.setChecked(
            str(self.settings.value("translate", "true")).lower() != "false"
        )
        self.hotkey_edit.setKeySequence(
            QtGui.QKeySequence(str(self.settings.value("hotkey", self.hotkey.sequence)))
        )

    def _save_preferences(self) -> None:
        self.settings.setValue("source_lang", self._combo_value(self.source_combo, "auto"))
        self.settings.setValue("target_lang", self._combo_value(self.target_combo, "zh"))
        self.settings.setValue("mode", self._combo_value(self.mode_combo, "auto"))
        self.settings.setValue("order", self._combo_value(self.order_combo, "auto"))
        self.settings.setValue("dilate", self.dilate_spin.value())
        self.settings.setValue("font_scale", self.font_scale_spin.value())
        self.settings.setValue("translate", self.translate_check.isChecked())
        self.settings.setValue("hotkey", self.hotkey.sequence)
        self.settings.setValue("hotkey_version", HOTKEY_SETTINGS_VERSION)
        self.settings.setValue("app_mode", self._mode)
        self.settings.setValue(
            "monitor_settings", json.dumps(self._monitor_settings().to_dict())
        )
        self.settings.setValue(
            "monitor_auto_overlay",
            self.monitor_auto_overlay.isChecked(),
        )
        self.settings.setValue(
            "monitor_minimize",
            self.monitor_minimize.isChecked(),
        )
        if self._monitor_region is not None:
            self.settings.setValue(
                "monitor_region", json.dumps(self._monitor_region.to_dict())
            )

    def _restore_window(self) -> None:
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()

    def _on_second_launch(self) -> None:
        """Handle a second double-click of the same executable."""
        if self._selector is not None and self._selector.is_open:
            # Never cover an active selection overlay with the workbench.
            return
        self._restore_window()
        if self.tray is not None:
            self.tray.showMessage(
                APP_NAME,
                "程序已在运行，已为你打开主窗口。",
                QtWidgets.QSystemTrayIcon.MessageIcon.Information,
                1800,
            )

    def _quit(self) -> None:
        self._force_quit = True
        self.close()
        QtWidgets.QApplication.instance().quit()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.tray is not None and not self._force_quit:
            event.ignore()
            self._save_preferences()
            self.hide()
            self.tray.showMessage(
                APP_NAME,
                f"已最小化到托盘。{self.hotkey.display} 可随时框选翻译。",
                QtWidgets.QSystemTrayIcon.MessageIcon.Information,
                2200,
            )
            return
        self._save_preferences()
        super().closeEvent(event)
        if self.tray is None:
            QtWidgets.QApplication.instance().quit()

    def _shutdown(self) -> None:
        self._selection_pending = False
        self.poll_timer.stop()
        if self._selector is not None:
            with suppress(Exception):
                self._selector.close()
            self._selector = None
        self.monitor.stop_monitoring()
        self.hotkey.unregister()
        self.backend.close()
        if self._floating is not None:
            self._floating.close()
        if self.tray is not None:
            self.tray.hide()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls() and any(
            is_image_path(url.toLocalFile()) for url in event.mimeData().urls()
        ):
            event.acceptProposedAction()

    def moveEvent(self, event: QtGui.QMoveEvent) -> None:
        super().moveEvent(event)
        if getattr(self, "_monitor_running", False):
            self._update_monitor_exclusion()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if getattr(self, "_monitor_running", False):
            self._update_monitor_exclusion()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if getattr(self, "_monitor_running", False):
            QTimer.singleShot(0, self._update_monitor_exclusion)

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        if getattr(self, "_monitor_running", False):
            QTimer.singleShot(0, self._update_monitor_exclusion)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        if self._busy:
            self.show_status("当前任务仍在处理中，请稍候。", warning=True)
            return
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if is_image_path(path):
                self.load_file(path, submit=True)
                event.acceptProposedAction()
                return


def _present_initial_window(window: MainWindow, *, show_window: bool) -> None:
    """Keep normal launches in the tray unless the caller asks for the UI."""
    if show_window:
        window.show()
        return
    # Normal launches stay in the tray. F2 captures and translation results
    # use the floating HUD, so the workbench must not cover the desktop on
    # startup. The tray balloon is the only launch feedback.
    window.notify_started()


def main(*, show_window: bool = False) -> int:
    if sys.platform == "win32":
        with suppress(OSError):
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "OTranslator.Desktop.0.2"
            )
    QtWidgets.QApplication.setApplicationName(APP_NAME)
    QtWidgets.QApplication.setOrganizationName("O-Translator")
    QtGui.QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 9))

    guard = SingleInstanceGuard()
    if not guard.acquire():
        # A second click must only surface the existing window.  Do not create
        # a MainWindow, tray icon, hotkey registration, or backend process.
        guard.notify_existing()
        return 0
    app.aboutToQuit.connect(guard.release)

    window = MainWindow()
    guard.activation_requested.connect(window._on_second_launch)
    _present_initial_window(window, show_window=show_window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())