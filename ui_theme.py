from dataclasses import dataclass


@dataclass(frozen=True)
class ThemeTokens:
    background: str
    surface: str
    text: str
    text_muted: str
    border: str
    primary: str
    primary_text: str
    primary_hover: str
    primary_pressed: str
    secondary: str
    secondary_text: str
    secondary_hover: str
    secondary_pressed: str
    accent: str
    success: str
    warning: str
    error: str
    focus_ring: str
    input_background: str
    input_text: str


LIGHT_THEME = ThemeTokens(
    background="#f5f7fa",
    surface="#ffffff",
    text="#1f2933",
    text_muted="#4b5563",
    border="#d0d7de",
    primary="#2563eb",
    primary_text="#ffffff",
    primary_hover="#1d4ed8",
    primary_pressed="#1e40af",
    secondary="#e5e7eb",
    secondary_text="#1f2933",
    secondary_hover="#d1d5db",
    secondary_pressed="#cbd5e1",
    accent="#7c3aed",
    success="#16a34a",
    warning="#d97706",
    error="#dc2626",
    focus_ring="#93c5fd",
    input_background="#ffffff",
    input_text="#111827",
)


def build_stylesheet(theme: ThemeTokens) -> str:
    return f"""
        QMainWindow {{
            background: {theme.background};
        }}

        QWidget {{
            font-family: 'Segoe UI', Arial, sans-serif;
        }}

        QLabel, QCheckBox {{
            color: {theme.text};
        }}

        QLabel#panelTitle {{
            color: {theme.text};
            font-size: 16px;
            font-weight: 600;
        }}

        QLabel#topBarTitle {{
            color: {theme.text};
            font-size: 18px;
            font-weight: 600;
        }}

        QLabel#sectionTitle {{
            color: {theme.text_muted};
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
        }}

        QLabel#status {{
            color: {theme.text_muted};
            font-size: 12px;
        }}

        /* Sidebar section titles (Workflow / Advanced) */
        QLabel#sidebarSectionTitle {{
            color: {theme.text_muted};
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 4px;
        }}

        QPushButton, QComboBox, QSpinBox, QDoubleSpinBox {{
            font-size: 14px;
        }}

        QComboBox, QPushButton, QSpinBox, QDoubleSpinBox {{
            height: 36px;
            border-radius: 8px;
        }}

        QPushButton {{
            background: {theme.secondary};
            color: {theme.secondary_text};
            border: 1px solid {theme.border};
        }}

        QPushButton:hover {{
            background: {theme.secondary_hover};
        }}

        QPushButton:focus {{
            border: 2px solid {theme.focus_ring};
        }}

        QPushButton:pressed {{
            background: {theme.secondary_pressed};
        }}

        QPushButton[variant="primary"] {{
            background: {theme.primary};
            color: {theme.primary_text};
            border: 1px solid {theme.primary};
        }}

        QPushButton[variant="primary"]:hover {{
            background: {theme.primary_hover};
        }}

        QPushButton[variant="primary"]:focus {{
            border: 2px solid {theme.focus_ring};
        }}

        QPushButton[variant="primary"]:pressed {{
            background: {theme.primary_pressed};
        }}

        QPushButton:disabled {{
            background: {theme.surface};
            color: {theme.text_muted};
            border: 1px solid {theme.border};
        }}

        QComboBox, QSpinBox, QDoubleSpinBox {{
            background: {theme.input_background};
            color: {theme.input_text};
            border: 1px solid {theme.border};
            padding: 4px 8px;
        }}

        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
            border: 1px solid {theme.secondary_hover};
        }}

        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 2px solid {theme.focus_ring};
        }}

        QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            background: {theme.surface};
            color: {theme.text_muted};
            border: 1px solid {theme.border};
        }}

        QTextEdit {{
            background: {theme.surface};
            color: {theme.input_text};
            font-family: Consolas, monospace;
            font-size: 13px;
            border-radius: 8px;
            border: 1px solid {theme.border};
            padding: 6px;
        }}

        QTextEdit:hover {{
            border: 1px solid {theme.secondary_hover};
        }}

        QTextEdit:focus {{
            border: 2px solid {theme.focus_ring};
        }}

        QTextEdit:disabled {{
            background: {theme.surface};
            color: {theme.text_muted};
            border: 1px solid {theme.border};
        }}

        QTabWidget::pane {{
            border: 1px solid {theme.border};
            border-radius: 10px;
            padding: 4px;
            background: {theme.surface};
        }}

        QTabBar::tab {{
            background: {theme.secondary};
            color: {theme.text};
            border: 1px solid {theme.border};
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 6px 12px;
            margin-right: 4px;
        }}

        QTabBar::tab:selected {{
            background: {theme.surface};
            color: {theme.text};
        }}

        QSplitter::handle {{
            background: {theme.background};
        }}

        QSplitter::handle:horizontal {{
            width: 8px;
        }}

        /* Default checkbox look (overridden for similarityToggle below) */
        QCheckBox::indicator {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 1px solid {theme.border};
            background: {theme.surface};
        }}

        QCheckBox::indicator:checked {{
            background: {theme.primary};
            border: 1px solid {theme.primary};
        }}

        /* Card containers */
        QFrame#card {{
            background: {theme.surface};
            border: 1px solid {theme.border};
            border-radius: 10px;
            padding: 10px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}

        /* Sidebar container */
        QFrame#sidebar {{
            background: {theme.surface};
            border-right: 1px solid {theme.border};
        }}

        /* Thin divider inside sidebar (between Workflow and Advanced) */
        QFrame#sidebarDivider {{
            margin-top: 12px;
            margin-bottom: 8px;
        }}

        /* Top bar */
        QFrame#topBar {{
            background: {theme.surface};
            border-bottom: 1px solid {theme.border};
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        }}

        /* Workflow list in sidebar */
        QListWidget#workflowList {{
            background: transparent;
            border: none;
            padding: 4px 0;
            outline: 0;
        }}

        QListWidget#workflowList::item {{
            color: {theme.text};
            padding: 8px 10px;
            margin: 2px 0;
            border-radius: 999px;
        }}

        QListWidget#workflowList::item:selected {{
            background: {theme.primary};
            color: {theme.primary_text};
        }}

        QListWidget#workflowList::item:hover:!selected {{
            background: {theme.secondary_hover};
            color: {theme.secondary_text};
        }}

        /* Advanced / similarity toggle checkbox in sidebar */
        QCheckBox#similarityToggle {{
            margin-left: 2px;
        }}

        QCheckBox#similarityToggle::indicator {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
            border: 1px solid {theme.border};
            background: {theme.secondary};
            margin-right: 6px;
        }}

        QCheckBox#similarityToggle::indicator:checked {{
            background: {theme.primary};
            border-color: {theme.primary};
        }}

        /* Chips (e.g., Specs / Risks / Uncertain in results header) */
        QLabel#chip {{
            background: {theme.surface};
            border: 1px solid {theme.border};
            border-radius: 10px;
            padding: 4px 8px;
        }}
    """


def apply_theme(app_or_widget) -> None:
    """Apply the light theme to the provided QApplication or widget."""

    stylesheet = build_stylesheet(LIGHT_THEME)
    try:
        app_or_widget.setStyleSheet(stylesheet)
    except AttributeError:
        pass
