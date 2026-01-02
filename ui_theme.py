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
        QLabel, QCheckBox {{
            color: {theme.text};
            font-family: 'Segoe UI', Arial, sans-serif;
        }}
        QLabel#status {{
            color: {theme.text_muted};
        }}
        QPushButton, QComboBox, QSpinBox, QDoubleSpinBox {{
            font-family: 'Segoe UI', Arial, sans-serif;
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
        QCheckBox::indicator {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 1px solid {theme.border};
            background: {theme.surface};
        }}
        QCheckBox::indicator:checked {{
            background: {theme.accent};
            border: 1px solid {theme.accent};
        }}
    """
