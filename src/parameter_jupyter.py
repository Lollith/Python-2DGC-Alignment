
import os
import ipywidgets as widgets
from IPython.display import display
from ipyfilechooser import FileChooser
from pathlib import Path
import traceback

class ParameterWidget:
    """Helper class for creating parameter input widgets with validation."""
    
    def __init__(self, name, default_value, description="", validator=None):
        self.name = name
        self.default_value = str(default_value)
        self.description = description
        self.validator = validator or (lambda x: True)
        
        self.widget = widgets.Text(value=self.default_value)
        self.help_text = self._create_help_text()
        
    def _create_help_text(self):
        """Create help text widget."""
        return widgets.HTML(value=f"""
            <div style="margin-left: 20px; font-style: italic; color: #666; font-size: 0.9em;">
                <p>{self.description}</p>
            </div>
        """)
    
    def get_display_widget(self):
        """Get the complete display widget."""
        bold_label = widgets.HTML(value=f'<b>{self.name}:</b>')
        return widgets.HBox([bold_label, self.widget])
    
    def get_value(self):
        """Get the current value."""
        return self.widget.value
    
    def validate(self):
        """Validate the current value."""
        try:
            value = float(self.widget.value)
            return self.validator(value)
        except ValueError:
            return False, f"{self.name} must be a valid number"


def create_method_widgets():
    """Create method selection widgets."""
    label_method = widgets.HTML(value="<b>Peak Detection Method</b>")
    method_radio = widgets.RadioButtons(
        options=['persistent_homology', 'peak_local_max', 'LoG', 'DoG', 'DoH'],
        value='persistent_homology',
        description='',
        disabled=False
    )
    method_widget = widgets.VBox([label_method, method_radio])
    
    label_mode = widgets.HTML(value="<b>Analysis Mode</b>")
    mode_radio = widgets.RadioButtons(
        options=['tic', 'mass_per_mass', '3D'],
        value='tic',
        description='',
        disabled=False
    )
    mode_widget = widgets.VBox([label_mode, mode_radio])
    
    return method_widget, mode_widget, method_radio, mode_radio


def create_action_widgets():
    """Create action buttons and output area."""
    run_button = widgets.Button(
        description="Run Analysis", 
        button_style='primary',
        icon='play'
    )
    clear_button = widgets.Button(
        description="Clear Results", 
        button_style='info',
        icon='eraser'
    )
    output = widgets.Output()
    
    clear_button.on_click(lambda b: output.clear_output())
    
    return run_button, clear_button, output