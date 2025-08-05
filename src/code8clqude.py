# import os
# import ipywidgets as widgets
# from IPython.display import display
# from ipyfilechooser import FileChooser
# from pathlib import Path
# import traceback


# class BaseFileUI:
#     """
#     Interface de base réutilisable pour la sélection de fichiers et configuration d'analyses.
#     Fournit les fonctionnalités communes pour la gestion des fichiers et paramètres.
#     """
    
#     def __init__(self, supported_extensions=('.cdf', '.h5')):
#         """Initialize the base UI with supported file extensions."""
#         self.supported_extensions = supported_extensions
#         self._setup_environment()
#         self._setup_style()
#         self._initialize_widgets()
#         self._create_base_widgets()
        
#     def _setup_environment(self):
#         """Set up environment variables and paths."""
#         self.docker_volume_path = os.getenv('DOCKER_VOLUME_PATH')
#         self.host_volume_path = os.getenv('HOST_VOLUME_PATH')
        
#     def _setup_style(self):
#         """Set up widget styling."""
#         self.style = {'description_width': 'initial'}
        
#     def _initialize_widgets(self):
#         """Initialize widget containers."""
#         self._choosers = []  # input
#         self._vbox = widgets.VBox(layout=widgets.Layout(border='2px solid green'))
#         self._vbox2 = widgets.VBox(layout=widgets.Layout(border='2px solid green'))
        
#     def _create_base_widgets(self):
#         """Create base widgets for file selection."""
#         self._create_path_widgets()
#         self._create_output_widgets()
        
#     def _create_path_widgets(self):
#         """Create path selection widgets."""
#         self.path_label = widgets.HTML(value=f'''
#             <b>Input files</b><br>
#             <i>Select files ({", ".join(self.supported_extensions)}) or folders</i><br>
#         ''')
        
#         self.add_path_button = widgets.Button(
#             description="Add Path", 
#             button_style='success',
#             icon='plus'
#         )
        
#         self.remove_button = widgets.Button(
#             description="Remove last Path", 
#             button_style='warning',
#             icon='trash'
#         )
        
#         self.add_path_button.on_click(self.add_path_chooser)
#         self.remove_button.on_click(self.remove_last_chooser)
        
#         self.button_box = widgets.HBox([
#             self.add_path_button, 
#             self.remove_button
#         ])
        
#         self._vbox.children = (self.path_label, self.button_box)
        
#     def _create_output_widgets(self):
#         """Create output path widgets."""
#         self.output_label = widgets.HTML(value='<b>Output Directory</b>')
#         self.add_output_button = widgets.Button(
#             description="Add Output Path", 
#             button_style='success',
#             icon='folder-open'
#         )
#         self.remove_output_button = widgets.Button(
#             description="Remove Output Path", 
#             button_style='warning',
#             icon='trash'
#         )
#         self.create_folder_button = widgets.Button(
#             description="Create New Folder",
#             button_style='info',
#             icon='folder-plus'
#         )   
        
#         self.add_output_button.on_click(self.add_output_chooser)
#         self.remove_output_button.on_click(self.remove_output_chooser)
#         self.create_folder_button.on_click(self.create_output_folder)

#         self.button_box_output = widgets.HBox([
#             self.add_output_button, 
#             self.remove_output_button,
#             self.create_folder_button
#         ])
#         self._vbox2.children = (self.output_label, self.button_box_output)

#     def add_path_chooser(self, b):
#         """Add a new path chooser (file or folder) to the interface."""
#         fc = FileChooser(
#             path=self.docker_volume_path,
#             select_dirs=False,
#             show_only_dirs=False,
#             sandbox_path=self.docker_volume_path,
#             title=f"Select a file ({', '.join(self.supported_extensions)}) or a folder"
#         )
        
#         self._choosers.append(fc)
#         self._update_chooser_display()

#     def add_output_chooser(self, b):
#         """Add a new output path chooser."""
#         print("Add Output clicked")
#         self.output_chooser = FileChooser(
#             path=self.docker_volume_path,
#             select_dirs=True,
#             show_only_dirs=True,
#             sandbox_path=self.docker_volume_path,
#             title="Select output folder"
#         )
#         self._vbox2.children = [
#             self.output_label,
#             self.button_box_output,
#             self.output_chooser
#         ]
#         print("Output chooser added successfully")
        
#     def remove_output_chooser(self, b):
#         """Remove the output path chooser."""
#         if hasattr(self, 'output_chooser'):
#             delattr(self, 'output_chooser')
#             self._vbox2.children = (
#                 self.output_label,
#                 self.button_box_output,
#             )
#             print("Output chooser removed")
#         else:
#             print("No output chooser to remove")

#     def remove_last_chooser(self, b):
#         """Remove the last added path chooser."""
#         if self._choosers:
#             self._choosers.pop()
#             self._update_chooser_display()
    
#     def _update_chooser_display(self):
#         """Update the display of path choosers."""
#         chooser_widgets = []
        
#         for i, fc in enumerate(self._choosers):
#             separator = widgets.HTML(f'<hr><b>Path {i+1}:</b>')
#             selection_info = widgets.HTML(
#                 value=f'<small style="color: #666;">Click on a file or double-click on a folder to select</small>'
#             )
#             chooser_widgets.extend([separator, selection_info, fc])
        
#         self._vbox.children = (self.path_label, self.button_box, *chooser_widgets)

#     def create_output_folder(self, b):
#         """Create a new output folder."""
#         location_chooser = FileChooser(
#             path=self.docker_volume_path,
#             select_dirs=True,
#             show_only_dirs=True,
#             title="Choose location for new folder"
#         )
    
#         if hasattr(self, 'output_chooser') and self.output_chooser.selected_path:
#             location_chooser.reset(path=self.output_chooser.selected_path)

#         folder_name_widget = widgets.Text(
#             placeholder="New folder name",
#             description="Name:",
#             style={'description_width': 'initial'},
#             layout=widgets.Layout(width='300px')
#         )
        
#         create_button = widgets.Button(
#             description="Create",
#             button_style='success',
#             icon='folder'
#         )
        
#         cancel_button = widgets.Button(
#             description="Cancel",
#             button_style='info',
#             icon='times'
#         )
        
#         status_label = widgets.HTML(value="")
#         location_info = widgets.HTML(value="")

#         def update_location_info():
#             if location_chooser.selected_path:
#                 user_path = location_chooser.selected_path.replace(
#                     self.docker_volume_path, self.host_volume_path, 1
#                 )
#                 location_info.value = f'<i>📁 Location: {user_path}</i>'
#             else:
#                 location_info.value = '<i>❓ Please select a location</i>'

#         def on_location_change(change):
#             update_location_info()
        
#         location_chooser.observe(on_location_change, names='selected_path')

#         def on_create_folder(b):
#             folder_name = folder_name_widget.value.strip()
#             selected_location = location_chooser.selected_path
            
#             if not folder_name:
#                 status_label.value = '<span style="color: red;">⚠️ Please enter a folder name</span>'
#                 return
#             if not selected_location:
#                 status_label.value = '<span style="color: red;">⚠️ Please select a location</span>'
#                 return
                
#             invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
#             if any(char in folder_name for char in invalid_chars):
#                 status_label.value = '<span style="color: red;">⚠️ Name contains invalid characters</span>'
#                 return
        
#             try:
#                 new_folder_path = os.path.join(selected_location, folder_name)
                
#                 if os.path.exists(new_folder_path):
#                     status_label.value = f'<span style="color: orange;">⚠️ Folder "{folder_name}" already exists</span>'
#                     return
                
#                 os.makedirs(new_folder_path, exist_ok=True)
#                 user_path = new_folder_path.replace(self.docker_volume_path, self.host_volume_path, 1)
#                 status_label.value = f'<span style="color: green;">✅ Folder created: {user_path}</span>'
                
#                 if hasattr(self, 'output_chooser'):
#                     self.output_chooser.reset(path=new_folder_path)
#                 else:
#                     self.output_chooser = FileChooser(
#                         path=new_folder_path,
#                         select_dirs=True,
#                         show_only_dirs=True,
#                         title="Select output folder"
#                     )
#                     current_children = list(self._vbox2.children)
#                     if self.output_chooser not in current_children:
#                         current_children.insert(-1, self.output_chooser)
#                         self._vbox2.children = tuple(current_children)
                
#                 print(f"📁 New folder created: {user_path}")
#                 folder_creation_widget.close()
                
#             except Exception as e:
#                 status_label.value = f'<span style="color: red;">❌ Error: {str(e)}</span>'
#                 print(f"❌ Error creating folder: {e}")

#         def on_cancel(b):
#             folder_creation_widget.close()
        
#         create_button.on_click(on_create_folder)
#         cancel_button.on_click(on_cancel)
#         update_location_info()
        
#         folder_creation_widget = widgets.VBox([
#             widgets.HTML('<b>📁 Create new output folder</b>'),
#             widgets.HTML('<small>1. Select location for new folder</small>'),
#             location_chooser,
#             location_info,
#             widgets.HTML('<small>2. Enter folder name</small>'),
#             folder_name_widget,
#             widgets.HBox([create_button, cancel_button]),
#             status_label
#         ], layout=widgets.Layout(
#             border='2px solid #4CAF50',
#             padding='15px',
#             margin='10px 0',
#             background_color='#f9f9f9'
#         ))
        
#         current_children = list(self._vbox2.children)
#         current_children.append(folder_creation_widget)
#         self._vbox2.children = tuple(current_children)

#     def get_output_path(self):
#         """Get the output path from the output chooser."""
#         if hasattr(self, 'output_chooser') and self.output_chooser.selected:
#             selected = self.output_chooser.selected
#             print(f"📁 Output path selected: {selected}")
#             return selected
#         return None

#     def get_all_files_from_selections(self):
#         """
#         Retrieves all supported files from all selections.
#         Automatically determines whether it's a file or a folder.
#         """
#         all_files = []
#         processed_paths = set()
#         already_seen_files = set()

#         for i, fc in enumerate(self._choosers):
#             selected = fc.selected_path
#             if not selected:
#                 continue

#             try:
#                 selected_path = Path(selected)

#                 if str(selected_path) in processed_paths:
#                     continue
#                 processed_paths.add(str(selected_path))

#                 if fc.selected_filename:
#                     name_without_ext = fc.selected_filename.rsplit('.', 1)[0]

#                     if fc.selected_filename.endswith(".cdf") and name_without_ext in already_seen_files:
#                         print(f"⚠️  File already processed: {name_without_ext}.cdf")
#                         continue    

#                     if fc.selected_filename.endswith(".h5"):
#                         already_seen_files.add(name_without_ext)

#                     if fc.selected_filename.endswith(self.supported_extensions):
#                         full_path = selected_path / fc.selected_filename
#                         all_files.append(str(full_path))
#                         print(f"📄 File added: {full_path}")
#                     else:
#                         print(f"⚠️  Unsupported file ignored: {fc.selected_filename}")
#                         print(f"   Supported extensions: {', '.join(self.supported_extensions)}")

#                 else:  # It's a folder
#                     print(f"📁 Processing folder: {selected_path}")
#                     dir_files = self._get_files_from_directory(selected_path)

#                     for f in dir_files:
#                         path = str(Path(f))
#                         if path not in processed_paths:
#                             all_files.append(path)
#                             processed_paths.add(path)
#                     print(f"   Found {len(dir_files)} compatible files")

#             except Exception as e:
#                 print(f"❌ Error while processing selection '{selected}': {e}")

#         return all_files

#     def _get_files_from_directory(self, directory_path):
#         """Recursively retrieves all supported files from a folder, prioritizing .h5 over .cdf files."""
#         files = []

#         try:
#             file_map = {}
#             for root, _, filenames in os.walk(directory_path):
#                 for filename in filenames:
#                     ext = os.path.splitext(filename)[1].lower()
#                     if ext not in ['.cdf', '.h5']:
#                         continue

#                     full_path = os.path.join(root, filename)
#                     name_without_ext = os.path.splitext(os.path.basename(full_path))[0]

#                     if name_without_ext in file_map:
#                         existing_ext = os.path.splitext(file_map[name_without_ext])[1].lower()
#                         if existing_ext == '.cdf' and ext == '.h5':
#                             file_map[name_without_ext] = full_path
#                     else:
#                         file_map[name_without_ext] = full_path

#             files = list(file_map.values())

#         except Exception as e:
#             print(f"❌ Error while scanning directory {directory_path}: {e}")

#         return files

#     def validate_selections(self):
#         """Validate that files and output are properly selected."""
#         errors = []
        
#         files = self.get_all_files_from_selections()
#         if not files:
#             errors.append("No compatible files selected")
            
#         if not hasattr(self, 'output_chooser') or not self.output_chooser.selected_path:
#             errors.append("Output directory not selected")
            
#         return errors, files

#     def create_help_text(self, text):
#         """Create formatted help text."""
#         return widgets.HTML(value=f"""
#             <div style="margin-left: 20px; font-style: italic; color: #666; font-size: 0.9em;">
#                 <p>{text}</p>
#             </div>
#         """)

#     def bold_widget(self, label, widget):
#         """Create a widget with bold label."""
#         bold_label = widgets.HTML(value=f'<b>{label}:</b>')
#         return widgets.HBox([bold_label, widget])

#     def get_file_selection_widgets(self):
#         """Return the file selection widgets for display."""
#         return widgets.VBox([self._vbox, self._vbox2])


# class ParameterWidget:
#     """Helper class for creating parameter input widgets with validation."""
    
#     def __init__(self, name, default_value, description="", validator=None):
#         self.name = name
#         self.default_value = str(default_value)
#         self.description = description
#         self.validator = validator or (lambda x: True)
        
#         self.widget = widgets.Text(value=self.default_value)
#         self.help_text = self._create_help_text()
        
#     def _create_help_text(self):
#         """Create help text widget."""
#         return widgets.HTML(value=f"""
#             <div style="margin-left: 20px; font-style: italic; color: #666; font-size: 0.9em;">
#                 <p>{self.description}</p>
#             </div>
#         """)
    
#     def get_display_widget(self):
#         """Get the complete display widget."""
#         bold_label = widgets.HTML(value=f'<b>{self.name}:</b>')
#         return widgets.HBox([bold_label, self.widget])
    
#     def get_value(self):
#         """Get the current value."""
#         return self.widget.value
    
#     def validate(self):
#         """Validate the current value."""
#         try:
#             value = float(self.widget.value)
#             return self.validator(value)
#         except ValueError:
#             return False, f"{self.name} must be a valid number"


# def create_method_widgets():
#     """Create method selection widgets."""
#     label_method = widgets.HTML(value="<b>Peak Detection Method</b>")
#     method_radio = widgets.RadioButtons(
#         options=['persistent_homology', 'peak_local_max', 'LoG', 'DoG', 'DoH'],
#         value='persistent_homology',
#         description='',
#         disabled=False
#     )
#     method_widget = widgets.VBox([label_method, method_radio])
    
#     label_mode = widgets.HTML(value="<b>Analysis Mode</b>")
#     mode_radio = widgets.RadioButtons(
#         options=['tic', 'mass_per_mass', '3D'],
#         value='tic',
#         description='',
#         disabled=False
#     )
#     mode_widget = widgets.VBox([label_mode, mode_radio])
    
#     return method_widget, mode_widget, method_radio, mode_radio


# def create_action_widgets():
#     """Create action buttons and output area."""
#     run_button = widgets.Button(
#         description="Run Analysis", 
#         button_style='primary',
#         icon='play'
#     )
#     clear_button = widgets.Button(
#         description="Clear Results", 
#         button_style='info',
#         icon='eraser'
#     )
#     output = widgets.Output()
    
#     clear_button.on_click(lambda b: output.clear_output())
    
#     return run_button, clear_button, output