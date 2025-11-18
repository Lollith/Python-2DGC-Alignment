import os
import ipywidgets as widgets
# from IPython.display import display
# from identification import sample_identification
# import netCDF4 as nc
# import h5py
from ipyfilechooser import FileChooser
# import traceback
from pathlib import Path
# import subprocess
# from IPython.display import display, HTML, clear_output
# import asyncio
# import time

class Interface:
    """
     GC×GC-MS Analysis UI with improved error handling and flexible file/folder selection.
    Provides a widget-based interface for configuring and running GCGCMS analysis.
    Users can select individual files, folders, or subfolders - all compatible files will be processed.
    """

    def __init__(self, supported_extensions):
        """Initialize the base UI with supported file extensions."""
        self.supported_extensions = supported_extensions
        self._setup_style()
        self._initialize_widgets()

    def _setup_environment(self):
        """Set up environment variables and paths."""
        self.docker_volume_path = os.getenv('DOCKER_VOLUME_PATH')
        self.host_volume_path = os.getenv('HOST_VOLUME_PATH')

    def _setup_style(self):
        """Set up widget styling."""
        self.style = {'description_width': 'initial'}

    def _initialize_widgets(self):
        """Initialize widget containers."""
        self._choosers = [] # input
        # self._choosers_output = []
        self._vbox = widgets.VBox(layout=widgets.Layout(border='2px solid green'))
        self._vbox2 = widgets.VBox(layout=widgets.Layout(border='2px solid green'))
    
    def _create_base_widgets(self):
        """Create base widgets for file selection."""
        self._create_path_widgets()
        self._create_output_widgets()

    def _create_path_widgets(self):
        """Create path selection widgets."""
        self.path_label = widgets.HTML(value=f'''
            <b>Input files</b><br>
            <i>Select files ({", ".join(self.supported_extensions)}) or folders</i><br>
        ''')

        self.add_path_button = widgets.Button(
            description="Add Path",
            button_style='success',
            icon='plus'
        )
        self.remove_button = widgets.Button(
            description="Remove last Path", 
            button_style='warning',
            icon='trash'
        )
        self.add_path_button.on_click(self.add_path_chooser)
        self.remove_button.on_click(self.remove_last_chooser)
        self.button_box = widgets.HBox([
            self.add_path_button,
            self.remove_button
        ])
        self._vbox.children = (self.path_label, self.button_box)

    def _create_output_widgets(self):
        """Create output path widgets."""
        self.output_label = widgets.HTML(value=f'''
                <b>Output Directory</b><br>
                <i>💡 By default, results will be saved in an 
                <b>'output/'</b> folder created at the same location 
                as your input data..</i><br>''')
        self.add_output_button = widgets.Button(
            description="Add Output Path", 
            button_style='success',
            icon='folder-open'
        )
        self.remove_output_button = widgets.Button(
            description="Remove Output Path", 
            button_style='warning',
            icon='trash'
        )
        self.create_folder_button = widgets.Button(
            description="Create New Folder",
            button_style='info',
            icon='folder-plus'
        )   
        
        self.add_output_button.on_click(self.add_output_chooser)
        self.remove_output_button.on_click(self.remove_output_chooser)
        self.create_folder_button.on_click(self.create_output_folder)

        self.button_box_output = widgets.HBox([
            self.add_output_button, 
            self.remove_output_button,
            self.create_folder_button
        ])
        self._vbox2.children = (self.output_label, self.button_box_output)
    
    def add_path_chooser(self, b):
        """Add a new path chooser (file or folder) to the interface."""
        fc = FileChooser(
            path=self.docker_volume_path,
            select_dirs=False,
            show_only_dirs=False,
            sandbox_path=self.docker_volume_path,
            title=f"Select a file ({', '.join(self.supported_extensions)}) or a folder"
        )
        self._choosers.append(fc)
        self._update_chooser_display()

    def add_output_chooser(self, b):
        """Add a new output path chooser."""
        # print("Add Output clicked")
        self.output_chooser = FileChooser(
            path=self.docker_volume_path,
            select_dirs=True,
            show_only_dirs=True,
            sandbox_path=self.docker_volume_path,
            title="Select output folder"
        )
        self._vbox2.children = [
            self.output_label,
            self.button_box_output,
            self.output_chooser
        ]
        # print("Output chooser added successfully")
        
    def remove_output_chooser(self, b):
        """Remove the output path chooser."""
        if hasattr(self, 'output_chooser'):
            delattr(self, 'output_chooser')
            # Remettre l'affichage sans le chooser
            self._vbox2.children = (
                self.output_label,
                self.button_box_output,
            )
            print("Output chooser removed")
        else:
            print("No output chooser to remove")

    def remove_last_chooser(self, b):
        """Remove the last added path chooser."""
        if self._choosers:
            self._choosers.pop()
            self._update_chooser_display()

    def _update_chooser_display(self):
        """Update the display of path choosers."""
        chooser_widgets = []
        for i, fc in enumerate(self._choosers):
            # Add a separator and index for each chooser
            separator = widgets.HTML(f'<hr><b>Path {i+1}:</b>')
            # Add selection info
            selection_info = widgets.HTML(
                value=f'<small style="color: #666;">Cliquez sur un fichier ou double-cliquez sur un dossier pour le sélectionner</small>'
            )
            chooser_widgets.extend([separator, selection_info, fc])
        self._vbox.children = (self.path_label, self.button_box, *chooser_widgets)

    def create_output_folder(self, b):
        """Create a new output folder."""
        # Créer un FileChooser pour sélectionner l'emplacement
        location_chooser = FileChooser(
            path=self.docker_volume_path,
            select_dirs=True,
            show_only_dirs=True,
            title="Choisir l'emplacement pour le nouveau dossier"
        )
    # Si un output_chooser existe déjà, utiliser son chemin comme point de départ
        if hasattr(self, 'output_chooser') and self.output_chooser.selected_path:
            location_chooser.reset(path=self.output_chooser.selected_path)
            # Créer un widget pour saisir le nom du dossier
        folder_name_widget = widgets.Text(
                placeholder="Nom du nouveau dossier",
                description="Nom:",
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='300px')
        )
        create_button = widgets.Button(
            description="Créer",
            button_style='success',
            icon='folder'
        )
        cancel_button = widgets.Button(
            description="Annuler",
            button_style='info',
            icon='times'
        )
        status_label = widgets.HTML(value="")
        # Label d'information sur l'emplacement actuel
        location_info = widgets.HTML(value="")

        def update_location_info():
            """Met à jour l'affichage de l'emplacement sélectionné."""
            if location_chooser.selected_path:
                user_path = location_chooser.selected_path.replace(
                    self.docker_volume_path, self.host_volume_path, 1
                )
                location_info.value = f'<i>📁 Emplacement: {user_path}</i>'
            else:
                location_info.value = '<i>❓ Veuillez sélectionner un emplacement</i>'

        def on_location_change(change):
            """Callback quand l'emplacement change."""
            update_location_info()
        
        # Observer les changements de sélection
        location_chooser.observe(on_location_change, names='selected_path')

        def on_create_folder(b):
            folder_name = folder_name_widget.value.strip()
            selected_location = location_chooser.selected_path
            if not folder_name:
                status_label.value = '<span style="color: red;">⚠️ Veuillez entrer un nom de dossier</span>'
                return
            if not selected_location:
                status_label.value = '<span style="color: red;">⚠️ Veuillez sélectionner un emplacement</span>'
                return
            invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            if any(char in folder_name for char in invalid_chars):
                status_label.value = '<span style="color: red;">⚠️ Le nom contient des caractères interdits</span>'
                return
        
            try:  
                new_folder_path = os.path.join(selected_location, folder_name)
                
                # Vérifier si le dossier existe déjà
                if os.path.exists(new_folder_path):
                    status_label.value = f'<span style="color: orange;">⚠️ Le dossier "{folder_name}" existe déjà</span>'
                    return
                
                # Créer le dossier
                os.makedirs(new_folder_path, exist_ok=True)
                
                # Convertir le chemin pour l'affichage utilisateur
                user_path = new_folder_path.replace(self.docker_volume_path, self.host_volume_path, 1)
                
                status_label.value = f'<span style="color: green;">✅ Dossier créé: {user_path}</span>'
                
                # Rafraîchir le FileChooser s'il existe
                if hasattr(self, 'output_chooser'):
                    self.output_chooser.reset(path=new_folder_path)
                else:
                    # Créer automatiquement l'output_chooser avec le nouveau dossier
                    self.output_chooser = FileChooser(
                        path=new_folder_path,
                        select_dirs=True,
                        show_only_dirs=True,
                        title="Select output folder"
                    )
                    # Mettre à jour l'affichage
                    current_children = list(self._vbox2.children)
                    if self.output_chooser not in current_children:
                        current_children.insert(-1, self.output_chooser)
                        self._vbox2.children = tuple(current_children)
                
                print(f"📁 Nouveau dossier créé: {user_path}")
                print(f"📁 Dossier automatiquement sélectionné comme sortie")
                
                # Fermer le widget de création après succès
                folder_creation_widget.close()
                
            except Exception as e:
                status_label.value = f'<span style="color: red;">❌ Erreur: {str(e)}</span>'
                print(f"❌ Erreur lors de la création du dossier: {e}")

        def on_cancel(b):
            # Fermer le widget de création
            folder_creation_widget.close()
        
        create_button.on_click(on_create_folder)
        cancel_button.on_click(on_cancel)

        update_location_info()
        
       # Créer l'interface de création de dossier
        folder_creation_widget = widgets.VBox([
            widgets.HTML('<b>📁 Créer un nouveau dossier de sortie</b>'),
            widgets.HTML('<small>1. Sélectionnez l\'emplacement où créer le dossier</small>'),
            location_chooser,
            location_info,
            widgets.HTML('<small>2. Donnez un nom au nouveau dossier</small>'),
            folder_name_widget,
            widgets.HBox([create_button, cancel_button]),
            status_label
        ], layout=widgets.Layout(
            border='2px solid #4CAF50',
            padding='15px',
            margin='10px 0',
            background_color='#f9f9f9'
        ))
        
        # Ajouter le widget à l'interface
        current_children = list(self._vbox2.children)
        current_children.append(folder_creation_widget)
        self._vbox2.children = tuple(current_children)

    def _setup_callbacks(self):
        """Set up callbacks for interactive widgets."""
        self.run_button.on_click(self._on_button_click)
        # self.stop_button.on_click(self._on_stop_click)
        

    def _bold_widget(self, label, widget):
        """Create a widget with bold label."""
        bold_label = widgets.HTML(value=f'<b>{label}:</b>')
        return widgets.HBox([bold_label, widget])

    def get_output_path(self):
        """Get the output path from the output chooser, or auto-generate from first input."""
        # ✅ Si l'output_chooser existe et a une sélection, l'utilise
        if hasattr(self, 'output_chooser'):
            path = self.output_chooser.selected_path or self.output_chooser.default_path
            if path:
                return path
        
        # ✅ SINON, génère automatiquement depuis le premier input
        if len(self._choosers) > 0 and self._choosers[0].selected_path:
            input_path = str(self._choosers[0].selected_path)
            normalized_path = input_path.replace('\\', '/')
            if normalized_path.endswith('/'):
                normalized_path = normalized_path[:-1]
            auto_output = f"{normalized_path}/output"
            # print(f"📁 Auto-generated output path: {auto_output}")
            return auto_output
        
        return None

    def get_all_files_from_selections(self):
        """
        Retrieves all supported files from all selections.
        Automatically determines whether it's a file or a folder.
        """
        all_files = []
        processed_files = set()
        already_seen_files = set()

        for i, fc in enumerate(self._choosers):
            selected = fc.selected_path
            if not selected:
                continue

            try:
                selected_path = Path(selected)

                if fc.selected_filename:
                    full_path = selected_path / fc.selected_filename
                    full_path_str = str(full_path)

                    name_without_ext = fc.selected_filename.rsplit('.', 1)[0]

                    if fc.selected_filename.endswith(self.supported_extensions):
                        all_files.append(full_path_str)
                        processed_files.add(full_path_str)
                    else:
                        print(f"⚠️  Unsupported file ignored: {fc.selected_filename}")
                        print(f"   Supported extensions: {', '.join(self.supported_extensions)}")


                else: # ce n 'est pas un fichier
                    # print(f"📁 Processing folder: {selected_path}")
                    selected_path_str = str(selected_path)
                    if selected_path_str in processed_files:
                        print(f"⚠️  Folder already processed: {selected_path_str}")
                        continue

                    processed_files.add(selected_path_str)
                    dir_files = self._get_files_from_directory(selected_path)

                    for f in dir_files:
                        path = str(Path(f))
                        if path not in processed_files:
                            all_files.append(path)
                            processed_files.add(path)
                    print(f"   Found {len(dir_files)} compatible files")

            except Exception as e:
                print(f"❌ Error while processing selection '{selected}': {e}")

        return all_files


    def _get_files_from_directory(self, directory_path):
        """Recursively retrieves all supported files from a folder, prioritizing .h5 over .cdf files."""
        files = []

        try:
            file_map = {}
            for root, _, filenames in os.walk(directory_path):
                for filename in filenames:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in self.supported_extensions:
                        continue

                    full_path = os.path.join(root, filename)
                    name_without_ext = os.path.splitext(os.path.basename(full_path))[0]

                    if name_without_ext in file_map:
                        existing_ext = os.path.splitext(file_map[name_without_ext])[1].lower()
                        # Priorité à l’ordre donné dans self.supported_extensions
                        if self.supported_extensions.index(ext) < self.supported_extensions.index(existing_ext):
                            file_map[name_without_ext] = full_path
                    else:
                        file_map[name_without_ext] = full_path

            files = list(file_map.values())

        except Exception as e:
            print(f"❌ Error while scanning directory  {directory_path}: {e}")

        return files
    
    def _create_action_widgets(self):
        """Create action buttons and output area."""
        run_button = widgets.Button(
            description="Run", 
            button_style='primary',
            icon='play'
        )
        stop_button = widgets.Button(
            description="Stop Analysis",
            #button_style='danger',
            button_style='', #gris au lieu de rouge car inactif
            icon='stop',
            disabled=True,  # désactivé
            tooltip="Feature temporarily disabled"
        )

        clear_button = widgets.Button(
            description="Clear Results", 
            button_style='info',
            icon='eraser'
        )
        output = widgets.Output()
        
        clear_button.on_click(lambda b: output.clear_output())
        
        return run_button, stop_button, clear_button, output


    def validate_selections(self):
        """Validate that files and output are properly selected."""
        errors = []
        
        files = self.get_all_files_from_selections()
        if not files:
            errors.append("No compatible files selected")
            
        if not hasattr(self, 'output_chooser') or not self.output_chooser.selected_path:
            errors.append("Output directory not selected")
            
        return errors, files

    def create_help_text(self, text):
        """Create formatted help text."""
        return widgets.HTML(value=f"""
            <div style="margin-left: 20px; font-style: italic; color: #666; font-size: 0.9em;">
                <p>{text}</p>
            </div>
        """)

    def get_file_selection_widgets(self):
        """Return the file selection widgets for display."""
        return widgets.VBox([self._vbox, self._vbox2])
    

    

    # def _on_stop_click(self, b):
    #     """Stop the running analysis process."""
        # with self.output:
    #         if hasattr(self, "current_process") and self.current_process:
    #             print(f"\n{'='*60}")
    #             print("⛔ Stopping analysis...")
    #             try:
    #                 self.current_process.terminate()  # envoie SIGTERM
    #                 retcode = self.current_process.wait()  # attends la fin
    #                 print(f"Process stopped, return code: {retcode}")
    #                 print(f"\n{'='*60}")
    #                 self.current_process = None
    #             except Exception as e:
    #                 print(f"⚠️ Impossible de terminer le process : {e}")
    

    # 