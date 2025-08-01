from math import log
from app.exceptions import DatabaseError
from app.repositories.project_repository import ProjectRepository
from app.repositories.participant_repository import ParticipantRepository
from app.repositories.session_repository import SessionRepository
from app.logging_config import get_logger
import os
import shutil
from datetime import datetime
import random

# Get logger for this module
logger = get_logger(__name__)

class ProjectService:
    def __init__(self, project_repository=None, session_repository=None, participant_repository=None):
        self.project_repo: ProjectRepository = project_repository
        self.participant_repo: ParticipantRepository = participant_repository
        self.session_repo: SessionRepository = session_repository
    
    def list_projects(self):
        """Get all projects"""
        return self.project_repo.get_all()
    
    def get_participant_by_code(self, participant_code):
        """Find participant by code"""
        return self.participant_repo.find_by_code(participant_code)

    def insert_project(self, project_name, participant_id, path):
        """Create a new project"""
        return self.project_repo.create(project_name, participant_id, path)

    def create_participant(self, participant_code):
        """Create a new participant, handling race conditions if it already exists"""
        return self.participant_repo.create(participant_code)

    def create_participant_with_details(self, participant_code, first_name, last_name, email, notes):
        """Create a new participant with detailed information"""
        result = self.participant_repo.create_with_details(participant_code, first_name, last_name, email, notes)
        print(result)
        return {
            'participant_id': result['participant_id'],
            'participant_code': result['participant_code']
        }

    def get_project_with_participant(self, project_id):
        """Get detailed project information including participant data"""
        return self.project_repo.find_with_participant(project_id)

    def cleanup_participant_if_needed(self, participant_id):
        """Check if participant has any remaining projects and delete if none exist"""
        remaining_projects = self.participant_repo.count_projects(participant_id)
        if remaining_projects == 0:
            self.participant_repo.delete(participant_id)
            return True
        return False

    def delete_project(self, project_id):
        """Delete a project"""
        return self.project_repo.delete(project_id)

    def rename_project(self, project_id, new_name):
        """Rename a project"""
        if not new_name or not new_name.strip():
            raise DatabaseError('Project name cannot be empty')
        
        # Trim whitespace from the new name
        new_name = new_name.strip()
        
        # Update the project name in the database
        return self.project_repo.update_name(project_id, new_name)

    def get_all_participants_with_stats(self):
        """Get all participants with their project and session statistics"""
        return self.participant_repo.get_all_with_stats()

    def update_participant(self, participant_id, participant_code, first_name, last_name, email, notes):
        """Update an existing participant's information"""
        result = self.participant_repo.update(participant_id, participant_code, first_name, last_name, email, notes)
        return {
            'participant_id': result['participant_id'],
            'participant_code': result['participant_code']
        }

    def get_participant_info(self, participant_id):
        """Get basic participant information"""
        return self.participant_repo.find_by_id(participant_id)

    def get_participant_projects(self, participant_id):
        """Get all projects for a participant"""
        return self.project_repo.find_by_participant(participant_id)

    def count_participant_sessions(self, participant_id):
        """Count total sessions for a participant across all their projects"""
        return self.participant_repo.count_sessions(participant_id)

    def delete_participant_cascade(self, participant_id):
        """Delete participant and all associated data (projects, sessions, lineage)"""
        return self.participant_repo.delete_cascade(participant_id)

    def create_project_with_files(self, project_name, participant_code, project_path, data_dir):
        """
        Create a new project with uploaded files, handling participant creation and file storage
        
        Args:
            project_name: Name of the project
            participant_code: Code for the participant
            project_path: Path to the project directory
            data_dir: Base directory for storing project data
            
        Returns:
            dict: Project creation result with project_id, participant_id, and project_path
        """
        # Get or create participant
        participant = self.get_participant_by_code(participant_code)
        if participant:
            participant_id = participant['participant_id']
        else:
            created_participant = self.create_participant(participant_code)
            participant_id = created_participant['participant_id']

        # Create project directory
        central_data_dir = os.path.expanduser(data_dir)
        os.makedirs(central_data_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        project_dir_name = f"{project_name}_{participant_code}_{timestamp}"
        new_project_path = os.path.join(central_data_dir, project_dir_name)
        
        try:
            # Create project directory and save files
            os.makedirs(new_project_path, exist_ok=True)
            
            # Copy the provided project path to the new project directory
            if not os.path.exists(project_path):
                raise DatabaseError(f"Provided project path does not exist: {project_path}")
            if not os.path.isdir(project_path):
                raise DatabaseError(f"Provided project path is not a directory: {project_path}")
            # Copy the entire directory structure from the provided path
            shutil.copytree(project_path, new_project_path, dirs_exist_ok=True)
            logger.info(f"Created project directory at {new_project_path}")
            
            # Create project record in database
            created_project = self.insert_project(project_name, participant_id, new_project_path)
            
            return {
                'project_id': created_project['project_id'],
                'participant_id': participant_id,
                'project_path': new_project_path
            }
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(new_project_path):
                shutil.rmtree(new_project_path)
            raise DatabaseError(f'Failed to create project with files: {str(e)}')

    def create_project_with_bulk_files(self, project_name, participant_code, bulkUploadFolderPath, data_dir):
        """
        Create a new project with uploaded files from bulk upload, handling the specific path structure
        
        Args:
            project_name: Name of the project
            participant_code: Code for the participant
            bulkUploadFolderPath: Path to the parent directory containing project folders
            data_dir: Base directory for storing project data
            
        Returns:
            dict: Project creation result with project_id, participant_id, and project_path
        """
        logger.info(f"Creating project '{project_name}' for participant '{participant_code}' with bulk files from {bulkUploadFolderPath}")
        # Get or create participant
        participant = self.get_participant_by_code(participant_code)
        if participant:
            participant_id = participant['participant_id']
        else:
            created_participant = self.create_participant(participant_code)
            participant_id = created_participant['participant_id']

        # Create project directory
        central_data_dir = os.path.expanduser(data_dir)
        os.makedirs(central_data_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        project_dir_name = f"{project_name}_{participant_code}_{timestamp}"
        new_project_path = os.path.join(central_data_dir, project_dir_name)
        
        try:
            # Create project directory and save files
            os.makedirs(new_project_path, exist_ok=True)
            
            # Copy project_name subdirectory from the bulk upload folder
            project_path = os.path.join(bulkUploadFolderPath, project_name)
            logger.info(f"Copying files from {project_path} to {new_project_path}")
            if not os.path.exists(project_path):
                raise DatabaseError(f"Provided project path does not exist: {project_path}")
            if not os.path.isdir(project_path):
                raise DatabaseError(f"Provided project path is not a directory: {project_path}")
            # Copy the entire directory structure from the provided path
            shutil.copytree(project_path, new_project_path, dirs_exist_ok=True)
            logger.info(f"Created project directory at {new_project_path}")
            
            # Create project record in database
            created_project = self.insert_project(project_name, participant_id, new_project_path)
            
            return {
                'project_id': created_project['project_id'],
                'participant_id': participant_id,
                'project_path': new_project_path
            }
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(new_project_path):
                shutil.rmtree(new_project_path)
            raise DatabaseError(f'Failed to create project with bulk files: {str(e)}')

    def get_labelings(self, project_id=None):
        """Get labelings for a project, filtering out deleted ones by default"""
        import json
        
        labelings_data = self.project_repo.get_labelings(project_id)
        
        # If no labelings data, return as is
        if not labelings_data or not labelings_data[0].get('labelings'):
            return labelings_data
            
        # Filter out deleted labelings
        try:
            labelings = json.loads(labelings_data[0]['labelings'])
            if isinstance(labelings, list):
                # Filter out labelings marked as deleted
                active_labelings = []
                for labeling in labelings:
                    if isinstance(labeling, dict):
                        # Only include if not deleted
                        if not labeling.get('is_deleted', False):
                            active_labelings.append(labeling)
                    elif isinstance(labeling, str):
                        # String labelings are considered active (old format)
                        active_labelings.append(labeling)
                
                # Update the labelings data with filtered results
                labelings_data[0]['labelings'] = json.dumps(active_labelings)
                
        except (json.JSONDecodeError, TypeError, KeyError):
            # If there's an error parsing, return original data
            pass
            
        return labelings_data

    def add_list_of_labeling_names_to_project(self, project_id, labels):
        """Add new labels to a project's labelings"""
        logger.info(f'Adding labels to project {project_id}: {labels}')

        # Validate input
        if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
            raise DatabaseError('Labels must be a list of strings')
        pretty_colors = [
            '#FF6B6B',  # Coral Red
            '#4ECDC4',  # Turquoise
            '#45B7D1',  # Sky Blue
            '#96CEB4',  # Mint Green
            '#FFEAA7',  # Warm Yellow
            '#DDA0DD',  # Plum
            '#98D8C8',  # Seafoam
            '#F7DC6F',  # Light Gold
            '#BB8FCE',  # Lavender
            '#85C1E9',  # Light Blue
            '#F8C471',  # Peach
            '#82E0AA',  # Light Green
            '#F1948A',  # Salmon
            '#85C1E9',  # Powder Blue
            '#D7BDE2'   # Light Purple
        ]
            
        return [self.update_labelings(project_id, {
            "name": label,
            "color": random.choice(pretty_colors)
        }) for label in labels]
    
    def update_labelings(self, project_id, label):
        """Update labelings for a specific project by appending a new label"""
        logger.info(f'Updating labelings for project {project_id} with label: {label}')
        return self.project_repo.update_labelings(project_id, label)
        
    def update_labeling_color(self, project_id, labeling_name, color):
        """Update the color of an existing labeling in a project
        
        Args:
            project_id: ID of the project containing the labeling
            labeling_name: Name of the labeling to update
            color: New color value in hex format (e.g., '#FF0000')
            
        Returns:
            dict: Status and message indicating success or failure
        """
        logger.info(f'Updating color for labeling "{labeling_name}" to {color} in project {project_id}')
        return self.project_repo.update_labeling_color(project_id, labeling_name, color)
        
    def rename_labeling(self, project_id, old_name, new_name):
        """Rename an existing labeling in a project
        
        Args:
            project_id: ID of the project containing the labeling
            old_name: Current name of the labeling
            new_name: New name for the labeling
            
        Returns:
            dict: Status and message indicating success or failure
        """
        logger.info(f'Renaming labeling from "{old_name}" to "{new_name}" in project {project_id}')
        return self.project_repo.rename_labeling(project_id, old_name, new_name)

    def delete_labeling(self, project_id, labeling_name):
        """Mark a labeling as deleted in a project
        
        Args:
            project_id: ID of the project containing the labeling
            labeling_name: Name of the labeling to delete
            
        Returns:
            dict: Status and message indicating success or failure
        """
        logger.info(f'Marking labeling "{labeling_name}" as deleted in project {project_id}')
        return self.project_repo.delete_labeling(project_id, labeling_name)

    def discover_project_sessions(self, project_path):
        """
        Discover session directories within a project path
        
        Args:
            project_path: Path to the project directory
            
        Returns:
            list: List of session dictionaries with name and file information, sorted by date
        """
        sessions = []
        if os.path.exists(project_path):
            for item in os.listdir(project_path):
                item_path = os.path.join(project_path, item)
                if os.path.isdir(item_path):
                    accel_file = os.path.join(item_path, 'accelerometer_data.csv')
                    if os.path.exists(accel_file):
                        sessions.append({'name': item, 'file': 'accelerometer_data.csv'})
                    
            # Sort sessions by date/time in the name
            try:
                from datetime import datetime
                sessions.sort(key=lambda s: datetime.strptime('_'.join(s['name'].split('_')[:4]), '%Y-%m-%d_%H_%M_%S'))
            except:
                # If sorting fails, keep original order
                pass
        
        return sessions

    def update_project_participant(self, project_id, new_participant_id):
        """Update which participant a project is assigned to"""
        # Verify the project exists
        project = self.get_project_with_participant(project_id)
        if not project:
            raise DatabaseError('Project not found')
        
        # Verify the new participant exists
        participant = self.participant_repo.find_by_id(new_participant_id)
        if not participant:
            raise DatabaseError('Participant not found')
        
        old_participant_id = project['participant_id']
        
        # Update the project assignment
        self.project_repo.update_participant(project_id, new_participant_id)
        
        # Clean up old participant if they have no remaining projects
        self.cleanup_participant_if_needed(old_participant_id)
        
        return {
            'project_id': project_id,
            'old_participant_id': old_participant_id,
            'new_participant_id': new_participant_id,
            'participant_cleaned_up': old_participant_id != new_participant_id
        }

    def group_files_by_project_directories(self, uploaded_files):
        """
        Group uploaded files by project directories for bulk upload.
        
        Args:
            uploaded_files: List of FileStorage objects from Flask request
            
        Returns:
            dict: Dictionary where keys are project names and values are lists of files
        """
        project_groups = {}
        
        for file in uploaded_files:
            # Use filename which contains the relative path from directory upload
            relative_path = file.filename
            if relative_path:
                path_parts = relative_path.split('/')
                if len(path_parts) >= 2:
                    # First level is the main folder, second level is project folder
                    project_name = path_parts[1]
                    if project_name not in project_groups:
                        project_groups[project_name] = []
                    project_groups[project_name].append(file)
        
        logger.info(f"Grouped {len(uploaded_files)} files into {len(project_groups)} project directories")
        for project_name, files in project_groups.items():
            logger.debug(f"  Project '{project_name}': {len(files)} files")
            
        return project_groups