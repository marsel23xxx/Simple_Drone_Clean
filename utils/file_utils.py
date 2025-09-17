"""
File operations and utilities
Handles file I/O, validation, and data persistence operations
"""

import os
import json
import csv
import shutil
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np

from config.settings import FILE_PATHS, DATA_DIR


class FileManager:
    """Centralized file management utilities."""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure all required directories exist."""
        directories = [
            self.data_dir,
            self.data_dir / 'logs',
            self.data_dir / 'exports',
            self.data_dir / 'backups',
            self.data_dir / 'point_clouds'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, data: Any, filepath: Union[str, Path], 
                  backup: bool = True) -> bool:
        """Save data to JSON file with optional backup."""
        try:
            filepath = Path(filepath)
            
            # Create backup if file exists and backup is requested
            if backup and filepath.exists():
                self.create_backup(filepath)
            
            # Write to temporary file first, then rename (atomic operation)
            temp_file = filepath.with_suffix('.tmp')
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Atomic rename
            temp_file.replace(filepath)
            
            print(f"Successfully saved data to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving JSON to {filepath}: {e}")
            return False
    
    def load_json(self, filepath: Union[str, Path], 
                  default: Any = None) -> Any:
        """Load data from JSON file with error handling."""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                print(f"JSON file {filepath} does not exist")
                return default
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return data
            
        except Exception as e:
            print(f"Error loading JSON from {filepath}: {e}")
            return default
    
    def save_csv(self, data: List[Dict], filepath: Union[str, Path], 
                 fieldnames: Optional[List[str]] = None) -> bool:
        """Save data to CSV file."""
        try:
            filepath = Path(filepath)
            
            if not data:
                print("No data to save to CSV")
                return False
            
            if fieldnames is None:
                fieldnames = list(data[0].keys())
            
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            print(f"Successfully saved CSV to {filepath}")
            return True
            
        except Exception as e:
            print(f"Error saving CSV to {filepath}: {e}")
            return False
    
    def load_csv(self, filepath: Union[str, Path]) -> List[Dict]:
        """Load data from CSV file."""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                print(f"CSV file {filepath} does not exist")
                return []
            
            data = []
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = list(reader)
            
            return data
            
        except Exception as e:
            print(f"Error loading CSV from {filepath}: {e}")
            return []
    
    def create_backup(self, filepath: Union[str, Path]) -> Optional[Path]:
        """Create backup of file with timestamp."""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{filepath.stem}_{timestamp}{filepath.suffix}"
            backup_path = self.data_dir / 'backups' / backup_name
            
            shutil.copy2(filepath, backup_path)
            print(f"Backup created: {backup_path}")
            
            return backup_path
            
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None
    
    def clean_old_backups(self, max_backups: int = 10):
        """Clean old backup files, keeping only the most recent ones."""
        try:
            backup_dir = self.data_dir / 'backups'
            
            if not backup_dir.exists():
                return
            
            # Get all backup files sorted by modification time
            backup_files = []
            for file in backup_dir.iterdir():
                if file.is_file():
                    backup_files.append((file.stat().st_mtime, file))
            
            backup_files.sort(reverse=True)  # Most recent first
            
            # Remove old backups
            for _, file_path in backup_files[max_backups:]:
                file_path.unlink()
                print(f"Removed old backup: {file_path}")
                
        except Exception as e:
            print(f"Error cleaning old backups: {e}")
    
    def export_data(self, data: Dict, filename: str, 
                   format_type: str = 'json') -> Optional[Path]:
        """Export data in various formats."""
        try:
            export_dir = self.data_dir / 'exports'
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format_type.lower() == 'json':
                filepath = export_dir / f"{filename}_{timestamp}.json"
                success = self.save_json(data, filepath, backup=False)
            elif format_type.lower() == 'csv':
                filepath = export_dir / f"{filename}_{timestamp}.csv"
                if isinstance(data, dict) and 'waypoints' in data:
                    csv_data = self._convert_waypoints_to_csv(data['waypoints'])
                    success = self.save_csv(csv_data, filepath)
                else:
                    print("Invalid data format for CSV export")
                    return None
            else:
                print(f"Unsupported export format: {format_type}")
                return None
            
            if success:
                print(f"Data exported to: {filepath}")
                return filepath
            
        except Exception as e:
            print(f"Error exporting data: {e}")
        
        return None
    
    def _convert_waypoints_to_csv(self, waypoints: List[Dict]) -> List[Dict]:
        """Convert waypoint data to CSV format."""
        csv_data = []
        for i, wp in enumerate(waypoints):
            if isinstance(wp, dict):
                pos = wp.get('position', (0, 0))
                csv_data.append({
                    'ID': i + 1,
                    'X': pos[0],
                    'Y': pos[1],
                    'Orientation': wp.get('orientation', 0.0),
                    'YawEnable': int(wp.get('yaw_enable', False)),
                    'Landing': int(wp.get('landing', False))
                })
            elif isinstance(wp, list) and len(wp) >= 5:
                csv_data.append({
                    'ID': i + 1,
                    'X': wp[0],
                    'Y': wp[1],
                    'Orientation': wp[2],
                    'YawEnable': wp[3],
                    'Landing': wp[4]
                })
        return csv_data
    
    def save_point_cloud(self, points: np.ndarray, 
                        colors: Optional[np.ndarray] = None,
                        filename: Optional[str] = None) -> Optional[Path]:
        """Save point cloud data to PLY format."""
        try:
            import open3d as o3d
            
            if len(points) == 0:
                print("No points to save")
                return None
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"pointcloud_{timestamp}.ply"
            
            filepath = self.data_dir / 'point_clouds' / filename
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            if colors is not None and len(colors) == len(points):
                pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Save to file
            success = o3d.io.write_point_cloud(str(filepath), pcd)
            
            if success:
                print(f"Point cloud saved to: {filepath}")
                return filepath
            else:
                print(f"Failed to save point cloud to: {filepath}")
                return None
                
        except Exception as e:
            print(f"Error saving point cloud: {e}")
            return None
    
    def load_point_cloud(self, filepath: Union[str, Path]) -> Optional[np.ndarray]:
        """Load point cloud data from PLY format."""
        try:
            import open3d as o3d
            
            filepath = Path(filepath)
            
            if not filepath.exists():
                print(f"Point cloud file {filepath} does not exist")
                return None
            
            pcd = o3d.io.read_point_cloud(str(filepath))
            
            if len(pcd.points) == 0:
                print(f"No points loaded from {filepath}")
                return None
            
            points = np.asarray(pcd.points)
            print(f"Loaded {len(points)} points from {filepath}")
            
            return points
            
        except Exception as e:
            print(f"Error loading point cloud from {filepath}: {e}")
            return None
    
    def create_project_archive(self, output_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Create a zip archive of the entire project data."""
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = self.data_dir / f"project_backup_{timestamp}.zip"
            
            output_path = Path(output_path)
            
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from data directory
                for file_path in self.data_dir.rglob('*'):
                    if file_path.is_file() and not file_path.name.endswith('.zip'):
                        arcname = file_path.relative_to(self.data_dir.parent)
                        zipf.write(file_path, arcname)
            
            print(f"Project archive created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating project archive: {e}")
            return None
    
    def extract_project_archive(self, archive_path: Union[str, Path],
                               extract_to: Optional[Union[str, Path]] = None) -> bool:
        """Extract project archive."""
        try:
            archive_path = Path(archive_path)
            
            if not archive_path.exists():
                print(f"Archive {archive_path} does not exist")
                return False
            
            if extract_to is None:
                extract_to = self.data_dir.parent
            
            extract_to = Path(extract_to)
            
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                zipf.extractall(extract_to)
            
            print(f"Project archive extracted to: {extract_to}")
            return True
            
        except Exception as e:
            print(f"Error extracting project archive: {e}")
            return False
    
    def get_file_info(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Get detailed file information."""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                return {'exists': False}
            
            stat = filepath.stat()
            
            return {
                'exists': True,
                'size': stat.st_size,
                'size_human': self._format_size(stat.st_size),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'created': datetime.fromtimestamp(stat.st_ctime),
                'is_file': filepath.is_file(),
                'is_directory': filepath.is_dir(),
                'extension': filepath.suffix,
                'name': filepath.name,
                'stem': filepath.stem
            }
            
        except Exception as e:
            print(f"Error getting file info: {e}")
            return {'exists': False, 'error': str(e)}
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        if size_bytes == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        i = 0
        
        while size_bytes >= 1024 and i < len(units) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {units[i]}"
    
    def validate_json_file(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Validate JSON file format and content."""
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                return {'valid': False, 'error': 'File does not exist'}
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return {
                'valid': True,
                'data_type': type(data).__name__,
                'size': len(data) if hasattr(data, '__len__') else 1,
                'keys': list(data.keys()) if isinstance(data, dict) else None
            }
            
        except json.JSONDecodeError as e:
            return {'valid': False, 'error': f'JSON decode error: {e}'}
        except Exception as e:
            return {'valid': False, 'error': f'File error: {e}'}
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            temp_patterns = ['*.tmp', '*.temp', '*~']
            
            for pattern in temp_patterns:
                for temp_file in self.data_dir.rglob(pattern):
                    if temp_file.is_file():
                        temp_file.unlink()
                        print(f"Removed temp file: {temp_file}")
                        
        except Exception as e:
            print(f"Error cleaning temp files: {e}")
    
    def get_directory_size(self, directory: Union[str, Path]) -> int:
        """Calculate total size of directory and subdirectories."""
        try:
            directory = Path(directory)
            
            if not directory.exists():
                return 0
            
            total_size = 0
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size
            
        except Exception as e:
            print(f"Error calculating directory size: {e}")
            return 0


# Global file manager instance
file_manager = FileManager()


def save_application_settings(settings: Dict[str, Any]) -> bool:
    """Save application settings to file."""
    return file_manager.save_json(settings, FILE_PATHS['settings_file'])


def load_application_settings() -> Dict[str, Any]:
    """Load application settings from file."""
    default_settings = {
        'window_geometry': None,
        'last_used_directory': str(DATA_DIR),
        'recent_files': [],
        'ui_preferences': {
            'theme': 'dark',
            'show_grid': True,
            'auto_save': True
        }
    }
    
    return file_manager.load_json(FILE_PATHS['settings_file'], default_settings)


def update_recent_files(filepath: str, max_recent: int = 10) -> bool:
    """Update recent files list."""
    try:
        settings = load_application_settings()
        recent_files = settings.get('recent_files', [])
        
        # Remove if already exists
        if filepath in recent_files:
            recent_files.remove(filepath)
        
        # Add to beginning
        recent_files.insert(0, filepath)
        
        # Limit to max_recent
        recent_files = recent_files[:max_recent]
        
        # Update settings
        settings['recent_files'] = recent_files
        
        return save_application_settings(settings)
        
    except Exception as e:
        print(f"Error updating recent files: {e}")
        return False