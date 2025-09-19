import logging
import shutil
from pathlib import Path
from typing import Set
import atexit

logger = logging.getLogger(__name__)


class CleanupManager:    
    def __init__(self):
        self.directories_to_cleanup: Set[Path] = set()
        self.files_to_cleanup: Set[Path] = set()
        self.cleanup_on_exit = True
        
        atexit.register(self._exit_cleanup)
        
        logger.info("CleanupManager initialized")
    
    def add_directory(self, directory_path: Path) -> None:
        if directory_path.exists():
            self.directories_to_cleanup.add(directory_path.resolve())
            logger.debug(f"Added directory for cleanup: {directory_path}")
    
    def add_file(self, file_path: Path) -> None:
        if file_path.exists():
            self.files_to_cleanup.add(file_path.resolve())
            logger.debug(f"Added file for cleanup: {file_path}")
    
    def remove_directory(self, directory_path: Path) -> None:
        resolved_path = directory_path.resolve()
        self.directories_to_cleanup.discard(resolved_path)
        logger.debug(f"Removed directory from cleanup: {directory_path}")
    
    def remove_file(self, file_path: Path) -> None:
        resolved_path = file_path.resolve()
        self.files_to_cleanup.discard(resolved_path)
        logger.debug(f"Removed file from cleanup: {file_path}")
    
    def cleanup_all(self) -> None:
        logger.info("Starting cleanup of temporary resources")
        
        files_cleaned = 0
        for file_path in list(self.files_to_cleanup):
            if self._cleanup_file(file_path):
                files_cleaned += 1
                self.files_to_cleanup.discard(file_path)

        dirs_cleaned = 0
        for directory_path in list(self.directories_to_cleanup):
            if self._cleanup_directory(directory_path):
                dirs_cleaned += 1
                self.directories_to_cleanup.discard(directory_path)
        
        logger.info(f"Cleanup completed: {files_cleaned} files, {dirs_cleaned} directories")
    
    def cleanup_directory(self, directory_path: Path) -> bool:
        return self._cleanup_directory(directory_path)
    
    def cleanup_file(self, file_path: Path) -> bool:
        return self._cleanup_file(file_path)
    
    def _cleanup_directory(self, directory_path: Path) -> bool:
        try:
            if directory_path.exists() and directory_path.is_dir():
                shutil.rmtree(directory_path)
                logger.debug(f"Cleaned up directory: {directory_path}")
                return True
            else:
                logger.debug(f"Directory does not exist or is not a directory: {directory_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clean up directory {directory_path}: {e}")
            return False
    
    def _cleanup_file(self, file_path: Path) -> bool:
        try:
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                logger.debug(f"Cleaned up file: {file_path}")
                return True
            else:
                logger.debug(f"File does not exist or is not a file: {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to clean up file {file_path}: {e}")
            return False
    
    def _exit_cleanup(self) -> None:
        if self.cleanup_on_exit and (self.directories_to_cleanup or self.files_to_cleanup):
            logger.info("Performing cleanup on exit")
            self.cleanup_all()
    
    def disable_exit_cleanup(self) -> None:
        self.cleanup_on_exit = False
        logger.info("Automatic exit cleanup disabled")
    
    def enable_exit_cleanup(self) -> None:
        self.cleanup_on_exit = True
        logger.info("Automatic exit cleanup enabled")
    
    def get_cleanup_stats(self) -> dict:
        return {
            'directories_pending': len(self.directories_to_cleanup),
            'files_pending': len(self.files_to_cleanup),
            'cleanup_on_exit': self.cleanup_on_exit,
            'directories': [str(d) for d in self.directories_to_cleanup],
            'files': [str(f) for f in self.files_to_cleanup]
        }
    
    def is_empty(self) -> bool:
        return len(self.directories_to_cleanup) == 0 and len(self.files_to_cleanup) == 0
