"""
Database initialization and management module.
Centralizes all database schema definitions and handles migrations.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional
from config import DATABASE_PATH

logger = logging.getLogger(__name__)

class DatabaseGenerator:
    """Handles database initialization and migrations."""
    
    VERSION = "1.0.0"  # Database schema version
    
    SCHEMA = {
        # Version tracking
        "version_info": """
            CREATE TABLE IF NOT EXISTS version_info (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                installed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        
        # Core tables
        "prompts": """
            CREATE TABLE IF NOT EXISTS prompts (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                scene TEXT NOT NULL,
                mood TEXT NOT NULL,
                prompt TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT 'flux',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """,
        
        "prompt_status": """
            CREATE TABLE IF NOT EXISTS prompt_status (
                prompt_id TEXT PRIMARY KEY,
                current_iteration INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prompt_id) REFERENCES prompts(id)
            )
        """,
        
        # Image generation tables
        "generated_images": """
            CREATE TABLE IF NOT EXISTS generated_images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                model TEXT NOT NULL DEFAULT 'flux',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'completed',
                UNIQUE(prompt_id, iteration),
                FOREIGN KEY (prompt_id) REFERENCES prompts(id)
            )
        """,
        
        # Refinement tables
        "refined_prompts": """
            CREATE TABLE IF NOT EXISTS refined_prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_prompt_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                refined_prompt TEXT NOT NULL,
                evaluation_text TEXT NOT NULL,
                needs_refinement BOOLEAN DEFAULT TRUE,
                model TEXT NOT NULL DEFAULT 'flux',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (original_prompt_id) REFERENCES prompts(id),
                UNIQUE(original_prompt_id, iteration)
            )
        """,
        
        # API tracking
        "api_calls": """
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                api_name TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
    }
    
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = Path(db_path)
        
    def initialize_database(self) -> None:
        """Initialize all database tables."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                # Create tables in order (respecting foreign key dependencies)
                for table_name, schema in self.SCHEMA.items():
                    logger.debug(f"Creating table: {table_name}")
                    conn.execute(schema)
                
                # Initialize version if needed
                self._initialize_version(conn)
                
                logger.info(f"Database initialized successfully at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def _initialize_version(self, conn: sqlite3.Connection) -> None:
        """Initialize or update version information."""
        cursor = conn.execute("SELECT version FROM version_info ORDER BY id DESC LIMIT 1")
        current_version = cursor.fetchone()
        
        if not current_version:
            conn.execute(
                "INSERT INTO version_info (version) VALUES (?)",
                (self.VERSION,)
            )
            logger.info(f"Initialized database version to {self.VERSION}")
        else:
            logger.debug(f"Current database version: {current_version[0]}")
    
    def get_version(self) -> Optional[str]:
        """Get current database version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT version FROM version_info ORDER BY id DESC LIMIT 1"
                )
                result = cursor.fetchone()
                return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to get database version: {str(e)}")
            return None
    
    def verify_database(self) -> bool:
        """Verify database structure and version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check each table exists
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table'
                """)
                existing_tables = {row[0] for row in cursor.fetchall()}
                
                # Verify all required tables exist
                missing_tables = set(self.SCHEMA.keys()) - existing_tables
                if missing_tables:
                    logger.warning(f"Missing tables: {missing_tables}")
                    return False
                
                # Verify version
                version = self.get_version()
                if version != self.VERSION:
                    logger.warning(
                        f"Version mismatch: current={version}, "
                        f"expected={self.VERSION}"
                    )
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Database verification failed: {str(e)}")
            return False

def initialize_database(db_path: str = DATABASE_PATH) -> None:
    """Convenience function to initialize database."""
    generator = DatabaseGenerator(db_path)
    generator.initialize_database()

if __name__ == "__main__":
    # Set up logging when run directly
    logging.basicConfig(level=logging.INFO)
    # Initialize database
    initialize_database() 
