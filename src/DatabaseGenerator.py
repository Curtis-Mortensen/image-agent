"""
Handles database initialization and schema management.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional
from config import DATABASE_PATH, DATABASE_CONFIG

logger = logging.getLogger(__name__)

class DatabaseGenerator:
    """Manages database initialization and updates."""

    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = Path(db_path)
        self.version = DATABASE_CONFIG["version"]

    def _create_tables(self, conn: sqlite3.Connection):
        """Create all database tables according to schema."""
        try:
            # Create tables
            for table_name, table_config in DATABASE_CONFIG["tables"].items():
                columns = ",\n    ".join(table_config["columns"])
                create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {columns}
                );
                """
                conn.execute(create_table_sql)
                
                # Create indices if specified
                if "indices" in table_config:
                    for index_sql in table_config["indices"]:
                        conn.execute(index_sql)
                        
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            raise

    def _init_version_info(self, conn: sqlite3.Connection):
        """Initialize or update version information."""
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS version_info (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert or update version
            conn.execute("""
                INSERT OR REPLACE INTO version_info (id, version, updated_at)
                VALUES (1, ?, CURRENT_TIMESTAMP)
            """, (self.version,))
            
        except Exception as e:
            logger.error(f"Error initializing version info: {str(e)}")
            raise

    def initialize_database(self):
        """Initialize the database with all required tables."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                self._init_version_info(conn)
                self._create_tables(conn)
                conn.commit()
                
            logger.info(f"Database initialized successfully at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise

    def get_version(self) -> Optional[str]:
        """Get the current database version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT version FROM version_info WHERE id = 1
                """)
                result = cursor.fetchone()
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Error getting database version: {str(e)}")
            return None

    def needs_update(self) -> bool:
        """Check if database needs updating."""
        current_version = self.get_version()
        return current_version != self.version if current_version else True

def initialize_database(db_path: str = DATABASE_PATH):
    """Convenience function to initialize database."""
    generator = DatabaseGenerator(db_path)
    generator.initialize_database()

if __name__ == "__main__":
    # Set up logging when run directly
    logging.basicConfig(level=logging.INFO)
    # Initialize database
    initialize_database() 
