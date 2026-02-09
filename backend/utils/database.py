from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URLs
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://username:password@localhost:5432/fortilm")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/fortilm")

# PostgreSQL setup
engine = create_engine(POSTGRES_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# MongoDB setup - use connection timeout to prevent hanging
try:
    mongo_client = MongoClient(
        MONGODB_URL,
        serverSelectionTimeoutMS=2000,  # 2 second timeout
        connectTimeoutMS=2000
    )
    # Test connection immediately
    mongo_client.server_info()
    mongo_db = mongo_client.fortilm
    _mongo_available = True
except Exception as e:
    print(f"⚠️ MongoDB not available: {e}")
    mongo_client = None
    mongo_db = None
    _mongo_available = False

def get_db():
    """Get PostgreSQL database session."""
    db = None
    try:
        db = SessionLocal()
        # Test connection with a simple query
        db.execute(text("SELECT 1"))
        yield db
    except Exception as e:
        print(f"⚠️ PostgreSQL connection error: {e}")
        # Create a dummy session that won't be used
        if db:
            db.close()
        # Yield a session that will be checked for None before use
        yield None
    finally:
        if db:
            db.close()

def get_mongo():
    """Get MongoDB database. Returns None if MongoDB is not available."""
    if not _mongo_available or mongo_db is None:
        return None
    return mongo_db



