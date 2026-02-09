"""
Database Migration Script for Iteration 2
Adds Privacy Preserver and Output Filter fields to existing tables.

Run this script to update your database schema:
    python migrations/add_iteration2_fields.py
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from utils.database import engine

def run_migration():
    """Add Iteration 2 fields to database tables."""
    
    print("=" * 60)
    print("ITERATION 2 DATABASE MIGRATION")
    print("=" * 60)
    
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            # Add Privacy Preserver fields to messages table
            print("\n📝 Adding Privacy Preserver fields to messages table...")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS original_content TEXT;
                """))
                print("   ✅ Added original_content column")
            except Exception as e:
                print(f"   ⚠️  original_content: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS masked_content TEXT;
                """))
                print("   ✅ Added masked_content column")
            except Exception as e:
                print(f"   ⚠️  masked_content: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS pii_mappings JSONB;
                """))
                print("   ✅ Added pii_mappings column")
            except Exception as e:
                # PostgreSQL might not have JSONB, try JSON
                try:
                    conn.execute(text("""
                        ALTER TABLE messages 
                        ADD COLUMN IF NOT EXISTS pii_mappings JSON;
                    """))
                    print("   ✅ Added pii_mappings column (as JSON)")
                except Exception as e2:
                    print(f"   ⚠️  pii_mappings: {e2}")
            
            # Add Output Filter fields to messages table
            print("\n🔍 Adding Output Filter fields to messages table...")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS filtered_content TEXT;
                """))
                print("   ✅ Added filtered_content column")
            except Exception as e:
                print(f"   ⚠️  filtered_content: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS bias_detected BOOLEAN DEFAULT FALSE;
                """))
                print("   ✅ Added bias_detected column")
            except Exception as e:
                print(f"   ⚠️  bias_detected: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS bias_score FLOAT;
                """))
                print("   ✅ Added bias_score column")
            except Exception as e:
                print(f"   ⚠️  bias_score: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS jailbreak_detected_in_output BOOLEAN DEFAULT FALSE;
                """))
                print("   ✅ Added jailbreak_detected_in_output column")
            except Exception as e:
                print(f"   ⚠️  jailbreak_detected_in_output: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS jailbreak_score_in_output FLOAT;
                """))
                print("   ✅ Added jailbreak_score_in_output column")
            except Exception as e:
                print(f"   ⚠️  jailbreak_score_in_output: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS filter_analysis JSONB;
                """))
                print("   ✅ Added filter_analysis column")
            except Exception as e:
                # PostgreSQL might not have JSONB, try JSON
                try:
                    conn.execute(text("""
                        ALTER TABLE messages 
                        ADD COLUMN IF NOT EXISTS filter_analysis JSON;
                    """))
                    print("   ✅ Added filter_analysis column (as JSON)")
                except Exception as e2:
                    print(f"   ⚠️  filter_analysis: {e2}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS sanitization_strategy VARCHAR;
                """))
                print("   ✅ Added sanitization_strategy column")
            except Exception as e:
                print(f"   ⚠️  sanitization_strategy: {e}")
            
            # Add Output Filter fields to conversations table
            print("\n📊 Adding Output Filter fields to conversations table...")
            
            try:
                conn.execute(text("""
                    ALTER TABLE conversations 
                    ADD COLUMN IF NOT EXISTS bias_detected BOOLEAN DEFAULT FALSE;
                """))
                print("   ✅ Added bias_detected column to conversations")
            except Exception as e:
                print(f"   ⚠️  bias_detected (conversations): {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE conversations 
                    ADD COLUMN IF NOT EXISTS jailbreak_detected_in_output BOOLEAN DEFAULT FALSE;
                """))
                print("   ✅ Added jailbreak_detected_in_output column to conversations")
            except Exception as e:
                print(f"   ⚠️  jailbreak_detected_in_output (conversations): {e}")
            
            trans.commit()
            print("\n" + "=" * 60)
            print("✅ MIGRATION COMPLETED SUCCESSFULLY")
            print("=" * 60)
            
        except Exception as e:
            trans.rollback()
            print(f"\n❌ MIGRATION FAILED: {e}")
            raise

if __name__ == "__main__":
    run_migration()


Adds Privacy Preserver and Output Filter fields to existing tables.

Run this script to update your database schema:
    python migrations/add_iteration2_fields.py
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from utils.database import engine

def run_migration():
    """Add Iteration 2 fields to database tables."""
    
    print("=" * 60)
    print("ITERATION 2 DATABASE MIGRATION")
    print("=" * 60)
    
    with engine.connect() as conn:
        trans = conn.begin()
        try:
            # Add Privacy Preserver fields to messages table
            print("\n📝 Adding Privacy Preserver fields to messages table...")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS original_content TEXT;
                """))
                print("   ✅ Added original_content column")
            except Exception as e:
                print(f"   ⚠️  original_content: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS masked_content TEXT;
                """))
                print("   ✅ Added masked_content column")
            except Exception as e:
                print(f"   ⚠️  masked_content: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS pii_mappings JSONB;
                """))
                print("   ✅ Added pii_mappings column")
            except Exception as e:
                # PostgreSQL might not have JSONB, try JSON
                try:
                    conn.execute(text("""
                        ALTER TABLE messages 
                        ADD COLUMN IF NOT EXISTS pii_mappings JSON;
                    """))
                    print("   ✅ Added pii_mappings column (as JSON)")
                except Exception as e2:
                    print(f"   ⚠️  pii_mappings: {e2}")
            
            # Add Output Filter fields to messages table
            print("\n🔍 Adding Output Filter fields to messages table...")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS filtered_content TEXT;
                """))
                print("   ✅ Added filtered_content column")
            except Exception as e:
                print(f"   ⚠️  filtered_content: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS bias_detected BOOLEAN DEFAULT FALSE;
                """))
                print("   ✅ Added bias_detected column")
            except Exception as e:
                print(f"   ⚠️  bias_detected: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS bias_score FLOAT;
                """))
                print("   ✅ Added bias_score column")
            except Exception as e:
                print(f"   ⚠️  bias_score: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS jailbreak_detected_in_output BOOLEAN DEFAULT FALSE;
                """))
                print("   ✅ Added jailbreak_detected_in_output column")
            except Exception as e:
                print(f"   ⚠️  jailbreak_detected_in_output: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS jailbreak_score_in_output FLOAT;
                """))
                print("   ✅ Added jailbreak_score_in_output column")
            except Exception as e:
                print(f"   ⚠️  jailbreak_score_in_output: {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS filter_analysis JSONB;
                """))
                print("   ✅ Added filter_analysis column")
            except Exception as e:
                # PostgreSQL might not have JSONB, try JSON
                try:
                    conn.execute(text("""
                        ALTER TABLE messages 
                        ADD COLUMN IF NOT EXISTS filter_analysis JSON;
                    """))
                    print("   ✅ Added filter_analysis column (as JSON)")
                except Exception as e2:
                    print(f"   ⚠️  filter_analysis: {e2}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE messages 
                    ADD COLUMN IF NOT EXISTS sanitization_strategy VARCHAR;
                """))
                print("   ✅ Added sanitization_strategy column")
            except Exception as e:
                print(f"   ⚠️  sanitization_strategy: {e}")
            
            # Add Output Filter fields to conversations table
            print("\n📊 Adding Output Filter fields to conversations table...")
            
            try:
                conn.execute(text("""
                    ALTER TABLE conversations 
                    ADD COLUMN IF NOT EXISTS bias_detected BOOLEAN DEFAULT FALSE;
                """))
                print("   ✅ Added bias_detected column to conversations")
            except Exception as e:
                print(f"   ⚠️  bias_detected (conversations): {e}")
            
            try:
                conn.execute(text("""
                    ALTER TABLE conversations 
                    ADD COLUMN IF NOT EXISTS jailbreak_detected_in_output BOOLEAN DEFAULT FALSE;
                """))
                print("   ✅ Added jailbreak_detected_in_output column to conversations")
            except Exception as e:
                print(f"   ⚠️  jailbreak_detected_in_output (conversations): {e}")
            
            trans.commit()
            print("\n" + "=" * 60)
            print("✅ MIGRATION COMPLETED SUCCESSFULLY")
            print("=" * 60)
            
        except Exception as e:
            trans.rollback()
            print(f"\n❌ MIGRATION FAILED: {e}")
            raise

if __name__ == "__main__":
    run_migration()

