"""
Snowflake data ingestion script.
Loads heart disease data from S3 into Snowflake raw table.
Can be run locally (with .env) or in CI/CD (with environment variables).
"""

import os
import sys
from pathlib import Path
import snowflake.connector
from dotenv import load_dotenv

# Try to load .env if running locally (won't exist in GitHub Actions)
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Get environment variables (from .env locally or GitHub Actions secrets)
SNOWFLAKE_ACCOUNT = os.environ.get("SNOWFLAKE_ACCOUNT")
SNOWFLAKE_USER = os.environ.get("SNOWFLAKE_USER")
SNOWFLAKE_PASSWORD = os.environ.get("SNOWFLAKE_PASSWORD")
SNOWFLAKE_ROLE = os.environ.get("SNOWFLAKE_ROLE", "access")
SNOWFLAKE_WAREHOUSE = os.environ.get("SNOWFLAKE_WAREHOUSE", "cal_wh")
SNOWFLAKE_DATABASE = os.environ.get("SNOWFLAKE_DATABASE", "healthdata")
SNOWFLAKE_SCHEMA = os.environ.get("SNOWFLAKE_RAW_SCHEMA", "raw")
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
S3_BUCKET = os.environ.get("S3_BUCKET", "pritham-heartdata")
S3_PREFIX = os.environ.get("S3_PREFIX", "heart_disease_uci.csv")

# Validate required variables
required_vars = {
    "SNOWFLAKE_ACCOUNT": SNOWFLAKE_ACCOUNT,
    "SNOWFLAKE_USER": SNOWFLAKE_USER,
    "SNOWFLAKE_PASSWORD": SNOWFLAKE_PASSWORD,
    "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
    "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    print(f"Error: Missing required environment variables: {', '.join(missing_vars)}")
    sys.exit(1)

def execute_sql(cursor, sql, description):
    """Execute SQL and handle errors."""
    try:
        print(f"Executing: {description}")
        cursor.execute(sql)
        print(f"✓ Success: {description}")
        return True
    except Exception as e:
        print(f"✗ Error in {description}: {str(e)}")
        raise

def main():
    """Main ingestion function."""
    print("=" * 60)
    print("Starting Snowflake Data Ingestion")
    print("=" * 60)
    
    try:
        # Connect to Snowflake
        print(f"\nConnecting to Snowflake account: {SNOWFLAKE_ACCOUNT}")
        conn = snowflake.connector.connect(
            account=SNOWFLAKE_ACCOUNT,
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            role=SNOWFLAKE_ROLE,
            warehouse=SNOWFLAKE_WAREHOUSE,
            database=SNOWFLAKE_DATABASE,
        )
        cursor = conn.cursor()
        print("✓ Connected to Snowflake")
        
        # Set context
        execute_sql(cursor, f"USE WAREHOUSE {SNOWFLAKE_WAREHOUSE};", "Set warehouse")
        execute_sql(cursor, f"USE DATABASE {SNOWFLAKE_DATABASE};", "Set database")
        execute_sql(cursor, f"USE SCHEMA {SNOWFLAKE_RAW_SCHEMA};", "Set schema")
        
        # Create stage (access role will own it)
        create_stage_sql = f"""
        CREATE STAGE IF NOT EXIXTS healthstage
            URL = 's3://{S3_BUCKET}'
            CREDENTIALS = (
                AWS_KEY_ID = '{AWS_ACCESS_KEY_ID}'
                AWS_SECRET_KEY = '{AWS_SECRET_ACCESS_KEY}'
            );
        """
        execute_sql(cursor, create_stage_sql, "Create S3 stage")
        
        # Create table (access role will own it)
        create_table_sql = """
        CREATE OR REPLACE TABLE raw_health (
            id INTEGER,
            age INTEGER,
            sex VARCHAR(10),
            dataset VARCHAR(50),
            cp VARCHAR(50),
            trestbps INTEGER,
            chol INTEGER,
            fbs BOOLEAN,
            restecg VARCHAR(50),
            thalch INTEGER,
            exang BOOLEAN,
            oldpeak FLOAT,
            slope VARCHAR(50),
            ca INTEGER,
            thal VARCHAR(50),
            num INTEGER
        )
        CLUSTER BY (dataset, num);
        """
        execute_sql(cursor, create_table_sql, "Create raw_health table")
        
        # Copy data from S3
        copy_sql = f"""
        COPY INTO raw_health
        FROM '@healthstage/{S3_PREFIX}'
        FILE_FORMAT = (
            TYPE = 'CSV'
            SKIP_HEADER = 1
            FIELD_OPTIONALLY_ENCLOSED_BY = '"'
        );
        """
        execute_sql(cursor, copy_sql, f"Copy data from S3 ({S3_PREFIX})")
        
        # Verify row count
        cursor.execute("SELECT COUNT(*) FROM raw_health;")
        row_count = cursor.fetchone()[0]
        print(f"\n✓ Ingestion complete! Total rows in raw_health: {row_count}")
        
        # Close connection
        cursor.close()
        conn.close()
        print("\n" + "=" * 60)
        print("Ingestion completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

