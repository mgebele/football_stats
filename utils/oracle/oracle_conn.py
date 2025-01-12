import os
import oracledb
import pandas as pd
from dotenv import load_dotenv
from contextlib import contextmanager

class OracleDB:
    def __init__(self):
        self._setup_environment()
        
    def _setup_environment(self):
        load_dotenv()
        self._validate_env_vars()
        
    def _validate_env_vars(self):
        required = ["DB_USER", "DB_PASSWORD", "DB_DSN"]
        missing = [var for var in required if not os.getenv(var)]
        if missing:
            raise EnvironmentError(f"Missing environment variables: {', '.join(missing)}")
    
    def list_htdatan_tables(self) -> pd.DataFrame:
        query = """
        SELECT TABLE_NAME 
        FROM ALL_TABLES 
        WHERE REGEXP_LIKE(TABLE_NAME, '^(B_|L1_|LL_|PL_|SA_|XG_)')
        ORDER BY TABLE_NAME
        """
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                tables = cursor.fetchall()
                return pd.DataFrame([t[0] for t in tables], columns=["TABLE_NAME"])

    def show_table(self, table_name: str) -> pd.DataFrame:
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {table_name}")
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                processed_rows = []
                for row in rows:
                    processed_row = []
                    for item in row:
                        if isinstance(item, oracledb.LOB):
                            processed_item = item.read() if item is not None else None
                            processed_row.append(processed_item)
                        else:
                            processed_row.append(item)
                    processed_rows.append(processed_row)
                return pd.DataFrame(processed_rows, columns=columns)

    @contextmanager
    def get_connection(self, use_sqlalchemy: bool = False):
        try:
            if use_sqlalchemy:
                from sqlalchemy import create_engine
                engine = create_engine(
                    f'oracle+oracledb://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_DSN")}'
                )
                yield engine
            else:
                # Using thin mode by default (no thick client needed)
                conn = oracledb.connect(
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASSWORD"),
                    dsn=os.getenv("DB_DSN"),
                )
                yield conn
        except Exception as e:
            print(f"Database connection error: {e}")
            raise
        finally:
            if not use_sqlalchemy and 'conn' in locals():
                conn.close()
            elif use_sqlalchemy and 'engine' in locals():
                engine.dispose()