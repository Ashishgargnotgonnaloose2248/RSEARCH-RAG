from src.db.postgres import get_connection

def test_connection():
    try:
        conn = get_connection()
        print("Connected to Postgres ✅")
        conn.close()
    except Exception as e:
        print("Connection failed ❌")
        print(e)

if __name__ == "__main__":
    test_connection()
