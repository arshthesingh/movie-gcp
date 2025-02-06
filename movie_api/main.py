import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    # Get the port from the environment (Cloud Run sets PORT=8080 by default)
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
