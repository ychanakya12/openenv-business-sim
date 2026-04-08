from src.server import app

def main():
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()
