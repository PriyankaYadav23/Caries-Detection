## Terminal Commands Reference

### Environment
conda activate dl_projects

### FastAPI (local test)
uvicorn main:app --reload --port 8001

### Docker
docker build -t caries-detection-api .          # image make
docker run -p 8000:8000 caries-detection-api     # container run

### Git
git add .
git commit -m "message likho"
git push