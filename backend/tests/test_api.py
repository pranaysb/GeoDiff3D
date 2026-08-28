from fastapi.testclient import TestClient
from app.main import app
import time

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_invalid_image_count():
    files = [("images", ("test1.jpg", b"dummy content", "image/jpeg"))]
    response = client.post("/api/reconstruct", files=files)
    assert response.status_code == 400
    assert "between 2 and 12" in response.json()["detail"]

def test_invalid_file_type():
    files = [
        ("images", ("test1.txt", b"dummy content", "text/plain")),
        ("images", ("test2.txt", b"dummy content", "text/plain"))
    ]
    response = client.post("/api/reconstruct", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_reconstruction_pipeline_e2e():
    files = [
        ("images", ("img1.jpg", b"123", "image/jpeg")),
        ("images", ("img2.jpg", b"123", "image/jpeg"))
    ]
    response = client.post("/api/reconstruct", files=files)
    assert response.status_code == 200
    data = response.json()
    job_id = data["job_id"]
    
    # Poll until completed
    max_retries = 30
    completed = False
    state = ""
    for _ in range(max_retries):
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        state = resp.json()["state"]
        if state == "completed":
            completed = True
            break
        elif state == "failed":
            break
        time.sleep(0.5)
        
    assert completed, f"Job did not complete. Last state: {state}"
    
    scene_id = resp.json()["scene_id"]
    
    scene_resp = client.get(f"/api/scene/{scene_id}")
    assert scene_resp.status_code == 200
    assert scene_resp.json()["image_count"] == 2
    
    art_resp = client.get(f"/api/scene/{scene_id}/artifacts")
    assert art_resp.status_code == 200
    artifacts = art_resp.json()["artifacts"]
    assert "baseline.ply" in artifacts
    assert "guided.ply" in artifacts
    
    del_resp = client.delete(f"/api/scene/{scene_id}")
    assert del_resp.status_code == 200
