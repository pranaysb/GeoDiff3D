const fs = require('fs');
const path = require('path');

async function runTest() {
  console.log("Starting End-to-End API Integration Test from Frontend Client...");
  
  const API_URL = "http://localhost:8000/api";
  
  // 1. Create a dummy image
  const dummyImagePath = path.join(__dirname, 'dummy.jpg');
  fs.writeFileSync(dummyImagePath, 'dummy content');
  
  // Create FormData
  const formData = new FormData();
  
  const blob1 = new Blob([fs.readFileSync(dummyImagePath)], { type: 'image/jpeg' });
  const blob2 = new Blob([fs.readFileSync(dummyImagePath)], { type: 'image/jpeg' });
  
  formData.append('images', blob1, 'img1.jpg');
  formData.append('images', blob2, 'img2.jpg');
  
  console.log("-> 1. Submitting images to /reconstruct");
  const res = await fetch(`${API_URL}/reconstruct`, {
    method: 'POST',
    body: formData
  });
  
  if (!res.ok) throw new Error("Reconstruct failed");
  const data = await res.json();
  const jobId = data.job_id;
  console.log(`-> Received Job ID: ${jobId}`);
  
  console.log("-> 2. Polling job state");
  let completed = false;
  while (!completed) {
    const jobRes = await fetch(`${API_URL}/jobs/${jobId}`);
    const jobData = await jobRes.json();
    console.log(`   State: ${jobData.state}`);
    
    if (jobData.state === 'completed') {
      completed = true;
      console.log(`-> 3. Job Completed! Scene ID: ${jobData.scene_id}`);
      
      console.log("-> 4. Verifying Artifacts");
      const artRes = await fetch(`${API_URL}/scene/${jobData.scene_id}/artifacts`);
      const artifacts = await artRes.json();
      
      if (artifacts.artifacts['baseline.ply'] && artifacts.artifacts['guided.ply']) {
        console.log("-> Success! PLY artifacts are ready for the 3D Viewer.");
      } else {
        throw new Error("Artifacts missing!");
      }
    } else if (jobData.state === 'failed') {
      throw new Error(`Job failed: ${jobData.message}`);
    }
    
    // wait 1 sec
    await new Promise(r => setTimeout(r, 1000));
  }
  
  // cleanup
  fs.unlinkSync(dummyImagePath);
  console.log("End-to-End flow test PASSED.");
}

runTest().catch(console.error);
