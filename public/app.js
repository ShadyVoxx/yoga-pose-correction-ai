document.addEventListener('DOMContentLoaded', () => {
  const els = {
    poseSelect: document.getElementById('pose-select'),
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
    setupSection: document.getElementById('setup-section'),
    liveSection: document.getElementById('live-section'),
    video: document.getElementById('webcam-video'),
    canvas: document.getElementById('capture-canvas'),
    stepNumber: document.getElementById('step-number'),
    stepInstruction: document.getElementById('step-instruction'),
    aiFeedback: document.getElementById('ai-feedback'),
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text')
  };

  let posesData = [];
  let currentPose = null;
  let currentStepIndex = 0;
  let stream = null;
  let analysisInterval = null;
  let isAnalyzing = false;
  let stepAdvanceTimeout = null;

  // 1. Fetch poses
  fetch('/api/poses')
    .then(res => res.json())
    .then(poses => {
      posesData = poses;
      els.poseSelect.innerHTML = '<option value="" disabled selected>Select a pose...</option>';
      poses.forEach(pose => {
        if(pose.steps && pose.steps.length > 0) {
          const opt = document.createElement('option');
          opt.value = pose.name;
          opt.textContent = pose.name;
          els.poseSelect.appendChild(opt);
        }
      });
    });

  // 2. Start Practice
  els.btnStart.addEventListener('click', async () => {
    const selectedName = els.poseSelect.value;
    if (!selectedName) {
      alert('Please select a pose first.');
      return;
    }
    
    currentPose = posesData.find(p => p.name === selectedName);
    currentStepIndex = 0;
    
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
      els.video.srcObject = stream;
      
      els.setupSection.hidden = true;
      els.liveSection.hidden = false;
      
      updateStepUI();
      
      // Start loop (every 4.5 seconds to respect rate limits while maintaining real-time feel)
      analysisInterval = setInterval(captureAndAnalyze, 4500); 
      els.statusDot.className = 'status-dot recording';
      els.statusText.textContent = 'Watching...';
      
      // Do initial capture immediately after camera warms up
      setTimeout(captureAndAnalyze, 1000);
      
    } catch (err) {
      alert('Could not access webcam. Please allow camera permissions. ' + err.message);
    }
  });

  // 3. Stop Practice
  els.btnStop.addEventListener('click', stopPractice);

  function stopPractice() {
    if (analysisInterval) clearInterval(analysisInterval);
    if (stepAdvanceTimeout) clearTimeout(stepAdvanceTimeout);
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
    }
    els.setupSection.hidden = false;
    els.liveSection.hidden = true;
    isAnalyzing = false;
  }

  function updateStepUI() {
    if (currentStepIndex < currentPose.steps.length) {
      els.stepNumber.textContent = `Step ${currentStepIndex + 1} of ${currentPose.steps.length}`;
      els.stepInstruction.textContent = currentPose.steps[currentStepIndex];
      els.aiFeedback.textContent = 'Waiting for your position...';
      els.aiFeedback.className = 'ai-feedback';
    } else {
      els.stepNumber.textContent = 'Practice Complete!';
      els.stepInstruction.textContent = 'Great job finishing the pose sequence.';
      els.aiFeedback.textContent = '';
      els.aiFeedback.className = 'ai-feedback success';
      if (analysisInterval) clearInterval(analysisInterval);
      els.statusDot.className = 'status-dot';
      els.statusText.textContent = 'Finished';
    }
  }

  async function captureAndAnalyze() {
    // Prevent overlapping requests or analyzing if finished
    if (isAnalyzing || currentStepIndex >= currentPose.steps.length) return;
    
    isAnalyzing = true;
    els.statusDot.className = 'status-dot analyzing';
    els.statusText.textContent = 'AI Checking...';
    
    // Capture frame from video
    const ctx = els.canvas.getContext('2d');
    els.canvas.width = els.video.videoWidth || 640;
    els.canvas.height = els.video.videoHeight || 480;
    ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);
    
    // Compress heavily to save bandwidth and API payload size (0.5 quality)
    const imageBase64 = els.canvas.toDataURL('image/jpeg', 0.5);
    
    try {
      const res = await fetch('/api/analyze-frame', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          poseName: currentPose.name,
          currentStepIndex: currentStepIndex,
          imageBase64: imageBase64
        })
      });
      
      const data = await res.json();
      
      if (data.success && data.analysis) {
        const ai = data.analysis;
        
        if (ai.stepComplete) {
          els.aiFeedback.className = 'ai-feedback success';
          els.aiFeedback.textContent = '✅ ' + (ai.feedback || 'Perfect, moving on!');
          
          // Pause analysis while transitioning
          isAnalyzing = true; 
          
          // Advance to next step after 2.5 seconds so they can read the praise
          stepAdvanceTimeout = setTimeout(() => {
            currentStepIndex++;
            updateStepUI();
            isAnalyzing = false; // unblock analysis
          }, 2500);
          
        } else {
          els.aiFeedback.className = 'ai-feedback';
          els.aiFeedback.innerHTML = '<strong>Adjustment:</strong> ' + (ai.feedback || 'Keep adjusting.');
          isAnalyzing = false;
        }
      } else {
        isAnalyzing = false;
      }
    } catch (e) {
      console.error(e);
      isAnalyzing = false;
    } finally {
      if (!isAnalyzing && currentStepIndex < currentPose.steps.length) {
        els.statusDot.className = 'status-dot recording';
        els.statusText.textContent = 'Watching...';
      }
    }
  }
});
