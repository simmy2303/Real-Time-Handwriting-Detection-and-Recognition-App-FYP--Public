// script.js
window.addEventListener("DOMContentLoaded", () => {
  // 🎯 Grab all elements once
  const upload      = document.getElementById("upload");
  const cameraBtn   = document.getElementById("camera-btn");
  const galleryBtn  = document.getElementById("gallery-btn");
  const copyBtn     = document.getElementById("copy-btn");
  const saveBtn     = document.getElementById("save-btn");
  const outputImg   = document.getElementById("output");
  const textOutput  = document.getElementById("text-output");
  const downloadBtn = document.getElementById("download-btn");
  const loadingDiv  = document.getElementById("loading");

  // Status management for the new UI
  function updateStatus(message, type = 'info') {
    const statusIndicator = document.querySelector('.status-indicator span');
    if (statusIndicator) {
      statusIndicator.textContent = message;
      
      // Update status dot color based on type
      const statusDot = document.querySelector('.status-dot');
      if (statusDot) {
        statusDot.style.background = type === 'success' ? '#10b981' : 
                                   type === 'error' ? '#ef4444' : 
                                   type === 'warning' ? '#f59e0b' : '#10b981';
      }
    }
  }

  // Enhanced loading state management
  function showLoading() {
    loadingDiv.style.display = "flex";
    updateStatus("Processing image...", 'warning');
  }

  function hideLoading() {
    loadingDiv.style.display = "none";
    updateStatus("Ready for text recognition", 'success');
  }

  // Enhanced text display with placeholder handling
  function displayText(text) {
    const placeholderDiv = textOutput.querySelector('.placeholder-text');
    if (placeholderDiv) {
      placeholderDiv.remove();
    }
    textOutput.textContent = text;
    textOutput.style.color = document.body.classList.contains('high-contrast') ? '#00ff00' : '#ffffff';
  }

  function showPlaceholder() {
    textOutput.innerHTML = `
      <div class="placeholder-text">
        <i class="fas fa-text-width"></i>
        Recognized text will appear here...
      </div>
    `;
  }

  // 🔊 Enhanced TTS helper with better error handling
  function speak(text, options = {}) {
    try {
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.rate = options.rate || 1;
      u.volume = options.volume || 1;
      u.pitch = options.pitch || 1;
      
      u.onstart = () => {
        updateStatus("Speaking...", 'info');
      };
      
      u.onend = () => {
        updateStatus("Ready for text recognition", 'success');
      };
      
      u.onerror = (e) => {
        console.error('Speech synthesis error:', e);
        updateStatus("Speech error occurred", 'error');
      };
      
      window.speechSynthesis.speak(u);
    } catch (error) {
      console.error('TTS Error:', error);
      updateStatus("Text-to-speech unavailable", 'error');
    }
  }

  // — Global speech-control functions (used by inline onclicks in index.html) —
  window.speakText = () => {
    const txt = textOutput.textContent.trim();
    if (!txt || txt.includes('Recognized text will appear here')) {
      updateStatus("No text to read", 'warning');
      return;
    }
    
    const rateSelect = document.getElementById("rate");
    const rate = rateSelect ? parseFloat(rateSelect.value) : 1;
    speak(txt, { rate });
  };

  window.pauseSpeech = () => {
    if (window.speechSynthesis.speaking && !window.speechSynthesis.paused) {
      window.speechSynthesis.pause();
      updateStatus("Speech paused", 'warning');
    }
  };

  window.resumeSpeech = () => {
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
      updateStatus("Speech resumed", 'info');
    }
  };

  window.stopSpeech = () => {
    window.speechSynthesis.cancel();
    updateStatus("Speech stopped", 'info');
  };

  window.clearAll = () => {
    // Clear preview image
    outputImg.src = "";
    outputImg.style.display = "none";
    
    // Reset text output with placeholder
    showPlaceholder();
    
    // Reset file input
    upload.value = null;
    
    // Hide annotated-image download button
    downloadBtn.hidden = true;
    
    // Stop any ongoing speech
    window.speechSynthesis.cancel();
    
    updateStatus("Cleared all content", 'info');
    speak("Content cleared");
  };

  window.toggleHighContrast = () => {
    document.body.classList.toggle("high-contrast");
    const isHighContrast = document.body.classList.contains("high-contrast");
    
    updateStatus(isHighContrast ? "High contrast enabled" : "High contrast disabled", 'info');
    speak(isHighContrast ? "High contrast mode enabled" : "High contrast mode disabled");
    
    // Update button visual state
    const contrastBtn = document.querySelector('[onclick="toggleHighContrast()"]');
    if (contrastBtn) {
      if (isHighContrast) {
        contrastBtn.classList.add('active');
      } else {
        contrastBtn.classList.remove('active');
      }
    }
  };

  // — Enhanced welcome message with better user interaction handling —
  const welcomeMessage = "Welcome to Snap 2 Speech. Double-tap to open the camera, triple-tap to repeat this message.";
  let hasPlayed = false;

  function onFirstInteraction(e) {
    // Ignore taps on control elements
    if (e.target.closest("button") || 
        e.target.closest(".controls") || 
        e.target.closest(".upload-section") ||
        e.target.closest("select")) {
      return;
    }
    
    if (!hasPlayed) {
      speak(welcomeMessage);
      hasPlayed = true;
      updateStatus("Welcome message played", 'success');
    }
    
    // Remove listeners after first interaction
    document.body.removeEventListener("touchstart", onFirstInteraction);
    document.body.removeEventListener("click", onFirstInteraction);
  }

  // Bind interaction listeners for mobile TTS unlock
  document.body.addEventListener("touchstart", onFirstInteraction, { once: true });
  document.body.addEventListener("click", onFirstInteraction, { once: true });

  // Try auto-play welcome message (will fail on most modern browsers)
  setTimeout(() => {
    if (!hasPlayed) {
      const tryUtter = new SpeechSynthesisUtterance(welcomeMessage);
      tryUtter.onstart = () => { 
        hasPlayed = true; 
        updateStatus("Welcome message played", 'success');
      };
      tryUtter.onerror = () => {
        updateStatus("Tap anywhere to enable audio", 'info');
      };
      window.speechSynthesis.speak(tryUtter);
    }
  }, 500);

  // — Enhanced double/triple tap logic —
  let tapCount = 0, tapTimer = null;
  const TAP_THRESHOLD = 400;

  document.body.addEventListener("click", (e) => {
    // Skip if first interaction hasn't occurred
    if (!hasPlayed) return;

    // Ignore taps on interactive elements
    if (e.target.closest("button") || 
        e.target.closest(".controls") || 
        e.target.closest(".upload-section") ||
        e.target.closest("select") ||
        e.target.closest("input")) {
      return;
    }

    tapCount++;
    clearTimeout(tapTimer);
    
    tapTimer = setTimeout(() => {
      if (tapCount >= 3) {
        speak("Welcome to Snap 2 Speech. Double-tap to open the camera. Triple-tap to repeat instructions.");
        updateStatus("Instructions repeated", 'info');
      } else if (tapCount === 2) {
        speak("Opening camera");
        updateStatus("Opening camera...", 'info');
        upload.setAttribute("capture", "environment");
        upload.click();
      }
      tapCount = 0;
    }, TAP_THRESHOLD);
  });

  // — Enhanced upload flow with better error handling —
  upload.addEventListener("change", async () => {
    const file = upload.files[0];
    if (!file) return;

    showLoading();
    displayText("");

    // Preview the image
    const reader = new FileReader();
    reader.onload = (e) => {
      outputImg.src = e.target.result;
      outputImg.style.display = "block";
    };
    reader.onerror = () => {
      updateStatus("Error reading file", 'error');
      speak("Error reading file");
    };
    
    speak("Image selected, Image processsing");
    updateStatus("Image selected, processing...", 'info');
    reader.readAsDataURL(file);

    // Send to server
    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await fetch("/infer/image", {
        method: "POST",
        body: fd
      });

      if (!res.ok) {
        throw new Error(`HTTP error! status: ${res.status}`);
      }

      const json = await res.json();
      
      // Handle the response
      if (json.recognized_text && json.recognized_text.length > 0) {
        const recognizedText = json.recognized_text.join("\n");
        displayText(recognizedText);
        updateStatus("Text recognized successfully", 'success');
        
        // Auto-read the recognized text
        setTimeout(() => {
          window.speakText();
        }, 500);
      } else {
        displayText("[No text found in image]");
        updateStatus("No text found in image", 'warning');
        speak("No text found in the image");
      }

      // Handle annotated image download
      if (json.image_url) {
        downloadBtn.hidden = false;
        downloadBtn.onclick = () => {
          window.open(json.image_url, "_blank");
          updateStatus("Downloading annotated image", 'info');
        };
      }

    } catch (err) {
      console.error("Upload error:", err);
      displayText("Error occurred during processing.");
      updateStatus("Processing failed", 'error');
      speak("Error occurred during processing");
      
      // Show more specific error messages
      if (err.message.includes('Failed to fetch')) {
        updateStatus("Network error - check connection", 'error');
      } else if (err.message.includes('413')) {
        updateStatus("File too large", 'error');
      }
    } finally {
      hideLoading();
    }
  });

  // Enhanced camera button
  cameraBtn.addEventListener("click", () => {
    upload.setAttribute("capture", "environment");
    upload.click();
    speak("Opening camera...")
    updateStatus("Opening camera...", 'info');
  });

  // Enhanced gallery button
  galleryBtn.addEventListener("click", () => {
    upload.removeAttribute("capture");
    upload.click();
    speak("Opening file picker...")
    updateStatus("Opening file picker...", 'info');
  });

  // — Enhanced copy functionality —
  copyBtn.addEventListener("click", async () => {
    const txt = textOutput.textContent.trim();
    if (!txt || txt.includes('Recognized text will appear here')) {
      updateStatus("Nothing to copy", 'warning');
      return;
    }

    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(txt);
        updateStatus("Text copied to clipboard", 'success');
        speak("Text copied");
      } else {
        fallbackCopy(txt);
      }
    } catch (error) {
      console.error('Copy error:', error);
      fallbackCopy(txt);
    }
  });

  function fallbackCopy(str) {
    try {
      const ta = document.createElement("textarea");
      ta.value = str;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      const success = document.execCommand("copy");
      document.body.removeChild(ta);
      
      if (success) {
        updateStatus("Text copied to clipboard", 'success');
        speak("Text copied");
      } else {
        updateStatus("Copy failed", 'error');
      }
    } catch (error) {
      console.error('Fallback copy error:', error);
      updateStatus("Copy not supported", 'error');
    }
  }

  // — Enhanced save functionality —
  saveBtn.addEventListener("click", () => {
    const txt = textOutput.textContent.trim();
    if (!txt || txt.includes('Recognized text will appear here')) {
      updateStatus("Nothing to save", 'warning');
      return;
    }

    try {
      const blob = new Blob([txt], { type: "text/plain;charset=utf-8" });
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      
      // Generate filename with timestamp
      const now = new Date();
      const timestamp = now.toISOString().slice(0, 19).replace(/[:.]/g, '-');
      link.download = `recognized_text_${timestamp}.txt`;
      
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Clean up the blob URL
      setTimeout(() => URL.revokeObjectURL(link.href), 100);
      
      updateStatus("Text saved successfully", 'success');
      speak("Text saved");
    } catch (error) {
      console.error('Save error:', error);
      updateStatus("Save failed", 'error');
    }
  });

  // — Keyboard shortcuts —
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey) {
      switch(e.key.toLowerCase()) {
        case 'c':
          if (textOutput.textContent.trim() && !textOutput.textContent.includes('Recognized text will appear here')) {
            e.preventDefault();
            copyBtn.click();
          }
          break;
        case 's':
          if (textOutput.textContent.trim() && !textOutput.textContent.includes('Recognized text will appear here')) {
            e.preventDefault();
            saveBtn.click();
          }
          break;
        case ' ':
          e.preventDefault();
          window.speakText();
          break;
        case 'r':
          e.preventDefault();
          window.clearAll();
          break;
      }
    }
    
    // Space bar for play/pause
    if (e.code === 'Space' && !e.target.matches('input, textarea, select, button')) {
      e.preventDefault();
      if (window.speechSynthesis.speaking) {
        if (window.speechSynthesis.paused) {
          window.resumeSpeech();
        } else {
          window.pauseSpeech();
        }
      } else {
        window.speakText();
      }
    }
  });

  // Initialize status
  updateStatus("Ready for text recognition", 'success');
  
  // Initialize placeholder
  showPlaceholder();
});