// Initialize audio recorder
const audioRecorder = new AudioRecorder();

// Global variables
let selectedModel = 'rule_based';

// DOM elements
const recordButton = document.getElementById('recordButton');
const recordStatus = document.getElementById('recordStatus');
const analyzeRecordedBtn = document.getElementById('analyzeRecordedBtn');
const resultsSection = document.getElementById('resultsSection');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsContent = document.getElementById('resultsContent');
const errorMessage = document.getElementById('errorMessage');
const tryAgainBtn = document.getElementById('tryAgainBtn');
const retryBtn = document.getElementById('retryBtn');
const modelSelection = document.getElementById('modelSelection');
const selectedModelName = document.getElementById('selectedModelName');

// Emotion icons mapping
const emotionIcons = {
    'angry': '😠',
    'calm': '😌',
    'disgust': '🤢',
    'fearful': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprised': '😲'
};

// File upload functionality removed - only recording is supported

// Recording functionality
recordButton.addEventListener('click', async () => {
    if (!audioRecorder.isCurrentlyRecording()) {
        try {
            console.log('Starting recording...');
            const success = await audioRecorder.startRecording();
            if (success) {
                startRecordingUI();
            } else {
                showError('Failed to access microphone. Please check permissions.');
            }
        } catch (error) {
            console.error('Recording error:', error);
            showError(error.message);
        }
    } else {
        console.log('Stopping recording...');
        audioRecorder.stopRecording();
        stopRecordingUI();
    }
});

analyzeRecordedBtn.addEventListener('click', async () => {
    const recordedAudio = audioRecorder.getRecordedAudio();
    if (!recordedAudio) return;

    showLoading();

    try {
        const response = await fetch('/record', {
            method: 'POST',
            body: recordedAudio,
            headers: {
                'Content-Type': recordedAudio.type || 'audio/webm'
            }
        });

        const result = await response.json();
        
        if (result.success) {
            showResults(result.emotion, result.confidence, recordedAudio);
        } else {
            showError(result.error || 'Failed to analyze recorded audio');
        }
    } catch (error) {
        console.error('❌ Network error:', error);
        
        if (error.name === 'TimeoutError') {
            showError('Request timed out. The server may be processing a large audio file. Please try again.');
        } else if (error.name === 'AbortError') {
            showError('Request was cancelled. Please try again.');
        } else if (error.message.includes('ERR_CONNECTION_RESET')) {
            showError('Connection was reset by the server. This might be due to audio processing issues. Please try with a shorter recording.');
        } else if (error.message.includes('Failed to fetch')) {
            showError('Failed to connect to server. Please check your connection and try again.');
        } else {
            showError('Network error: ' + error.message);
        }
    }
});

// UI Functions
function startRecordingUI() {
    recordButton.classList.add('recording', 'bg-red-600');
    recordButton.classList.remove('bg-red-500');
    recordStatus.textContent = 'Recording... Click to stop';
    analyzeRecordedBtn.disabled = true;
}

function stopRecordingUI() {
    recordButton.classList.remove('recording', 'bg-red-600');
    recordButton.classList.add('bg-red-500');
    recordStatus.textContent = 'Recording complete! Click analyze to process';
    analyzeRecordedBtn.disabled = false;
}

function showLoading() {
    resultsSection.classList.remove('hidden');
    loadingSpinner.classList.remove('hidden');
    resultsContent.classList.add('hidden');
    errorMessage.classList.add('hidden');
}

function showResults(emotion, confidence, audioBlob = null, modelType = null) {
    loadingSpinner.classList.add('hidden');
    resultsContent.classList.remove('hidden');
    
    const emotionIcon = document.getElementById('emotionIcon');
    const detectedEmotion = document.getElementById('detectedEmotion');
    const confidenceScore = document.getElementById('confidenceScore');
    const confidenceBar = document.getElementById('confidenceBar');
    
    emotionIcon.textContent = emotionIcons[emotion] || '😐';
    detectedEmotion.textContent = emotion.charAt(0).toUpperCase() + emotion.slice(1);
    confidenceScore.textContent = Math.round(confidence * 100) + '%';
    confidenceBar.style.width = (confidence * 100) + '%';
    
    // Add model type info if available
    if (modelType) {
        const existingModelInfo = document.getElementById('modelInfo');
        if (existingModelInfo) {
            existingModelInfo.remove();
        }
        const modelInfo = document.createElement('div');
        modelInfo.id = 'modelInfo';
        modelInfo.className = 'text-xs text-gray-500 mt-2';
        modelInfo.textContent = `Model: ${modelType}`;
        confidenceScore.parentNode.appendChild(modelInfo);
    }
    
    // Add audio playback if available
    if (audioBlob) {
        addAudioPlayback(audioBlob);
    }
}

function addAudioPlayback(audioBlob) {
    // Remove existing audio player if any
    const existingPlayer = document.getElementById('audioPlayer');
    if (existingPlayer) {
        existingPlayer.remove();
    }
    
    // Create audio player
    const audioPlayerDiv = document.createElement('div');
    audioPlayerDiv.id = 'audioPlayer';
    audioPlayerDiv.className = 'mt-6 p-4 bg-gray-50 rounded-lg';
    
    const audioUrl = URL.createObjectURL(audioBlob);
    const fileSizeKB = Math.round(audioBlob.size / 1024);
    
    audioPlayerDiv.innerHTML = `
        <h4 class="text-lg font-semibold text-gray-700 mb-3 flex items-center">
            🎵 Listen to Your Recording
        </h4>
        <audio controls class="w-full" preload="metadata" id="resultAudio">
            <source src="${audioUrl}" type="${audioBlob.type}">
            Your browser does not support the audio element.
        </audio>
        <div class="flex justify-between items-center mt-2 text-sm text-gray-500">
            <span>File size: ${fileSizeKB} KB</span>
            <span id="audioDuration">Duration: --</span>
        </div>
        <p class="text-sm text-gray-500 mt-1">Click play to hear your recorded audio</p>
    `;
    
    // Insert after the confidence section
    const confidenceSection = document.querySelector('.space-y-4');
    confidenceSection.appendChild(audioPlayerDiv);
    
    // Get duration when metadata loads
    const audioElement = document.getElementById('resultAudio');
    audioElement.addEventListener('loadedmetadata', () => {
        const duration = audioElement.duration;
        
        // Check if duration is valid (not NaN or Infinity)
        if (isFinite(duration) && duration > 0) {
            const minutes = Math.floor(duration / 60);
            const seconds = Math.floor(duration % 60);
            document.getElementById('audioDuration').textContent = 
                `Duration: ${minutes}:${seconds.toString().padStart(2, '0')}`;
        } else {
            // Fallback: estimate duration based on file size
            const estimatedDuration = Math.round(audioBlob.size / 16000); // Rough estimate: 16KB per second
            const minutes = Math.floor(estimatedDuration / 60);
            const seconds = Math.floor(estimatedDuration % 60);
            document.getElementById('audioDuration').textContent = 
                `Duration: ~${minutes}:${seconds.toString().padStart(2, '0')} (estimated)`;
        }
    });
    
    // Fallback if loadedmetadata doesn't fire
    audioElement.addEventListener('canplay', () => {
        const durationElement = document.getElementById('audioDuration');
        if (durationElement.textContent === 'Duration: --') {
            const duration = audioElement.duration;
            if (isFinite(duration) && duration > 0) {
                const minutes = Math.floor(duration / 60);
                const seconds = Math.floor(duration % 60);
                durationElement.textContent = 
                    `Duration: ${minutes}:${seconds.toString().padStart(2, '0')}`;
            } else {
                // Final fallback: show file size instead
                durationElement.textContent = `File size: ${fileSizeKB} KB`;
            }
        }
    });
    
    // Clean up the URL when the page is unloaded
    window.addEventListener('beforeunload', () => {
        URL.revokeObjectURL(audioUrl);
    });
}

function showError(message) {
    loadingSpinner.classList.add('hidden');
    errorMessage.classList.remove('hidden');
    document.getElementById('errorText').textContent = message;
}

// Reset functionality
tryAgainBtn.addEventListener('click', () => {
    resetUI();
});

retryBtn.addEventListener('click', () => {
    resetUI();
});

function resetUI() {
    resultsSection.classList.add('hidden');
    analyzeRecordedBtn.disabled = true;
    recordStatus.textContent = 'Click to start recording';
    recordButton.classList.remove('recording', 'bg-red-600');
    recordButton.classList.add('bg-red-500');
    audioRecorder.recordedAudio = null;
    
    // Clean up audio player
    const audioPlayer = document.getElementById('audioPlayer');
    if (audioPlayer) {
        audioPlayer.remove();
    }
}

// Check browser compatibility
function checkBrowserCompatibility() {
    const hasGetUserMedia = !!(navigator.mediaDevices?.getUserMedia || 
                              navigator.getUserMedia || 
                              navigator.webkitGetUserMedia || 
                              navigator.mozGetUserMedia);
    
    const hasMediaRecorder = !!window.MediaRecorder;
    const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    const isHTTPS = location.protocol === 'https:';
    
    console.log('Browser compatibility check:');
    console.log('- Has getUserMedia:', hasGetUserMedia);
    console.log('- Has MediaRecorder:', hasMediaRecorder);
    console.log('- Is mobile:', isMobile);
    console.log('- Is HTTPS:', isHTTPS);
    
    if (!hasGetUserMedia || !hasMediaRecorder) {
        document.getElementById('browserWarning').classList.remove('hidden');
        recordButton.disabled = true;
        recordButton.classList.add('opacity-50', 'cursor-not-allowed');
        recordStatus.textContent = 'Browser not supported for recording';
    } else if (!isHTTPS) {
        document.getElementById('mobileWarning').classList.remove('hidden');
        recordButton.disabled = true;
        recordButton.classList.add('opacity-50', 'cursor-not-allowed');
        recordStatus.textContent = 'Chrome requires HTTPS for microphone access - use https://192.168.0.110:4000';
    }
}

// Model Management
async function loadAvailableModels() {
    try {
        console.log('Loading models...');
        console.log('Model selection element:', modelSelection);
        
        if (!modelSelection) {
            console.error('Model selection element not found!');
            return;
        }
        
        const response = await fetch('/models');
        const data = await response.json();
        console.log('Models data:', data);
        displayModels(data.models);
    } catch (error) {
        console.error('Error loading models:', error);
    }
}

function displayModels(models) {
    console.log('Displaying models:', models);
    console.log('Model selection element:', modelSelection);
    
    if (!modelSelection) {
        console.error('Model selection element is null!');
        return;
    }
    
    modelSelection.innerHTML = '';
    
    models.forEach(model => {
        const modelCard = document.createElement('div');
        modelCard.className = `p-4 border-2 rounded-lg cursor-pointer transition-all ${
            selectedModel === model.id 
                ? 'border-blue-500 bg-blue-50' 
                : 'border-gray-200 hover:border-gray-300'
        }`;
        modelCard.onclick = () => selectModel(model);
        
        const accuracyColor = {
            'Low': 'text-red-600',
            'Medium': 'text-yellow-600',
            'High': 'text-green-600'
        }[model.accuracy] || 'text-gray-600';
        
        modelCard.innerHTML = `
            <h3 class="font-semibold text-gray-800 mb-2">${model.name}</h3>
            <p class="text-sm text-gray-600 mb-2">${model.description}</p>
            <div class="flex justify-between text-xs">
                <span class="${accuracyColor}">Accuracy: ${model.accuracy}</span>
                <span class="text-gray-500">Speed: ${model.speed}</span>
            </div>
        `;
        
        modelSelection.appendChild(modelCard);
    });
}

async function selectModel(model) {
    selectedModel = model.id;
    selectedModelName.textContent = model.name;
    const response = await fetch('/models');
    const data = await response.json();
    displayModels(data.models);
}

// Update record function to use selected model
const originalRecordHandler = analyzeRecordedBtn.onclick;
analyzeRecordedBtn.onclick = async () => {
    const recordedAudio = audioRecorder.getRecordedAudio();
    if (!recordedAudio) return;

    showLoading();

    try {
        console.log('📤 Sending audio data:', recordedAudio.size, 'bytes');
        console.log('📤 Audio type:', recordedAudio.type);
        console.log('📤 Model type:', selectedModel);
        
        const response = await fetch('/record', {
            method: 'POST',
            body: recordedAudio,
            headers: {
                'Content-Type': recordedAudio.type || 'audio/webm',
                'X-Model-Type': selectedModel
            },
            // Add timeout and better error handling
            signal: AbortSignal.timeout(30000) // 30 second timeout
        });

        console.log('📥 Response received:', response.status, response.statusText);
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status} ${response.statusText}`);
        }

        const result = await response.json();
        console.log('📥 Result:', result);
        
        if (result.success) {
            showResults(result.emotion, result.confidence, recordedAudio, result.model_type);
        } else {
            showError(result.error || 'Failed to analyze recorded audio');
        }
    } catch (error) {
        showError('Network error: ' + error.message);
    }
};

// Initialize UI
resetUI();
checkBrowserCompatibility();
loadAvailableModels();
