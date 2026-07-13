class AudioRecorder {
    constructor() {
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.recordedAudio = null;
    }

    async startRecording() {
        try {
            console.log('Requesting microphone access...');
            console.log('Navigator:', navigator.userAgent);
            console.log('MediaDevices:', !!navigator.mediaDevices);
            console.log('getUserMedia:', !!navigator.mediaDevices?.getUserMedia);
            console.log('navigator.getUserMedia:', !!navigator.getUserMedia);
            console.log('navigator.webkitGetUserMedia:', !!navigator.webkitGetUserMedia);
            console.log('navigator.mozGetUserMedia:', !!navigator.mozGetUserMedia);
            console.log('MediaRecorder:', !!window.MediaRecorder);
            
            // Check if getUserMedia is supported with fallbacks
            let getUserMedia;
            
            // Check for mobile-specific issues
            const isMobile = /Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
            const isHTTPS = location.protocol === 'https:';
            
            console.log('Is mobile:', isMobile);
            console.log('Is HTTPS:', isHTTPS);
            
            if (!isHTTPS && isMobile) {
                throw new Error('Mobile browsers require HTTPS for microphone access. Please access this site via HTTPS (https://localhost:4000)');
            }
            
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                console.log('Using navigator.mediaDevices.getUserMedia');
                getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
            } else if (navigator.getUserMedia) {
                console.log('Using navigator.getUserMedia');
                // Fallback for older browsers
                getUserMedia = (constraints) => {
                    return new Promise((resolve, reject) => {
                        navigator.getUserMedia(constraints, resolve, reject);
                    });
                };
            } else if (navigator.webkitGetUserMedia) {
                console.log('Using navigator.webkitGetUserMedia');
                // Fallback for WebKit browsers
                getUserMedia = (constraints) => {
                    return new Promise((resolve, reject) => {
                        navigator.webkitGetUserMedia(constraints, resolve, reject);
                    });
                };
            } else if (navigator.mozGetUserMedia) {
                console.log('Using navigator.mozGetUserMedia');
                // Fallback for Firefox
                getUserMedia = (constraints) => {
                    return new Promise((resolve, reject) => {
                        navigator.mozGetUserMedia(constraints, resolve, reject);
                    });
                };
            } else {
                console.error('No getUserMedia support found');
                if (isMobile && !isHTTPS) {
                    throw new Error('Mobile browsers require HTTPS for microphone access. Please access this site via HTTPS (https://localhost:4000)');
                } else {
                    throw new Error('getUserMedia is not supported in this browser. Please use a modern browser like Chrome, Firefox, Safari, or Edge.');
                }
            }
            
            // Try with advanced constraints first, fallback to basic if needed
            let stream;
            try {
                stream = await getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 44100
                    } 
                });
            } catch (constraintError) {
                console.log('Advanced constraints failed, trying basic audio:', constraintError);
                // Fallback to basic audio constraints
                stream = await getUserMedia({ audio: true });
            }
            
            console.log('Microphone access granted');
            
            // Try different MediaRecorder options for better compatibility
            let mediaRecorderOptions = {};
            
            // Check for supported MIME types
            if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
                mediaRecorderOptions.mimeType = 'audio/webm;codecs=opus';
            } else if (MediaRecorder.isTypeSupported('audio/webm')) {
                mediaRecorderOptions.mimeType = 'audio/webm';
            } else if (MediaRecorder.isTypeSupported('audio/mp4')) {
                mediaRecorderOptions.mimeType = 'audio/mp4';
            } else if (MediaRecorder.isTypeSupported('audio/wav')) {
                mediaRecorderOptions.mimeType = 'audio/wav';
            }
            
            console.log('Using MediaRecorder options:', mediaRecorderOptions);
            
            this.mediaRecorder = new MediaRecorder(stream, mediaRecorderOptions);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                console.log('Data available:', event.data.size, 'bytes');
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                console.log('Recording stopped, creating audio blob...');
                
                // Determine the correct MIME type for the blob
                let blobType = 'audio/webm';
                if (mediaRecorderOptions.mimeType) {
                    blobType = mediaRecorderOptions.mimeType;
                }
                
                const audioBlob = new Blob(this.audioChunks, { type: blobType });
                this.recordedAudio = audioBlob;
                console.log('Audio blob created:', audioBlob.size, 'bytes', 'type:', blobType);
                
                // Stop all tracks to release microphone
                stream.getTracks().forEach(track => track.stop());
            };

            this.mediaRecorder.onerror = (event) => {
                console.error('MediaRecorder error:', event.error);
            };

            this.mediaRecorder.start(100); // Collect data every 100ms
            this.isRecording = true;
            console.log('Recording started');
            return true;
        } catch (error) {
            console.error('Error starting recording:', error);
            
            // Provide specific error messages
            if (error.name === 'NotAllowedError') {
                throw new Error('Microphone access denied. Please allow microphone access and try again.');
            } else if (error.name === 'NotFoundError') {
                throw new Error('No microphone found. Please connect a microphone and try again.');
            } else if (error.name === 'NotSupportedError') {
                throw new Error('Microphone access not supported. Please use HTTPS or a modern browser.');
            } else {
                throw new Error('Failed to access microphone: ' + error.message);
            }
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            return true;
        }
        return false;
    }

    getRecordedAudio() {
        return this.recordedAudio;
    }

    isCurrentlyRecording() {
        return this.isRecording;
    }
}
