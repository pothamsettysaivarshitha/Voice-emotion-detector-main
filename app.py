from flask import Flask, render_template, request, jsonify
import os
import tempfile
import soundfile as sf
import librosa
import uuid
import time
from werkzeug.utils import secure_filename
from real_emotion_detector import get_detector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('models', exist_ok=True)

def predict_emotion(audio_file_path, model_type='rule_based'):
    """Predict emotion from audio file using specified model"""
    try:
        # Check if file exists
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
        detector = get_detector(model_type)
        emotion, confidence = detector.predict(audio_file_path)
        return emotion, confidence
    except Exception as e:
        print(f"Error predicting emotion: {e}")
        # Re-raise the exception so the upload handler can handle it properly
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test():
    return app.send_static_file('test.html')

@app.route('/test-record', methods=['POST'])
def test_record():
    """Simple test endpoint to verify POST requests work"""
    print("🧪 Test endpoint called")
    try:
        data = request.get_data()
        print(f"📊 Received {len(data)} bytes")
        return jsonify({'success': True, 'message': 'Test endpoint working', 'data_size': len(data)})
    except Exception as e:
        print(f"❌ Test endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-models')
def test_models():
    return app.send_static_file('test_models.html')

@app.route('/models')
def get_available_models():
    """Get list of available emotion detection models"""
    models = [
        {
            'id': 'rule_based',
            'name': 'Rule-Based',
            'description': 'Simple rule-based emotion detection',
            'accuracy': 'Low',
            'speed': 'Fast'
        },
        {
            'id': 'svm',
            'name': 'Support Vector Machine',
            'description': 'Traditional ML model with SVM',
            'accuracy': 'Medium',
            'speed': 'Fast'
        },
        {
            'id': 'random_forest',
            'name': 'Random Forest',
            'description': 'Ensemble learning with Random Forest',
            'accuracy': 'Medium',
            'speed': 'Fast'
        },
        {
            'id': 'huggingface',
            'name': 'RAVDESS Pre-trained',
            'description': 'Real Wav2Vec2 model trained on RAVDESS dataset (84.84% accuracy)',
            'accuracy': 'High',
            'speed': 'Medium'
        }
    ]
    
    # Update model descriptions based on availability
    try:
        from real_emotion_detector import TRANSFORMERS_AVAILABLE
        
        if TRANSFORMERS_AVAILABLE:
            # Update Hugging Face model description
            for model in models:
                if model['id'] == 'huggingface':
                    model['description'] = 'Real pre-trained emotion detection model from Hugging Face'
                    model['accuracy'] = 'High'
                    break
        else:
            # Remove Hugging Face model if transformers not available
            models = [model for model in models if model['id'] != 'huggingface']
            
    except ImportError:
        # Remove Hugging Face model if import fails
        models = [model for model in models if model['id'] != 'huggingface']
    
    return jsonify({'models': models})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get model type from request (default to rule_based)
    model_type = request.form.get('model_type', 'rule_based')
    
    if file:
        # Debug: Print file information before saving
        print(f"🔍 Original filename: {file.filename}")
        print(f"🔍 File content type: {file.content_type}")
        
        # Check if file has content without consuming it
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        print(f"🔍 File size: {file_size} bytes")
        
        # Additional file object debugging
        print(f"🔍 File object type: {type(file)}")
        print(f"🔍 File object has filename: {hasattr(file, 'filename')}")
        print(f"🔍 File object has save method: {hasattr(file, 'save')}")
        print(f"🔍 File object readable: {file.readable()}")
        
        # Test reading a small amount to see if content is there
        file.seek(0)
        test_read = file.read(10)
        file.seek(0)  # Reset again
        print(f"🔍 First 10 bytes: {test_read}")
        print(f"🔍 Content length: {len(test_read)}")
        
        # Save uploaded file with unique filename to avoid collisions
        filename = secure_filename(file.filename)
        # Add timestamp and UUID to make filename unique
        unique_id = f"{int(time.time())}_{str(uuid.uuid4())[:8]}"
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_{unique_id}{ext}"
        file_path = os.path.join('uploads', unique_filename)
        
        print(f"🔍 Attempting to save to: {file_path}")
        print(f"🔍 Uploads directory exists: {os.path.exists('uploads')}")
        
        try:
            file.save(file_path)
            print(f"✅ File save() method completed!")
            
            # Verify the file was actually saved
            if not os.path.exists(file_path):
                print(f"❌ File save reported success but file doesn't exist!")
                return jsonify({'error': 'File upload failed - file not saved'}), 500
                
            saved_size = os.path.getsize(file_path)
            print(f"🔍 Saved file size: {saved_size} bytes")
            
            if saved_size == 0:
                print(f"❌ File saved but has zero size!")
                print(f"🔍 File path: {file_path}")
                print(f"🔍 Absolute path: {os.path.abspath(file_path)}")
                
                # Try to read the file content to see what's there
                try:
                    with open(file_path, 'rb') as f:
                        content = f.read()
                    print(f"🔍 File content length: {len(content)}")
                    print(f"🔍 File content: {content[:20]}...")
                except Exception as read_error:
                    print(f"🔍 Error reading saved file: {read_error}")
                
                return jsonify({'error': 'File upload failed - empty file'}), 500
                
            print(f"✅ File verification passed: {saved_size} bytes")
            
        except Exception as save_error:
            print(f"❌ Error saving file: {save_error}")
            return jsonify({'error': f'Failed to save uploaded file: {str(save_error)}'}), 500
        
        # Convert to absolute path to avoid working directory issues
        file_path = os.path.abspath(file_path)
        
        try:
            # Debug: Print file information after saving
            print(f"🔍 Upload file saved to: {file_path}")
            print(f"🔍 File exists: {os.path.exists(file_path)}")
            print(f"🔍 File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'} bytes")
            print(f"🔍 Model type: {model_type}")
            
            # Predict emotion using specified model
            emotion, confidence = predict_emotion(file_path, model_type)
            
            # Clean up uploaded file
            print(f"🧹 Cleaning up file: {file_path}")
            if os.path.exists(file_path):
                print(f"✅ File exists before cleanup, removing...")
                os.remove(file_path)
                print(f"✅ File removed successfully")
            else:
                print(f"⚠️ File doesn't exist during cleanup")
            
            return jsonify({
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'model_type': model_type
            })
            
        except Exception as e:
            # Clean up uploaded file even if prediction fails
            print(f"❌ Analysis failed with error: {e}")
            print(f"🧹 Cleaning up file after error: {file_path}")
            if os.path.exists(file_path):
                print(f"✅ File exists before error cleanup, removing...")
                os.remove(file_path)
                print(f"✅ File removed after error")
            else:
                print(f"⚠️ File doesn't exist during error cleanup")
            return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/record', methods=['POST'])
def record_audio():
    """Handle recorded audio from frontend"""
    print("🎤 /record endpoint called")
    try:
        # Get audio data from request
        audio_data = request.get_data()
        print(f"📊 Received audio data: {len(audio_data)} bytes")
        
        # Get model type from request headers (default to rule_based)
        model_type = request.headers.get('X-Model-Type', 'rule_based')
        print(f"🤖 Model type: {model_type}")
        
        if not audio_data:
            print("❌ No audio data received")
            return jsonify({'error': 'No audio data received'}), 400
        
        # Determine file extension based on content type
        content_type = request.headers.get('Content-Type', 'audio/webm')
        print(f"📁 Content type: {content_type}")
        
        if 'webm' in content_type:
            suffix = '.webm'
        elif 'mp4' in content_type:
            suffix = '.mp4'
        elif 'wav' in content_type:
            suffix = '.wav'
        else:
            suffix = '.webm'  # Default fallback
        
        print(f"💾 Using file suffix: {suffix}")
        
        # Save temporary audio file
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            print(f"✅ Saved temp file: {tmp_file_path}")
        except Exception as save_error:
            print(f"❌ Error saving temp file: {save_error}")
            return jsonify({'error': f'Failed to save audio data: {str(save_error)}'}), 500
        
        # Convert to WAV for processing
        wav_file_path = tmp_file_path.replace(suffix, '.wav')
        
        # Convert to absolute paths to avoid working directory issues
        tmp_file_path = os.path.abspath(tmp_file_path)
        wav_file_path = os.path.abspath(wav_file_path)
        
        print(f"🔄 Converting audio: {tmp_file_path} -> {wav_file_path}")
        
        try:
            # For WebM files, use FFmpeg for proper conversion
            if suffix == '.webm':
                print("📚 WebM detected - using FFmpeg for proper conversion...")
                
                try:
                    import subprocess
                    import shutil
                    
                    # Find FFmpeg executable
                    ffmpeg_exe = shutil.which('ffmpeg')
                    if not ffmpeg_exe:
                        # Try common installation paths
                        possible_paths = [
                            r'C:\Users\Madhu\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe',
                            r'C:\Users\Madhu\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg.exe',
                            r'C:\ffmpeg\bin\ffmpeg.exe',
                            r'C:\Program Files\ffmpeg\bin\ffmpeg.exe',
                            r'C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe'
                        ]
                        
                        for path in possible_paths:
                            if os.path.exists(path):
                                ffmpeg_exe = path
                                break
                    
                    if not ffmpeg_exe:
                        raise Exception("FFmpeg not found in PATH or common locations")
                    
                    print(f"🔍 Using FFmpeg at: {ffmpeg_exe}")
                    
                    # Convert using FFmpeg with optimal settings for emotion detection
                    ffmpeg_cmd = [
                        ffmpeg_exe, '-i', tmp_file_path, 
                        '-ar', '22050',        # Sample rate: 22050 Hz
                        '-ac', '1',            # Mono channel
                        '-acodec', 'pcm_s16le', # PCM 16-bit encoding
                        '-y',                  # Overwrite output file
                        wav_file_path
                    ]
                    
                    print(f"🔄 Running FFmpeg: {' '.join(ffmpeg_cmd)}")
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
                    
                    if result.returncode == 0:
                        print("✅ FFmpeg conversion successful")
                        # Verify the file was created and has content
                        if os.path.exists(wav_file_path) and os.path.getsize(wav_file_path) > 0:
                            file_size = os.path.getsize(wav_file_path)
                            print(f"✅ WAV file created: {wav_file_path} ({file_size} bytes)")
                        else:
                            raise Exception("FFmpeg created empty file")
                    else:
                        print(f"❌ FFmpeg stderr: {result.stderr}")
                        raise Exception(f"FFmpeg conversion failed: {result.stderr}")
                        
                except Exception as ffmpeg_error:
                    print(f"❌ FFmpeg conversion failed: {ffmpeg_error}")
                    print("📚 Falling back to bypass method...")
                    
                    # Fallback: Copy WebM file as WAV (original bypass method)
                    import shutil
                    shutil.copy2(tmp_file_path, wav_file_path)
                    print(f"✅ Copied WebM as WAV (fallback): {wav_file_path} ({os.path.getsize(wav_file_path)} bytes)")
                    print("⚠️ Note: Using WebM file directly - some models may not work properly")
            
            else:
                # For non-WebM files, try librosa methods
                print("📚 Non-WebM file - attempting librosa conversion...")
                
                # Method 1: Try librosa with different parameters
                try:
                    print("📚 Method 1: Loading with librosa (sr=22050)...")
                    y, sr = librosa.load(tmp_file_path, sr=22050)
                    print(f"🎵 Loaded audio: {len(y)} samples at {sr} Hz")
                    
                    print("💾 Writing WAV file...")
                    sf.write(wav_file_path, y, sr)
                    print(f"✅ WAV file written: {wav_file_path}")
                    
                except Exception as librosa_error:
                    print(f"⚠️ Librosa failed: {librosa_error}")
                    
                    # Method 2: Try librosa with original sample rate
                    try:
                        print("📚 Method 2: Loading with librosa (original sr)...")
                        y, sr = librosa.load(tmp_file_path)
                        print(f"🎵 Loaded audio: {len(y)} samples at {sr} Hz")
                        
                        # Resample if needed
                        if sr != 22050:
                            print(f"🔄 Resampling from {sr} to 22050...")
                            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=22050)
                            y, sr = y_resampled, 22050
                        
                        print("💾 Writing WAV file...")
                        sf.write(wav_file_path, y, sr)
                        print(f"✅ WAV file written: {wav_file_path}")
                        
                    except Exception as librosa_error2:
                        print(f"⚠️ Librosa method 2 failed: {librosa_error2}")
                        
                        # Method 3: Try with ffmpeg as fallback
                        try:
                            print("📚 Method 3: Using ffmpeg conversion...")
                            import subprocess
                            
                            # Convert using ffmpeg
                            ffmpeg_cmd = [
                                'ffmpeg', '-i', tmp_file_path, 
                                '-ar', '22050',  # Sample rate
                                '-ac', '1',      # Mono
                                '-y',            # Overwrite output
                                wav_file_path
                            ]
                            
                            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
                            
                            if result.returncode == 0:
                                print("✅ FFmpeg conversion successful")
                                # Verify the file was created
                                if os.path.exists(wav_file_path) and os.path.getsize(wav_file_path) > 0:
                                    print(f"✅ WAV file created: {wav_file_path}")
                                else:
                                    raise Exception("FFmpeg created empty file")
                            else:
                                raise Exception(f"FFmpeg failed: {result.stderr}")
                                
                        except Exception as ffmpeg_error:
                            print(f"⚠️ FFmpeg method failed: {ffmpeg_error}")
                            
                            # Method 4: Skip conversion, try to use original file
                            print("📚 Method 4: Using original file without conversion...")
                            if suffix == '.wav':
                                # If it's already WAV, just copy it
                                import shutil
                                shutil.copy2(tmp_file_path, wav_file_path)
                                print(f"✅ Copied WAV file: {wav_file_path}")
                            else:
                                raise Exception("All conversion methods failed")
            
            # Predict emotion using specified model with error handling
            try:
                print(f"🧠 Predicting emotion with {model_type}...")
                emotion, confidence = predict_emotion(wav_file_path, model_type)
                print(f"🎯 Result: {emotion} (confidence: {confidence})")
            except Exception as e:
                print(f"❌ Emotion prediction error: {e}")
                # Clean up files before returning error
                try:
                    os.unlink(tmp_file_path)
                    if os.path.exists(wav_file_path):
                        os.unlink(wav_file_path)
                except:
                    pass
                return jsonify({'error': f'Emotion analysis failed: {str(e)}'}), 500
            
            # Clean up temporary files
            try:
                os.unlink(tmp_file_path)
                if os.path.exists(wav_file_path):
                    os.unlink(wav_file_path)
                print("🧹 Cleaned up temp files")
            except Exception as cleanup_error:
                print(f"⚠️ Cleanup warning: {cleanup_error}")
            
            print("✅ Returning successful result")
            return jsonify({
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'model_type': model_type
            })
                
        except Exception as conversion_error:
            print(f"❌ Audio conversion error: {conversion_error}")
            # Clean up files
            try:
                os.unlink(tmp_file_path)
                if os.path.exists(wav_file_path):
                    os.unlink(wav_file_path)
            except:
                pass
            return jsonify({'error': f'Audio conversion failed: {str(conversion_error)}'}), 500
            
    except Exception as e:
        print(f"❌ Server error in /record: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    # Chrome requires HTTPS for microphone access (even on desktop)
    # Use Flask's adhoc SSL context for development
    print("Starting Flask app...")
    print("Running with HTTPS on https://192.168.0.110:4000")
    print("Note: You may see a certificate warning - click 'Advanced' then 'Proceed to site'")
    print("This is safe for development purposes")
    print("Debug mode disabled to prevent auto-reload during audio processing")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=4000, ssl_context='adhoc')
    except Exception as e:
        print(f"HTTPS failed: {e}")
        print("Falling back to HTTP (microphone will not work)")
        app.run(debug=False, host='0.0.0.0', port=4000)
