# EchoNotes Frontend

A beautiful, responsive web interface for the EchoNotes speech-to-notes system.

## Features

- 🎤 **Audio Recording** - Record directly from your microphone
- 📁 **File Upload** - Upload audio files (WAV, MP3, OGG, WebM, M4A)
- 📝 **Text Input** - Paste or upload transcript text files (.txt)
- 🤖 **AI Enhancement** - Toggle AI-powered content generation
- 📄 **Multiple Formats** - Export as HTML, Markdown, PDF, DOCX, or TXT
- 🎨 **Beautiful UI** - Modern, responsive design with animations

## Quick Start

### 1. Start the API Server

```bash
cd echonotes
python run_api.py --port 8000
```

### 2. Start the Frontend Server

```bash
python run_frontend.py --port 3000
```

### 3. Open in Browser

Navigate to: http://localhost:3000

## Usage

### Recording Audio

1. Click the microphone button to start recording
2. Speak your lecture/meeting content
3. Click the stop button when done
4. Click "Process" to convert to notes

### Uploading Files

**Audio Files:**
- Drag & drop or click to upload
- Supported: WAV, MP3, OGG, WebM, M4A

**Text Files:**
- Switch to "Text / Transcript" tab
- Paste text or upload .txt file
- Click "Process with AI"

### Settings

- **Document Title** - Name for your output document
- **Output Format** - HTML (recommended), Markdown, PDF, DOCX, or TXT
- **AI Enhancement** - Enable for simplified explanations, examples, FAQ, etc.

## Configuration

### API URL

If your API server is running on a different port or host, edit `index.html`:

```javascript
const API_BASE_URL = 'http://localhost:8000';  // Change this
```

## Technology Stack

- **React 18** - UI framework
- **Tailwind CSS** - Styling
- **Web Audio API** - Audio recording
- **Fetch API** - HTTP requests

## File Structure

```
frontend/
├── index.html      # Main application (single-file React app)
└── README.md       # This file

run_frontend.py     # Simple Python HTTP server
```

## Browser Support

- Chrome (recommended)
- Firefox
- Safari
- Edge

**Note:** Microphone access requires HTTPS in production or localhost for development.

## Screenshots

### Recording Interface
- Large microphone button with pulse animation
- Real-time duration counter
- Audio waveform visualization

### File Upload
- Drag & drop zone
- File type indicators
- Progress feedback

### Processing
- Animated progress bar
- Status messages
- Loading indicators

### Results
- Transcript preview
- Key concepts badges
- Download button

## Troubleshooting

### "API server not reachable"
- Make sure the API server is running: `python run_api.py`
- Check the port matches (default: 8000)

### "Microphone access denied"
- Allow microphone permission in browser
- Use HTTPS or localhost

### Recording not working
- Check browser console for errors
- Try a different browser
- Ensure microphone is connected

## License

MIT License - Feel free to use and modify!
